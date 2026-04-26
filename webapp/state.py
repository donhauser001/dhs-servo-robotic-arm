"""ArmService：把 ``arm.Arm`` 包装成线程/异步安全的服务对象。

设计要点：
    - 串口总线是单线程独占的，所以所有 ``arm.*`` 调用都加 :class:`asyncio.Lock`
    - 阻塞 I/O 走 ``run_in_executor``（避免 block FastAPI 事件循环）
    - 启动时一次性 open 总线；连接失败时让 ``ready=False``，路由返回 503
    - 联锁状态评估：基于最近一次位置快照，告诉前端哪些关节当前被锁
"""

from __future__ import annotations

import asyncio
import logging
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from arm.arm import Arm, ArmConfig, ArmState, JointConfig, JointConstraint
from arm.servo import Feedback

log = logging.getLogger(__name__)


class ArmService:
    """全应用共享的单例。FastAPI 启动时构造一次。"""

    def __init__(self, config_path: Path):
        self.config_path = Path(config_path)
        self._lock = asyncio.Lock()
        self._arm: Optional[Arm] = None
        self._cfg: Optional[ArmConfig] = None
        self._ready = False
        self._error: Optional[str] = None
        self._open_lock = threading.Lock()  # config 重载时短暂同步保护

    # ---- 生命周期 ----

    async def startup(self) -> None:
        """异步启动：加载 yaml + open 总线。失败也不抛，只记录 _error。"""
        try:
            await asyncio.get_running_loop().run_in_executor(None, self._open)
            self._ready = True
            self._error = None
            log.info("ArmService ready: %s", self.config_path)
        except Exception as e:
            self._ready = False
            self._error = f"{type(e).__name__}: {e}"
            log.exception("ArmService 启动失败")

    def _open(self) -> None:
        with self._open_lock:
            self._cfg = ArmConfig.from_yaml(self.config_path)
            self._arm = Arm(self._cfg)
            self._arm.open()

    async def shutdown(self) -> None:
        if self._arm is not None:
            try:
                await self._run(lambda: self._arm.set_torque(False))
            except Exception:
                log.exception("关扭矩失败（已忽略）")
            try:
                await self._run(self._arm.close)
            except Exception:
                log.exception("close 总线失败（已忽略）")
            self._arm = None
            self._ready = False

    async def reload_config(self) -> None:
        """重读 yaml，然后重建 Arm（不动总线本身的开关，只换 cfg）。"""
        async with self._lock:
            await asyncio.get_running_loop().run_in_executor(None, self._reload)

    def _reload(self) -> None:
        new_cfg = ArmConfig.from_yaml(self.config_path)
        if self._arm is None:
            return
        old_arm = self._arm
        old_arm.config = new_cfg
        # JointConfig 列表如果数量/顺序变化，要重建 servos
        if len(new_cfg.joints) != len(old_arm.servos) or any(
            j.id != s.sid for j, s in zip(new_cfg.joints, old_arm.servos)
        ):
            from arm.servo import Servo

            old_arm.servos = [Servo(old_arm.bus, j.id) for j in new_cfg.joints]
        self._cfg = new_cfg

    # ---- 内部 helper ----

    async def _run(self, fn):
        """在线程池里跑阻塞函数；外层调用者应已持 self._lock。"""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, fn)

    def _check(self) -> Arm:
        if not self._ready or self._arm is None:
            raise RuntimeError(self._error or "ArmService 未就绪")
        return self._arm

    # ---- 状态查询 ----

    @property
    def ready(self) -> bool:
        return self._ready

    @property
    def error(self) -> Optional[str]:
        return self._error

    @property
    def cfg(self) -> Optional[ArmConfig]:
        return self._cfg

    async def snapshot(self) -> Dict[str, Any]:
        """读关节当前 raw 一帧，附带 pct / deg / 联锁状态。

        * 不读 temp/current（feedback 慢；前端按需用 ``feedback`` 接口）。
        """
        async with self._lock:
            arm = self._check()
            raws = await self._run(arm.read_joints_raw)
        return self._compose_state(raws)

    async def feedback_full(self) -> Dict[str, Any]:
        """读完整反馈（含 temp / current / load），慢一点。"""
        async with self._lock:
            arm = self._check()
            fbs: List[Feedback] = await self._run(arm.read_feedback)
        raws = [fb.position for fb in fbs]
        state = self._compose_state(raws)
        for ji, fb in zip(state["joints"], fbs):
            ji["temp_c"] = fb.temperature
            ji["current_ma"] = round(fb.current_ma, 1) if fb.current_ma is not None else None
            ji["voltage_v"] = round(fb.voltage, 2)
            ji["load_pct"] = round(fb.load / 10.0, 1)
            ji["moving"] = fb.moving
        return state

    def _compose_state(self, raws: Sequence[int]) -> Dict[str, Any]:
        arm = self._check()
        cfg = arm.config
        # 先逐关节构造 base 信息
        joints = []
        for j, r in zip(cfg.joints, raws):
            joints.append({
                "name": j.name,
                "id": j.id,
                "model": j.model,
                "raw": int(r),
                "pct": round(j.raw_to_pct(r), 4),
                "deg": (round(j.raw_to_deg(r), 2)
                        if j.raw_to_deg(r) is not None else None),
                "home_raw": j.home_raw,
                "max_raw": j.max_raw,
                "low_raw": j.low_raw,
                "min_deg": j.min_deg,
                "max_deg": j.max_deg,
                "home_deg": j.home_deg,
                "is_bidirectional": j.is_bidirectional,
                "locked": False,
                "lock_reason": None,
            })

        # 联锁评估
        name_to_idx = {j["name"]: i for i, j in enumerate(joints)}
        for c in cfg.constraints:
            ti = name_to_idx.get(c.target)
            gi = name_to_idx.get(c.gate)
            if ti is None or gi is None:
                continue
            gate_j = cfg.joints[gi]
            gate_deg = gate_j.raw_to_deg(raws[gi])
            if gate_deg is None:
                continue
            if c.op == "<":
                violated = gate_deg >= c.gate_min_deg
            else:
                violated = gate_deg <= c.gate_min_deg
            if violated:
                joints[ti]["locked"] = True
                joints[ti]["lock_reason"] = (
                    f"需 {c.gate} {c.op} {c.gate_min_deg}°（当前 {gate_deg:.1f}°）"
                )

        return {
            "ready": self._ready,
            "error": self._error,
            "state": arm.state.value,
            "joints": joints,
            "poses": dict(cfg.poses),
            "constraints": [
                {
                    "target": c.target,
                    "gate": c.gate,
                    "gate_min_deg": c.gate_min_deg,
                    "op": c.op,
                    "enabled": c.enabled,
                    "note": c.note,
                }
                for c in cfg.constraints
            ],
            "prerequisites": [
                {
                    "target": p.target,
                    "target_op": p.target_op,
                    "target_deg": p.target_deg,
                    "gate": p.gate,
                    "gate_op": p.gate_op,
                    "gate_deg": p.gate_deg,
                    "buffer_deg": p.buffer_deg,
                    "enabled": p.enabled,
                    "note": p.note,
                }
                for p in cfg.prerequisites
            ],
            "zone_limits": [
                {
                    "target": z.target,
                    "target_op": z.target_op,
                    "target_deg": z.target_deg,
                    "gate": z.gate,
                    "gate_op": z.gate_op,
                    "gate_deg": z.gate_deg,
                    "enabled": z.enabled,
                    "note": z.note,
                }
                for z in cfg.zone_limits
            ],
            "interventions": list(arm.recent_interventions),
        }

    # ---- 控制 ----

    async def enable(self) -> None:
        async with self._lock:
            arm = self._check()
            await self._run(arm.enable)

    async def disable(self, to: Optional[str] = "safe") -> None:
        async with self._lock:
            arm = self._check()
            await self._run(lambda: arm.disable(to=to))

    async def emergency_stop(self) -> None:
        async with self._lock:
            arm = self._check()
            await self._run(arm.emergency_stop)

    async def set_torque(self, on: bool, joint: Optional[str] = None) -> None:
        async with self._lock:
            arm = self._check()
            if joint is None:
                await self._run(lambda: arm.set_torque(on))
            else:
                idx = self._joint_index(joint)
                await self._run(lambda: arm.servos[idx].torque(on))

    async def goto_pose(
        self, name: str, speed: int = 600, acc: int = 30, wait: bool = True,
    ) -> None:
        async with self._lock:
            arm = self._check()
            await self._run(
                lambda: arm.goto_pose(name, speed=speed, acc=acc, wait=wait, timeout=15.0)
            )

    async def move_home(self, speed: int = 600, acc: int = 30) -> None:
        async with self._lock:
            arm = self._check()
            await self._run(lambda: arm.move_home(speed=speed, acc=acc))

    async def move_pct(
        self, pcts: Sequence[float], speed: Optional[int] = None,
        acc: Optional[int] = None, wait: bool = False,
    ) -> None:
        async with self._lock:
            arm = self._check()
            await self._run(lambda: arm.move_pct(list(pcts), speed=speed, acc=acc))
            if wait:
                await self._run(lambda: arm.wait_until_idle(timeout=15.0))

    async def move_deg(
        self, degs: Sequence[float], speed: Optional[int] = None,
        acc: Optional[int] = None, wait: bool = False,
    ) -> None:
        async with self._lock:
            arm = self._check()
            await self._run(lambda: arm.move_deg(list(degs), speed=speed, acc=acc))
            if wait:
                await self._run(lambda: arm.wait_until_idle(timeout=15.0))

    async def move_raw(
        self, raws: Sequence[int], speed: Optional[int] = None,
        acc: Optional[int] = None, wait: bool = False, bypass_constraints: bool = False,
    ) -> None:
        async with self._lock:
            arm = self._check()
            n = len(arm.config.joints)
            speeds = [speed] * n if speed is not None else None
            accs = [acc] * n if acc is not None else None
            await self._run(lambda: arm.move_joints_raw(
                list(raws), speeds, accs, bypass_constraints=bypass_constraints,
            ))
            if wait:
                await self._run(lambda: arm.wait_until_idle(timeout=15.0))

    async def move_single_joint(
        self, name: str, *, pct: Optional[float] = None, deg: Optional[float] = None,
        raw: Optional[int] = None, speed: Optional[int] = None,
        acc: Optional[int] = None,
    ) -> None:
        """只动一个关节，其余目标 = 当前位置（受联锁影响）。"""
        async with self._lock:
            arm = self._check()
            idx = self._joint_index(name)
            j = arm.config.joints[idx]
            current = await self._run(arm.read_joints_raw)
            target = list(current)
            if pct is not None:
                target[idx] = j.pct_to_raw(float(pct))
            elif deg is not None:
                target[idx] = j.deg_to_raw(float(deg))
            elif raw is not None:
                target[idx] = int(raw)
            else:
                return
            n = len(arm.config.joints)
            speeds = [speed] * n if speed is not None else None
            accs = [acc] * n if acc is not None else None
            await self._run(lambda: arm.move_joints_raw(target, speeds, accs))

    # ---- 配置：标定 ----

    def _joint_index(self, name: str) -> int:
        if self._cfg is None:
            raise RuntimeError("config 未加载")
        for i, j in enumerate(self._cfg.joints):
            if j.name == name:
                return i
        raise KeyError(f"无此关节：{name}")

    async def capture_joint_value(self, name: str, field: str) -> int:
        """读当前 raw，作为 home_raw / max_raw / low_raw 之一（不写 yaml，仅返回）。"""
        async with self._lock:
            arm = self._check()
            idx = self._joint_index(name)
            cur = await self._run(arm.servos[idx].read_position)
        return int(cur)
