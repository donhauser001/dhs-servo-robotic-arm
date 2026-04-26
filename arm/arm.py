"""6 DOF 机械臂高层控制器。

负责：
    - 加载 YAML 配置（关节定义、两/三点标定、连杆几何）
    - 关节级映射：百分比 / 物理角度 ↔ 舵机原始位置
    - 同步读 / 同步写（一次操作多个关节）
    - 软件限位保护、紧急停止

核心抽象：**百分比是首选 API**。

每个关节按需标定 2 或 3 个点：
    home_raw → 0%   （初始位 / 上电默认位置）
    max_raw  → +100%（正向极限）
    low_raw  → -100%（负向极限，可选）

类型自动区分：
    单向关节（如抬升）：只配 home + max  →  pct 范围 [0, +1]，传负值会被夹到 0
    双向关节（如旋转）：配 home + max + low  →  pct 范围 [-1, +1]

方向由 max / low 在 home 哪一侧自然决定（数值大于或小于 home 都允许）。
应用层永远只看归一化值；超界天然不可达。

可选地额外标定 min_deg / max_deg，用于和真实物理角度互转（IK 用）。约定：
    单向：min_deg ↔ home（0%）   max_deg ↔ max（+100%）
    双向：min_deg ↔ low（-100%）  max_deg ↔ max（+100%）
"""

from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import yaml

from . import protocol as p
from .bus import Bus
from .protocol import Packet, Reg
from .servo import Feedback, Servo

log = logging.getLogger(__name__)


# 已知 JointConfig 字段，用于 from_yaml 时过滤未知键，方便 schema 渐进升级
_JOINT_FIELDS = {
    "name", "id", "model",
    "home_raw", "max_raw", "low_raw",
    "min_deg", "max_deg", "home_deg",
    "max_speed", "max_acc",
}


# ---------------------------------------------------------------------------
# 关节配置
# ---------------------------------------------------------------------------

@dataclass
class JointConfig:
    """单个关节的配置（与 config/arm_config.yaml 字段一一对应）。

    标定模型：
        home_raw → 0%
        max_raw  → +100%
        low_raw  → -100%（**可选**：有则双向，无则单向）

    方向由 ``max_raw`` / ``low_raw`` 在 ``home_raw`` 哪一侧自然决定，
    例如 ``home=4079, max=2700`` 也合法（pct 增大对应 raw 减小）。

    单向关节：``arm.move_pct(0.5)`` 等同于走到 50%；负值会被夹到 0。
    双向关节：``arm.move_pct(-0.5)`` 走到反方向一半；范围 [-1, +1]。
    """

    name: str
    id: int
    model: str = "ST3020"

    # 标定点（raw）
    home_raw: int = 2047
    max_raw: int = 4095
    low_raw: Optional[int] = None      # 配则双向；不配则单向

    # 可选：与物理角度的对应（仅 IK / 显示用）
    # 三点分段线性映射：
    #     双向：min_deg ↔ low_raw（-100%），home_deg ↔ home_raw（0%），
    #           max_deg ↔ max_raw（+100%）
    #     单向：home_deg ↔ home_raw（0%），max_deg ↔ max_raw（+100%）
    #           （单向时 min_deg 字段不参与计算，可省略）
    # home_deg 默认 0°，因此 "home 是 0°" 是天然约定；非对称范围
    # （例如 min_deg=+34.9, max_deg=-159.8）也能正确映射 home→0°。
    min_deg: Optional[float] = None
    max_deg: Optional[float] = None
    home_deg: float = 0.0

    # 运动参数
    max_speed: int = 1500
    max_acc: int = 50

    def __post_init__(self) -> None:
        if self.home_raw == self.max_raw:
            raise ValueError(f"joint {self.name}: home_raw 和 max_raw 不能相等")
        if self.low_raw is not None:
            if self.low_raw == self.home_raw:
                raise ValueError(f"joint {self.name}: low_raw 和 home_raw 不能相等")
            # low 和 max 应该在 home 的不同侧（否则就不是真双向）
            pos_dir = 1 if self.max_raw > self.home_raw else -1
            neg_dir = 1 if self.low_raw > self.home_raw else -1
            if pos_dir == neg_dir:
                raise ValueError(
                    f"joint {self.name}: low_raw({self.low_raw}) 和 max_raw({self.max_raw}) "
                    f"必须在 home_raw({self.home_raw}) 的不同侧"
                )
        if self.min_deg is not None and self.max_deg is not None:
            if self.min_deg == self.max_deg:
                raise ValueError(f"joint {self.name}: min_deg 和 max_deg 不能相等")

    # ---------- 内部辅助 ----------

    @property
    def is_bidirectional(self) -> bool:
        return self.low_raw is not None

    @property
    def _raw_lo(self) -> int:
        candidates = [self.home_raw, self.max_raw]
        if self.low_raw is not None:
            candidates.append(self.low_raw)
        return min(candidates)

    @property
    def _raw_hi(self) -> int:
        candidates = [self.home_raw, self.max_raw]
        if self.low_raw is not None:
            candidates.append(self.low_raw)
        return max(candidates)

    @property
    def pct_min(self) -> float:
        """该关节允许的最小 pct：双向是 -1，单向是 0。"""
        return -1.0 if self.is_bidirectional else 0.0

    @property
    def pct_max(self) -> float:
        return 1.0

    # ---------- 百分比 ↔ raw（首选 API） ----------

    def pct_to_raw(self, pct: float) -> int:
        """百分比（双向 -1..+1，单向 0..+1）→ 舵机原始位置。超界自动夹紧。

        - pct >= 0：从 home 向 max 插值
        - pct <  0：从 home 向 low 插值（仅双向关节；单向时夹到 0）
        """
        pct = float(pct)
        if pct >= 0:
            pct = min(1.0, pct)
            target = self.max_raw
        else:
            if self.low_raw is None:
                pct = 0.0
                target = self.home_raw
            else:
                pct = max(-1.0, pct)
                target = self.low_raw
        raw = self.home_raw + abs(pct) * (target - self.home_raw)
        return int(round(raw))

    def raw_to_pct(self, raw: int) -> float:
        """舵机原始位置 → 百分比。超界自动夹紧到允许范围。"""
        delta = raw - self.home_raw
        if delta == 0:
            return 0.0
        pos_delta = self.max_raw - self.home_raw
        if pos_delta == 0:
            return 0.0
        # 与 max 同侧 → 正向
        same_side_as_max = (delta > 0) == (pos_delta > 0)
        if same_side_as_max:
            return min(1.0, abs(delta) / abs(pos_delta))
        # 与 max 反侧 → 负向（仅双向关节）
        if self.low_raw is None:
            return 0.0
        neg_delta = self.low_raw - self.home_raw
        if neg_delta == 0:
            return 0.0
        return -min(1.0, abs(delta) / abs(neg_delta))

    @property
    def home_pct(self) -> float:
        """home 永远 = 0%（保留属性以兼容旧调用 / UI 显示）。"""
        return 0.0

    # ---------- 物理角度 ↔ raw（可选，三点分段线性） ----------

    def has_deg_mapping(self) -> bool:
        """双向需要 min_deg + max_deg；单向只要 max_deg。home_deg 永远有默认 0°。"""
        if self.is_bidirectional:
            return self.min_deg is not None and self.max_deg is not None
        return self.max_deg is not None

    def deg_to_raw(self, deg: float) -> int:
        """物理角度 → 舵机原始位置（分段线性，home_deg ↔ home_raw 是锚点）。

        约定（home_deg 默认 0°，可在 yaml 重设）：
            双向：deg=min_deg → pct=-1（low_raw）
                  deg=home_deg → pct= 0（home_raw）
                  deg=max_deg → pct=+1（max_raw）
            单向：deg=home_deg → pct=0
                  deg=max_deg → pct=1
                  deg < home_deg 一律夹到 home_raw
        """
        if not self.has_deg_mapping():
            raise ValueError(
                f"joint {self.name}: min_deg/max_deg 未标定，无法使用度数 API"
            )
        home = float(self.home_deg)
        delta = float(deg) - home
        if delta == 0:
            return self.home_raw

        span_max = self.max_deg - home
        # deg 与 max_deg 在 home_deg 同一侧 → pct ∈ [0, 1]
        if span_max != 0 and delta * span_max > 0:
            pct = min(1.0, delta / span_max)
            return self.pct_to_raw(pct)

        # 反方向：仅在双向关节有意义
        if self.is_bidirectional and self.min_deg is not None:
            span_min = self.min_deg - home
            if span_min != 0 and delta * span_min > 0:
                pct = max(-1.0, -delta / span_min)
                return self.pct_to_raw(pct)

        # 单向越界或 deg 落在不可达侧 → 夹到 home
        return self.home_raw

    def raw_to_deg(self, raw: int) -> Optional[float]:
        """舵机原始位置 → 物理角度（分段线性，未标定则返回 None）。"""
        if not self.has_deg_mapping():
            return None
        pct = self.raw_to_pct(raw)
        home = float(self.home_deg)
        if pct >= 0:
            # home → max 段
            return home + pct * (self.max_deg - home)
        # home → low 段（仅双向，pct<0）
        if self.min_deg is None:
            return home
        return home + (-pct) * (self.min_deg - home)


# ---------------------------------------------------------------------------
# 关节联锁约束
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class JointConstraint:
    """关节联锁约束：当 ``gate`` 关节物理角度不满足条件时，``target`` 关节被冻结。

    例::

        JointConstraint(target="wrist_pitch", gate="elbow", gate_min_deg=0, op=">")
        # 当 elbow > 0° 时，wrist_pitch 才能转动。

        JointConstraint(target="end_roll", gate="shoulder", gate_min_deg=90, op="<")
        # 当 shoulder < 90° 时，end_roll 才能转动。

    评估时机：每次发 :meth:`Arm.move_joints_raw` 前读 ``gate`` 关节当前位置；
    若不满足，``target`` 关节的目标会被覆盖为它当前的 raw 值（其他关节
    正常移动）。这是一次性"瞬时检查"，不做连续监控。

    若 ``gate`` 关节没有度数映射（``min_deg``/``max_deg`` 未配置），
    约束自动跳过并发出一次 warning。
    """

    target: str               # 受约束的关节名（满足条件前不能动）
    gate: str                 # 决定是否解锁 target 的关节名
    gate_min_deg: float       # 阈值（单位 °），具体含义看 ``op``
    op: str = ">"             # 比较运算：">" 表示 gate>阈值才解锁；"<" 表示 gate<阈值才解锁
    enabled: bool = True      # 是否启用（关闭后规则跳过）
    note: str = ""            # 可选备注（仅 UI 显示，不影响逻辑）


_CONSTRAINT_FIELDS = {
    "target", "gate", "gate_joint", "gate_min_deg", "op",
    "enabled", "note",
}
_CONSTRAINT_OPS = (">", "<")


@dataclass(frozen=True)
class JointPrerequisite:
    """前置联动：当 ``target`` 关节本次想去的角度满足
    ``target_op target_deg`` 时，``gate`` 关节必须先位于
    ``gate_op gate_deg`` 区间。若不满足，会**先**自动把 gate 转到
    刚刚好满足条件的位置（带 ``buffer_deg`` 度余量）并等待到位，
    然后再发送原动作。

    例::

        JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow",        gate_op=">",   gate_deg=50,
        )
        # 当 wrist_roll 想去 > 30° 时，elbow 必须先 > 50°；
        # 若 elbow 当前在 ≤50°，自动先转到 52°（默认 buffer=2°）等到位再动作。

    与 :class:`JointConstraint` 的区别：
        * Constraint 是"被动冻结"：违反时只把 target 锁住不动
        * Prerequisite 是"主动联动"：触发时会**实际驱动 gate** 到安全位置
    """

    target: str
    target_op: str            # ">" / "<"
    target_deg: float
    gate: str
    gate_op: str
    gate_deg: float
    buffer_deg: float = 2.0   # 自动转到的余量（保证严格 > 或 <）
    enabled: bool = True
    note: str = ""


_PREREQUISITE_FIELDS = {
    "target", "target_op", "target_deg",
    "gate", "gate_op", "gate_deg",
    "buffer_deg",
    "enabled", "note",
}


@dataclass(frozen=True)
class JointZoneLimit:
    """条件限位（"软限位"的一个条件版本）：
    当 ``gate`` 关节角度满足 ``gate_op gate_deg`` 时，``target`` 关节角度被
    禁止 ``target_op target_deg``（即被夹紧到合规边界）。

    例::

        JointZoneLimit(
            gate="elbow",        gate_op="<",   gate_deg=0,
            target="wrist_roll", target_op=">", target_deg=90,
        )
        # 当 elbow < 0° 时，wrist_roll 禁止 > 90°（目标 deg 会被夹到 90°）。

    与 :class:`JointConstraint` 的区别：
        * Constraint 在条件违反时把 target 完全**冻结**到当前 raw（不能动）
        * ZoneLimit 只在 target 目标越过禁区边界时**夹紧**到边界（仍可在合规区自由动）

    评估时机：每次 :meth:`Arm.move_joints_raw` 前读 ``gate`` 当前角度。
    若 gate 或 target 没有度数映射，本条规则跳过并 warning。
    """

    target: str
    target_op: str            # ">" 表示禁止 > target_deg；"<" 表示禁止 < target_deg
    target_deg: float
    gate: str
    gate_op: str
    gate_deg: float
    enabled: bool = True
    note: str = ""


_ZONE_LIMIT_FIELDS = {
    "target", "target_op", "target_deg",
    "gate", "gate_op", "gate_deg",
    "enabled", "note",
}


@dataclass
class ArmConfig:
    """整机配置。"""

    port: str = "auto"
    baudrate: int = 1_000_000
    timeout: float = 0.05
    joints: List[JointConfig] = field(default_factory=list)
    enable_torque_on_init: bool = False
    speed_limit: int = 1800
    poses: Dict[str, List[int]] = field(default_factory=dict)
    """整机命名姿态：每个值是按关节顺序（J1..Jn）的 raw 列表。

    约定的姿态名：
        ``safe``   停止工作的保持姿态（扭矩 OFF 仍能稳定停在那里）
        ``ready``  准备工作姿态（工作起点）

    注：单关节级别的"0 位"参考是 :class:`JointConfig` 的 ``home_raw``，
    走那里用 :meth:`Arm.move_home`，不放在这里以免与整机姿态语义混淆。
    """
    constraints: List[JointConstraint] = field(default_factory=list)
    """关节联锁约束列表（默认为空 = 关闭联锁）。"""
    prerequisites: List[JointPrerequisite] = field(default_factory=list)
    """前置联动列表（默认为空 = 关闭前置联动）。"""
    zone_limits: List[JointZoneLimit] = field(default_factory=list)
    """条件限位列表（默认为空 = 关闭条件限位）。"""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ArmConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        bus = data.get("bus", {})
        safety = data.get("safety", {})

        joints: List[JointConfig] = []
        for raw in data.get("joints", []):
            kept = {k: v for k, v in raw.items() if k in _JOINT_FIELDS}
            unknown = set(raw) - _JOINT_FIELDS
            if unknown:
                warnings.warn(
                    f"忽略关节 {raw.get('name', '?')} 中未知字段：{sorted(unknown)}",
                    stacklevel=2,
                )
            joints.append(JointConfig(**kept))

        poses: Dict[str, List[int]] = {}
        for name, vals in (data.get("poses") or {}).items():
            if not isinstance(vals, (list, tuple)):
                warnings.warn(f"pose '{name}' 不是列表，忽略", stacklevel=2)
                continue
            poses[str(name)] = [int(v) for v in vals]

        joint_names = {j.name for j in joints}
        constraints: List[JointConstraint] = []
        for raw_c in (data.get("constraints") or []):
            if not isinstance(raw_c, dict):
                warnings.warn(f"constraint {raw_c!r} 不是 dict，忽略", stacklevel=2)
                continue
            unknown = set(raw_c) - _CONSTRAINT_FIELDS
            if unknown:
                warnings.warn(
                    f"忽略 constraint 中未知字段：{sorted(unknown)}", stacklevel=2,
                )
            target = raw_c.get("target")
            gate = raw_c.get("gate") or raw_c.get("gate_joint")
            gate_min_deg = raw_c.get("gate_min_deg")
            op = raw_c.get("op", ">")
            if not (target and gate and gate_min_deg is not None):
                warnings.warn(
                    f"constraint 缺少 target/gate/gate_min_deg，跳过：{raw_c!r}",
                    stacklevel=2,
                )
                continue
            if target not in joint_names:
                warnings.warn(
                    f"constraint target '{target}' 不在 joints 列表里，跳过",
                    stacklevel=2,
                )
                continue
            if gate not in joint_names:
                warnings.warn(
                    f"constraint gate '{gate}' 不在 joints 列表里，跳过",
                    stacklevel=2,
                )
                continue
            if op not in _CONSTRAINT_OPS:
                warnings.warn(
                    f"constraint op '{op}' 非法（只支持 '>' / '<'），按 '>' 处理",
                    stacklevel=2,
                )
                op = ">"
            constraints.append(
                JointConstraint(
                    target=str(target), gate=str(gate),
                    gate_min_deg=float(gate_min_deg),
                    op=str(op),
                    enabled=bool(raw_c.get("enabled", True)),
                    note=str(raw_c.get("note", "")),
                )
            )

        prerequisites: List[JointPrerequisite] = []
        for raw_p in (data.get("prerequisites") or []):
            if not isinstance(raw_p, dict):
                warnings.warn(f"prerequisite {raw_p!r} 不是 dict，忽略", stacklevel=2)
                continue
            unknown = set(raw_p) - _PREREQUISITE_FIELDS
            if unknown:
                warnings.warn(
                    f"忽略 prerequisite 中未知字段：{sorted(unknown)}", stacklevel=2,
                )
            target = raw_p.get("target")
            gate = raw_p.get("gate")
            target_op = raw_p.get("target_op", ">")
            gate_op = raw_p.get("gate_op", ">")
            target_deg = raw_p.get("target_deg")
            gate_deg = raw_p.get("gate_deg")
            buffer_deg = raw_p.get("buffer_deg", 2.0)
            if not (target and gate and target_deg is not None and gate_deg is not None):
                warnings.warn(
                    f"prerequisite 缺少 target/gate/target_deg/gate_deg，跳过：{raw_p!r}",
                    stacklevel=2,
                )
                continue
            if target not in joint_names:
                warnings.warn(
                    f"prerequisite target '{target}' 不在 joints 列表里，跳过",
                    stacklevel=2,
                )
                continue
            if gate not in joint_names:
                warnings.warn(
                    f"prerequisite gate '{gate}' 不在 joints 列表里，跳过",
                    stacklevel=2,
                )
                continue
            if target == gate:
                warnings.warn(
                    f"prerequisite target 与 gate 不能相同：{target}", stacklevel=2,
                )
                continue
            if target_op not in _CONSTRAINT_OPS:
                warnings.warn(
                    f"prerequisite target_op '{target_op}' 非法，按 '>' 处理",
                    stacklevel=2,
                )
                target_op = ">"
            if gate_op not in _CONSTRAINT_OPS:
                warnings.warn(
                    f"prerequisite gate_op '{gate_op}' 非法，按 '>' 处理",
                    stacklevel=2,
                )
                gate_op = ">"
            prerequisites.append(
                JointPrerequisite(
                    target=str(target),
                    target_op=str(target_op),
                    target_deg=float(target_deg),
                    gate=str(gate),
                    gate_op=str(gate_op),
                    gate_deg=float(gate_deg),
                    buffer_deg=float(buffer_deg),
                    enabled=bool(raw_p.get("enabled", True)),
                    note=str(raw_p.get("note", "")),
                )
            )

        zone_limits: List[JointZoneLimit] = []
        for raw_z in (data.get("zone_limits") or []):
            if not isinstance(raw_z, dict):
                warnings.warn(f"zone_limit {raw_z!r} 不是 dict，忽略", stacklevel=2)
                continue
            unknown = set(raw_z) - _ZONE_LIMIT_FIELDS
            if unknown:
                warnings.warn(
                    f"忽略 zone_limit 中未知字段：{sorted(unknown)}", stacklevel=2,
                )
            target = raw_z.get("target")
            gate = raw_z.get("gate")
            target_op = raw_z.get("target_op", ">")
            gate_op = raw_z.get("gate_op", ">")
            target_deg = raw_z.get("target_deg")
            gate_deg = raw_z.get("gate_deg")
            if not (target and gate and target_deg is not None and gate_deg is not None):
                warnings.warn(
                    f"zone_limit 缺少 target/gate/target_deg/gate_deg，跳过：{raw_z!r}",
                    stacklevel=2,
                )
                continue
            if target not in joint_names:
                warnings.warn(
                    f"zone_limit target '{target}' 不在 joints 列表里，跳过",
                    stacklevel=2,
                )
                continue
            if gate not in joint_names:
                warnings.warn(
                    f"zone_limit gate '{gate}' 不在 joints 列表里，跳过",
                    stacklevel=2,
                )
                continue
            if target == gate:
                warnings.warn(
                    f"zone_limit target 与 gate 不能相同：{target}", stacklevel=2,
                )
                continue
            if target_op not in _CONSTRAINT_OPS:
                warnings.warn(
                    f"zone_limit target_op '{target_op}' 非法，按 '>' 处理",
                    stacklevel=2,
                )
                target_op = ">"
            if gate_op not in _CONSTRAINT_OPS:
                warnings.warn(
                    f"zone_limit gate_op '{gate_op}' 非法，按 '>' 处理",
                    stacklevel=2,
                )
                gate_op = ">"
            zone_limits.append(
                JointZoneLimit(
                    target=str(target),
                    target_op=str(target_op),
                    target_deg=float(target_deg),
                    gate=str(gate),
                    gate_op=str(gate_op),
                    gate_deg=float(gate_deg),
                    enabled=bool(raw_z.get("enabled", True)),
                    note=str(raw_z.get("note", "")),
                )
            )

        cls._validate_rule_consistency(constraints, prerequisites, zone_limits)

        return cls(
            port=bus.get("port", "auto"),
            baudrate=bus.get("baudrate", 1_000_000),
            timeout=bus.get("timeout", 0.05),
            joints=joints,
            enable_torque_on_init=safety.get("enable_torque_on_init", False),
            speed_limit=safety.get("speed_limit", 1800),
            poses=poses,
            constraints=constraints,
            prerequisites=prerequisites,
            zone_limits=zone_limits,
        )

    @staticmethod
    def _validate_rule_consistency(
        constraints: List["JointConstraint"],
        prerequisites: List["JointPrerequisite"],
        zone_limits: List["JointZoneLimit"],
    ) -> None:
        """跨规则一致性体检：方向冲突 / 依赖环 / 同 target 多 zone_limit 空集。

        所有问题都以 :class:`UserWarning` 形式抛出，不阻塞加载。
        """
        # 1) constraint vs prerequisite 在同一对 (target, gate) 上方向冲突
        c_by_pair = {(c.target, c.gate): c for c in constraints if c.enabled}
        for p in prerequisites:
            if not p.enabled:
                continue
            c = c_by_pair.get((p.target, p.gate))
            if c is None:
                continue
            if c.op != p.gate_op or abs(c.gate_min_deg - p.gate_deg) > 1e-6:
                warnings.warn(
                    f"规则冲突：constraint [{c.target}←{c.gate} 需 {c.op}{c.gate_min_deg}°] "
                    f"与 prerequisite [{p.target}←{p.gate} 需 {p.gate_op}{p.gate_deg}°] "
                    f"作用于同一对关节但条件不一致，可能在某些状态下死锁",
                    stacklevel=3,
                )

        # 2) prerequisite 形成依赖环（target → gate → ...）
        adj: Dict[str, set] = {}
        for p in prerequisites:
            if not p.enabled:
                continue
            adj.setdefault(p.target, set()).add(p.gate)
        # DFS 检环
        WHITE, GRAY, BLACK = 0, 1, 2
        color: Dict[str, int] = {n: WHITE for n in adj}

        def dfs(u: str, path: List[str]) -> None:
            color[u] = GRAY
            for v in adj.get(u, ()):
                if color.get(v, WHITE) == GRAY:
                    cycle = " → ".join(path + [u, v])
                    warnings.warn(
                        f"prerequisite 依赖环：{cycle}（可能死锁）",
                        stacklevel=4,
                    )
                elif color.get(v, WHITE) == WHITE:
                    color[v] = WHITE
                    color.setdefault(v, WHITE)
                    dfs(v, path + [u])
            color[u] = BLACK

        for n in list(adj.keys()):
            if color.get(n, WHITE) == WHITE:
                dfs(n, [])

        # 3) 同 target 上的多条 zone_limit 是否方向矛盾导致空集
        z_by_target: Dict[str, list] = {}
        for z in zone_limits:
            if not z.enabled:
                continue
            z_by_target.setdefault(z.target, []).append(z)
        for tgt, items in z_by_target.items():
            ups = [z.target_deg for z in items if z.target_op == ">"]
            dns = [z.target_deg for z in items if z.target_op == "<"]
            if ups and dns:
                lo = max(dns)   # 禁 < lo → 下界
                hi = min(ups)   # 禁 > hi → 上界
                if lo > hi:
                    warnings.warn(
                        f"zone_limit 矛盾：在 '{tgt}' 上同时有 "
                        f"禁 < {lo}° 与 禁 > {hi}°，"
                        f"两条规则同时激活时合规区间 [{lo}, {hi}] 为空",
                        stacklevel=3,
                    )


# ---------------------------------------------------------------------------
# 工作状态
# ---------------------------------------------------------------------------

class ArmState(Enum):
    """整机工作状态。

    DISABLED  扭矩 OFF / 停止工作；机械臂应处于 ``safe_pose`` 这种
              扭矩 OFF 也能保持的稳定姿态。运动指令在此状态下被拒绝。
    ENABLED   扭矩 ON / 准备工作；进入此状态时机械臂会被带到 ``ready_pose``
              工作起点，之后接受 ``move_pct`` / ``move_deg`` 等运动指令。
    """

    DISABLED = "disabled"
    ENABLED = "enabled"


# ---------------------------------------------------------------------------
# Arm
# ---------------------------------------------------------------------------

class Arm:
    """6 DOF 机械臂高层控制器。

    用法：
        cfg = ArmConfig.from_yaml("config/arm_config.yaml")
        with Arm(cfg) as arm:
            arm.enable()                 # safe_pose → ready_pose（开扭矩）
            arm.move_pct([0.5] * 6)
            arm.disable()                # → safe_pose（关扭矩）
    """

    def __init__(self, config: ArmConfig, bus: Optional[Bus] = None):
        self.config = config
        self.bus = bus or Bus(
            port=config.port,
            baudrate=config.baudrate,
            timeout=config.timeout,
        )
        self.servos: List[Servo] = [Servo(self.bus, j.id) for j in config.joints]
        self._state: ArmState = ArmState.DISABLED
        # 干预事件 ring buffer（最近 N 条），WebSocket 状态会带出去
        from collections import deque
        self.recent_interventions: deque = deque(maxlen=20)

    def _record_intervention(self, kind: str, msg: str) -> None:
        """记录一条规则干预事件（log + ring buffer）。

        kind: "constraint" / "prerequisite" / "zone_limit"
        """
        self.recent_interventions.append({
            "ts": time.time(),
            "kind": kind,
            "msg": msg,
        })
        log.info("[%s] %s", kind, msg)

    # ---- 生命周期 ----

    def open(self) -> "Arm":
        self.bus.open()
        if self.config.enable_torque_on_init:
            self.set_torque(True)
        return self

    def close(self) -> None:
        self.bus.close()

    def __enter__(self) -> "Arm":
        return self.open()

    def __exit__(self, *args) -> None:
        self.close()

    # ---- 探活 ----

    def ping_all(self) -> dict[int, bool]:
        return {s.sid: s.ping() for s in self.servos}

    # ---- 扭矩 ----

    def set_torque(self, on: bool) -> None:
        """同步开关所有关节扭矩（一次 sync_write）。"""
        ids = [j.id for j in self.config.joints]
        items = [(sid, bytes([1 if on else 0])) for sid in ids]
        self.bus.transact(
            Packet.sync_write(Reg.TORQUE_ENABLE, 1, items),
            expect_response=False,
        )

    # ---- 关节读 ----

    def read_joints_raw(self) -> List[int]:
        """逐关节读当前 raw 位置（0..4095）。"""
        return [s.read_position() for s in self.servos]

    def read_pct(self) -> List[float]:
        """读全部关节当前百分比（0.0..1.0）。"""
        raws = self.read_joints_raw()
        return [j.raw_to_pct(r) for j, r in zip(self.config.joints, raws)]

    def read_deg(self) -> List[Optional[float]]:
        """读全部关节当前物理角度（未标定的关节返回 None）。"""
        raws = self.read_joints_raw()
        return [j.raw_to_deg(r) for j, r in zip(self.config.joints, raws)]

    def read_feedback(self) -> List[Feedback]:
        """读全部关节完整反馈（每颗单独通信，较慢）。"""
        return [s.feedback() for s in self.servos]

    # ---- 关节写（同步） ----

    def move_joints_raw(
        self,
        positions: Sequence[int],
        speeds: Optional[Sequence[int]] = None,
        accs: Optional[Sequence[int]] = None,
        bypass_constraints: bool = False,
        bypass_prerequisites: bool = False,
    ) -> None:
        """同步把所有关节移到指定 raw 位置（一帧 sync_write）。

        若 ``config.constraints`` / ``config.prerequisites`` 不为空且
        ``bypass_constraints=False``：

        * **prerequisites**：在主 sync_write 之前先评估，必要时**先单独
          移动 gate** 到刚好满足前置条件的位置并 wait_until_idle，再继续
          （会改写 gate 关节在本次 positions 中的目标，避免随后又被拉回违反位置）。
        * **constraints**：在 prerequisites 之后评估，把违反约束的 target
          关节冻结到当前 raw（"被动锁定"）。
        """
        n = len(self.config.joints)
        if len(positions) != n:
            raise ValueError(f"expected {n} positions, got {len(positions)}")

        positions = list(positions)
        if (self.config.prerequisites
                and not bypass_constraints and not bypass_prerequisites):
            positions = self._apply_prerequisites(positions, speeds, accs)
        if self.config.zone_limits and not bypass_constraints:
            positions = self._apply_zone_limits(positions)
        if self.config.constraints and not bypass_constraints:
            positions = self._apply_constraints(positions)

        speeds = list(speeds) if speeds is not None else [j.max_speed for j in self.config.joints]
        accs = list(accs) if accs is not None else [j.max_acc for j in self.config.joints]

        items: List[tuple[int, bytes]] = []
        for j, pos, spd, acc in zip(self.config.joints, positions, speeds, accs):
            spd_clamped = min(int(spd), self.config.speed_limit)
            # 软限位夹紧：raw 必须落在 [home_raw, max_raw] 之间（不论方向）
            pos_clamped = max(j._raw_lo, min(j._raw_hi, int(pos)))
            payload = (
                bytes([acc & 0xFF])
                + p.to_le16(pos_clamped)
                + p.to_le16(0)              # GOAL_TIME
                + p.to_le16(spd_clamped)
            )
            items.append((j.id, payload))

        self.bus.transact(
            Packet.sync_write(Reg.ACC, 7, items),
            expect_response=False,
        )

    # ---- 条件限位 ----

    def _apply_zone_limits(self, positions: List[int]) -> List[int]:
        """对每条 zone_limit：

        1. 读 gate 当前 deg，判断是否处于激活区间（``gate_op gate_deg``）
        2. 激活时把 ``positions[target_idx]`` 的 deg 夹紧到禁区边界
           （``target_deg``），随后转回 raw 写入 positions

        若 gate / target 没有度数映射，本条规则跳过并 warning。
        """
        if not self.config.zone_limits:
            return positions

        name_to_idx = {j.name: i for i, j in enumerate(self.config.joints)}

        try:
            current = self.read_joints_raw()
        except Exception as e:
            log.warning("条件限位读关节位置失败：%s（限位本次跳过）", e)
            return positions

        for z in self.config.zone_limits:
            if not z.enabled:
                continue
            ti = name_to_idx.get(z.target)
            gi = name_to_idx.get(z.gate)
            if ti is None or gi is None:
                continue

            gate_j = self.config.joints[gi]
            target_j = self.config.joints[ti]

            gate_deg = gate_j.raw_to_deg(current[gi])
            if gate_deg is None:
                warnings.warn(
                    f"关节 '{z.gate}' 没有度数映射，"
                    f"限位 [{z.target} 禁 {z.target_op} {z.target_deg}° "
                    f"if {z.gate} {z.gate_op} {z.gate_deg}°] 跳过",
                    stacklevel=2,
                )
                continue

            if z.gate_op == ">":
                active = gate_deg > z.gate_deg
            else:
                active = gate_deg < z.gate_deg
            if not active:
                continue

            tgt_deg = target_j.raw_to_deg(positions[ti])
            if tgt_deg is None:
                warnings.warn(
                    f"关节 '{z.target}' 没有度数映射，"
                    f"限位 [{z.target} 禁 {z.target_op} {z.target_deg}°] 跳过",
                    stacklevel=2,
                )
                continue

            if z.target_op == ">":
                violated = tgt_deg > z.target_deg
                clamped_deg = z.target_deg if violated else tgt_deg
            else:
                violated = tgt_deg < z.target_deg
                clamped_deg = z.target_deg if violated else tgt_deg

            if violated:
                clamped_raw = target_j.deg_to_raw(clamped_deg)
                self._record_intervention(
                    "zone_limit",
                    f"{z.gate} 当前 {gate_deg:.1f}°（{z.gate_op}{z.gate_deg}° 激活），"
                    f"{z.target} 目标 {tgt_deg:.1f}° 越过禁区边界 → 夹紧到 {clamped_deg:.1f}°",
                )
                positions[ti] = clamped_raw

        return positions

    # ---- 关节联锁约束 ----

    def _apply_constraints(self, positions: List[int]) -> List[int]:
        """读 gate 关节当前位置，把违反约束的 target 关节冻结到当前 raw。

        策略：
            1. 一次性 :meth:`read_joints_raw` 获取所有关节当前 raw（避免散读）
            2. 对每条约束计算 gate 当前角度
            3. 不满足时把 ``positions[target_idx]`` 改写为 ``current[target_idx]``

        若 gate 关节没有度数映射会发出 warning 并跳过该约束。
        """
        if not self.config.constraints:
            return positions

        name_to_idx = {j.name: i for i, j in enumerate(self.config.joints)}

        try:
            current = self.read_joints_raw()
        except Exception as e:
            log.warning("约束检查时读关节位置失败：%s（联锁本次跳过）", e)
            return positions

        for c in self.config.constraints:
            if not c.enabled:
                continue
            target_idx = name_to_idx.get(c.target)
            gate_idx = name_to_idx.get(c.gate)
            if target_idx is None or gate_idx is None:
                continue

            gate_j = self.config.joints[gate_idx]
            gate_deg = gate_j.raw_to_deg(current[gate_idx])
            if gate_deg is None:
                warnings.warn(
                    f"关节 '{c.gate}' 没有度数映射（min_deg/max_deg 未配），"
                    f"约束 [{c.target} 需 {c.gate}{c.op}{c.gate_min_deg}°] 跳过",
                    stacklevel=2,
                )
                continue

            if c.op == "<":
                violated = gate_deg >= c.gate_min_deg
            else:  # ">"
                violated = gate_deg <= c.gate_min_deg

            if violated:
                if positions[target_idx] != current[target_idx]:
                    self._record_intervention(
                        "constraint",
                        f"{c.gate} 当前 {gate_deg:.1f}° 不满足 {c.op}{c.gate_min_deg}°，"
                        f"{c.target} 被冻结到 raw={current[target_idx]}",
                    )
                    positions[target_idx] = current[target_idx]

        return positions

    # ---- 前置联动 ----

    def _apply_prerequisites(
        self,
        positions: List[int],
        speeds: Optional[Sequence[int]],
        accs: Optional[Sequence[int]],
    ) -> List[int]:
        """对每条 prerequisite：

        1. 用 ``positions[target]`` 的目标 deg 判断是否触发；
        2. 若触发，读 gate 当前 raw → deg；
        3. 若 gate 当前不满足前置区间，**先**通过 servo 单独发指令把 gate
           转到 ``gate_deg ± buffer_deg`` 并 :meth:`wait_until_idle`；
        4. 同时把 ``positions[gate_idx]`` 改写为该安全 raw（避免随后
           sync_write 再把 gate 拉回违反位置）。

        若 target / gate 没有度数映射，本条 prerequisite 跳过并 warning。
        """
        if not self.config.prerequisites:
            return positions

        name_to_idx = {j.name: i for i, j in enumerate(self.config.joints)}

        for pr in self.config.prerequisites:
            if not pr.enabled:
                continue
            ti = name_to_idx.get(pr.target)
            gi = name_to_idx.get(pr.gate)
            if ti is None or gi is None:
                continue
            target_j = self.config.joints[ti]
            gate_j = self.config.joints[gi]

            tgt_deg = target_j.raw_to_deg(positions[ti])
            if tgt_deg is None:
                warnings.warn(
                    f"前置 [{pr.target} {pr.target_op} {pr.target_deg}° → "
                    f"{pr.gate} {pr.gate_op} {pr.gate_deg}°]：target 没有度数映射，跳过",
                    stacklevel=2,
                )
                continue

            if pr.target_op == ">":
                triggered = tgt_deg > pr.target_deg
            else:
                triggered = tgt_deg < pr.target_deg
            if not triggered:
                continue

            try:
                cur_gate_raw = self.servos[gi].read_position()
            except Exception as e:
                log.warning("前置检查读 %s 位置失败：%s（本条跳过）", pr.gate, e)
                continue

            cur_gate_deg = gate_j.raw_to_deg(cur_gate_raw)
            if cur_gate_deg is None:
                warnings.warn(
                    f"前置 [...{pr.gate} {pr.gate_op} {pr.gate_deg}°]："
                    f"gate 没有度数映射，跳过",
                    stacklevel=2,
                )
                continue

            if pr.gate_op == ">":
                cur_satisfied = cur_gate_deg > pr.gate_deg
            else:
                cur_satisfied = cur_gate_deg < pr.gate_deg

            usr_gate_deg = gate_j.raw_to_deg(positions[gi])
            if usr_gate_deg is None:
                usr_satisfied = False
            elif pr.gate_op == ">":
                usr_satisfied = usr_gate_deg > pr.gate_deg
            else:
                usr_satisfied = usr_gate_deg < pr.gate_deg

            if cur_satisfied and usr_satisfied:
                continue        # 当前满足，且用户原 gate 目标也满足，啥都不用做

            # 算"刚好满足条件 + buffer"的安全位置
            buf = pr.buffer_deg
            safe_deg = pr.gate_deg + buf if pr.gate_op == ">" else pr.gate_deg - buf
            safe_raw = gate_j.deg_to_raw(safe_deg)

            if not cur_satisfied:
                self._record_intervention(
                    "prerequisite",
                    f"{pr.target} 目标 {tgt_deg:.1f}° 触发前置 → "
                    f"{pr.gate} 当前 {cur_gate_deg:.1f}° 不满足 {pr.gate_op}{pr.gate_deg}°，"
                    f"自动先转到 {safe_deg:.1f}°",
                )
                # 关键修复（P0）：让 gate 的自动驱动也走完整规则链，
                # 避免直接 write_position 绕开 constraint / zone_limit。
                # 用 bypass_prerequisites=True 防止前置规则间无限递归。
                try:
                    cur_all = self.read_joints_raw()
                except Exception as e:
                    raise RuntimeError(
                        f"前置联动读关节位置失败：{e}"
                    ) from e

                stage_pos = list(cur_all)
                stage_pos[gi] = safe_raw
                self.move_joints_raw(stage_pos, bypass_prerequisites=True)

                if not self.wait_until_idle(timeout=10.0):
                    raise RuntimeError(
                        f"前置联动：{pr.gate} 在 10s 内未到位 "
                        f"(目标 {safe_deg:.1f}°)，已中止本次移动"
                    )

            if not usr_satisfied:
                self._record_intervention(
                    "prerequisite",
                    f"用户 {pr.gate} 目标 "
                    f"{(usr_gate_deg if usr_gate_deg is not None else float('nan')):.1f}° "
                    f"违反前置条件，本次将其改写为 {safe_deg:.1f}°",
                )
                positions[gi] = safe_raw

        return positions

    def move_pct(
        self,
        pcts: Sequence[float],
        speed: Optional[int] = None,
        acc: Optional[int] = None,
    ) -> None:
        """**首选 API**：同步把所有关节移到指定百分比（0.0..1.0）。"""
        n = len(self.config.joints)
        if len(pcts) != n:
            raise ValueError(f"expected {n} pct values, got {len(pcts)}")
        positions = [j.pct_to_raw(v) for j, v in zip(self.config.joints, pcts)]
        speeds = [speed] * n if speed is not None else None
        accs = [acc] * n if acc is not None else None
        self.move_joints_raw(positions, speeds, accs)

    def move_deg(
        self,
        angles_deg: Sequence[float],
        speed: Optional[int] = None,
        acc: Optional[int] = None,
    ) -> None:
        """同步把所有关节移到指定物理角度（°），需要每个关节都已标定 min_deg/max_deg。"""
        n = len(self.config.joints)
        if len(angles_deg) != n:
            raise ValueError(f"expected {n} angles, got {len(angles_deg)}")
        for j in self.config.joints:
            if not j.has_deg_mapping():
                raise ValueError(
                    f"关节 {j.name} 未标定 min_deg/max_deg；如需用度数 API，"
                    f"请先在 config 中补充。或改用 move_pct()。"
                )
        positions = [j.deg_to_raw(a) for j, a in zip(self.config.joints, angles_deg)]
        speeds = [speed] * n if speed is not None else None
        accs = [acc] * n if acc is not None else None
        self.move_joints_raw(positions, speeds, accs)

    def move_home(self, speed: int = 600, acc: int = 30) -> None:
        """所有关节回到 home_raw（应用层 0%）。"""
        positions = [j.home_raw for j in self.config.joints]
        n = len(self.config.joints)
        self.move_joints_raw(positions, speeds=[speed] * n, accs=[acc] * n)

    # ---- 命名姿态 / 工作状态 ----

    @property
    def state(self) -> ArmState:
        """当前工作状态（DISABLED / ENABLED）。"""
        return self._state

    def goto_pose(
        self,
        name: str,
        speed: int = 600,
        acc: int = 30,
        wait: bool = True,
        timeout: float = 10.0,
    ) -> None:
        """走到 ``config.poses`` 中定义的命名姿态（如 ``safe`` / ``ready``）。

        发出运动指令前不会主动开扭矩——若舵机扭矩为 OFF，写 GOAL_POSITION
        在底层是合法操作但舵机不会动；调用方需自己保证扭矩状态。
        """
        positions = self.config.poses.get(name)
        if positions is None:
            raise ValueError(
                f"unknown pose '{name}', available: {sorted(self.config.poses)}"
            )
        n = len(self.config.joints)
        if len(positions) != n:
            raise ValueError(
                f"pose '{name}' 有 {len(positions)} 个位置，期望 {n}"
            )
        self.move_joints_raw(positions, speeds=[speed] * n, accs=[acc] * n)
        if wait:
            self.wait_until_idle(timeout=timeout)

    def enable(self, speed: int = 600, acc: int = 30, timeout: float = 10.0) -> None:
        """进入 ENABLED（准备工作）状态：开扭矩 → 走到 ``ready_pose``。

        过程：
            1. 把每个关节的 GOAL 设为当前 raw（防止开扭矩瞬间跳到上次 GOAL）
            2. 同步开启全部扭矩
            3. 缓慢从当前位置走到 ``ready_pose``

        配置中必须存在名为 ``ready`` 的 pose。
        """
        if "ready" not in self.config.poses:
            raise RuntimeError("config 中未定义 'ready' 姿态，无法 enable()")

        if self._state != ArmState.ENABLED:
            current = self.read_joints_raw()
            n = len(self.servos)
            self.move_joints_raw(current, speeds=[speed] * n, accs=[acc] * n)
            self.set_torque(True)

        self.goto_pose("ready", speed=speed, acc=acc, wait=True, timeout=timeout)
        self._state = ArmState.ENABLED

    def disable(
        self,
        to: Optional[str] = "safe",
        speed: int = 400,
        acc: int = 30,
        timeout: float = 10.0,
    ) -> None:
        """进入 DISABLED（停止工作）状态：走到目标姿态 → 关扭矩。

        参数：
            to       目标姿态名（``config.poses`` 中的 key）。
                     - ``"safe"`` 默认，最安全的折叠/收起停止位
                     - 其他已命名姿态（如 ``"ready"`` 或自定义）
                     - ``None``：不移动，直接关扭矩（机械臂会因重力自然下沉）
                     若想回单关节级 0 位 ``home_raw``，请用 :meth:`move_home`。
            speed    关节移动速度（步/秒），停止动作建议慢一点
            acc      关节加速度
            timeout  等待运动到位的最大秒数

        当指定 ``to`` 但 config 中没有该姿态时抛 ``ValueError``。
        """
        if to is not None:
            if to not in self.config.poses:
                raise ValueError(
                    f"unknown pose '{to}', available: {sorted(self.config.poses)}"
                )
            self.goto_pose(to, speed=speed, acc=acc, wait=True, timeout=timeout)

        self.set_torque(False)
        self._state = ArmState.DISABLED

    # ---- 等待运动完成 ----

    def wait_until_idle(self, timeout: float = 10.0, poll: float = 0.05) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            moving = False
            for s in self.servos:
                try:
                    if self.bus.read_u8(s.sid, Reg.MOVING):
                        moving = True
                        break
                except Exception:
                    moving = True
                    break
            if not moving:
                return True
            time.sleep(poll)
        return False

    # ---- 紧急停止 ----

    def emergency_stop(self) -> None:
        """立刻把目标位置设为当前位置，并放扭矩（柔性停止）。

        紧急停止必须无条件生效，所有联动规则一律绕过。
        """
        try:
            current = self.read_joints_raw()
            self.move_joints_raw(
                current,
                speeds=[0] * len(current),
                accs=[0] * len(current),
                bypass_constraints=True,
            )
        finally:
            self.set_torque(False)


__all__ = [
    "Arm", "ArmConfig", "JointConfig",
    "JointConstraint", "JointPrerequisite", "JointZoneLimit",
]
