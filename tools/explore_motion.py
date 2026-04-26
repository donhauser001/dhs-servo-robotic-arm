"""自动探索运动边界（堵转检测 + 慢速渐进）。

运行前请先停止 webapp 服务，避免与本脚本争抢串口：

    sudo systemctl stop arm-web
    python -m tools.explore_motion --phase 0
    # 看完报告再决定下一步
    python -m tools.explore_motion --phase 1 --confirm
    python -m tools.explore_motion --phase 2 --target ready --confirm
    sudo systemctl start arm-web

三个阶段，由轻到重：

    phase 0  只读勘察。Ping、读位置/温度/负载，给出健康报告
              和到 home/ready/safe 的距离。**不动臂、不开扭矩**。
    phase 1  末端单轴小幅试探（J6/J5/J4 各 ±10°）。慢速 + 堵转
              监测，得到每颗舵机在自由空间的负载基线。
    phase 2  按 J6→J1 顺序逐关节移到目标姿态。慢速 + 堵转监测，
              失败的关节会被跳过（不影响后续关节），最后再尝试一次。

任意时刻按 Ctrl+C 立刻紧急停止（卸扭矩 + 冻结目标）。
脚本退出时（无论正常 / 异常）都会调一次 ``arm.emergency_stop()``。
"""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from arm.arm import Arm, ArmConfig
from arm.servo import Feedback


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "arm_config.yaml"

# ---- 探索的安全阈值（保守版：宁可误停，绝不撞坏）-----------------------

SAFE_MAX_SPEED = 300        # 步/秒；约 1/5 全速。探索时强制不超过此值
SAFE_MAX_ACC = 12           # 启停加速度；越小越柔
DEFAULT_SPEED = 200
DEFAULT_ACC = 8

LOAD_WARN = 400             # |load| > 400 → 打印警告（继续走）
LOAD_STOP = 600             # |load| > 600 持续 STALL_HOLD_MS → 立即停
STALL_HOLD_MS = 150         # 高负载持续多久判定堵转

POS_TOLERANCE_RAW = 8       # 到位判定容差（约 0.7°）
NO_MOVE_RAW = 3             # 位置变化 < 3 raw 视为"没动"
NO_MOVE_HOLD_MS = 400       # 没动持续 N ms 但目标差 > POS_TOLERANCE → 堵转

WRONG_DIR_RAW = 8           # 朝错误方向走 > 8 raw → 异常停

TEMP_PAUSE_C = 60           # 温度 > 60℃ → 暂停

POLL_HZ = 8                 # 监测轮询频率；feedback 6 颗 ~50ms，留余量
POLL_PERIOD_S = 1.0 / POLL_HZ

PHASE1_DELTA_DEG = 10       # phase 1 末端摇摆幅度
PHASE1_TIMEOUT_S = 4.0      # 单段移动最长等待
PHASE2_TIMEOUT_S = 6.0


log = logging.getLogger("explore")


# ---------------------------------------------------------------------------
# 全局状态：用于 SIGINT handler 紧急停止
# ---------------------------------------------------------------------------

_active_arm: Optional[Arm] = None


def _emergency_handler(signum, frame):  # noqa: ANN001
    print(f"\n!!! 收到信号 {signum}，立刻紧急停止 !!!", flush=True)
    if _active_arm is not None:
        try:
            _active_arm.emergency_stop()
        except Exception as e:
            print(f"emergency_stop 失败：{e}", flush=True)
    sys.exit(130)


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class JointSnapshot:
    name: str
    sid: int
    raw: int
    deg: Optional[float]
    load: int
    temp_c: int
    voltage: float
    moving: bool
    current_ma: Optional[float]


@dataclass
class MoveResult:
    """一次单段移动的结果。"""
    label: str
    target_raw: List[int]
    final_raw: List[int]
    elapsed_s: float
    status: str               # "ok" | "stall" | "timeout" | "wrong_dir" | "overload" | "skipped"
    detail: str = ""
    peak_load: Dict[str, int] = field(default_factory=dict)


@dataclass
class PhaseReport:
    phase: str
    started_at: float
    finished_at: float = 0.0
    snapshots: List[JointSnapshot] = field(default_factory=list)
    moves: List[MoveResult] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# 辅助打印
# ---------------------------------------------------------------------------

def _fmt_load(load: int) -> str:
    """load 是 -1000..+1000，绝对值越大越吃力。"""
    a = abs(load)
    if a >= LOAD_STOP:
        tag = "STOP"
    elif a >= LOAD_WARN:
        tag = "WARN"
    else:
        tag = "ok  "
    return f"{load:+5d} [{tag}]"


def _fmt_deg(deg: Optional[float]) -> str:
    if deg is None:
        return "  --   "
    return f"{deg:+7.2f}"


def _read_snapshot(arm: Arm) -> List[JointSnapshot]:
    fbs = arm.read_feedback()
    out: List[JointSnapshot] = []
    for j, s, fb in zip(arm.config.joints, arm.servos, fbs):
        out.append(JointSnapshot(
            name=j.name,
            sid=s.sid,
            raw=fb.position,
            deg=j.raw_to_deg(fb.position),
            load=fb.load,
            temp_c=fb.temperature,
            voltage=fb.voltage,
            moving=fb.moving,
            current_ma=fb.current_ma,
        ))
    return out


def _print_snapshot(snaps: List[JointSnapshot]) -> None:
    print(f"  {'idx':<3}{'name':<13}{'sid':>3}  {'raw':>4}  {'deg':>7}  "
          f"{'load':>13}  {'temp':>4}  {'V':>5}")
    for i, s in enumerate(snaps, 1):
        warn = " HOT!" if s.temp_c > TEMP_PAUSE_C else ""
        print(f"  J{i:<2}{s.name:<13}{s.sid:>3}  {s.raw:>4}  "
              f"{_fmt_deg(s.deg)}  {_fmt_load(s.load)}  "
              f"{s.temp_c:>3}℃  {s.voltage:>4.1f}V{warn}")


# ---------------------------------------------------------------------------
# 安全移动 + 监测
# ---------------------------------------------------------------------------

def safe_move(
    arm: Arm,
    target: List[int],
    *,
    speed: int,
    acc: int,
    timeout_s: float,
    label: str,
    dry_run: bool = False,
) -> MoveResult:
    """阻塞式安全移动：发命令后轮询监测，到位 / 堵转 / 超时三选一返回。

    所有规则一律 bypass，因为探索目的就是测物理边界。
    """
    speed = min(speed, SAFE_MAX_SPEED)
    acc = min(acc, SAFE_MAX_ACC)

    n = len(arm.config.joints)
    cur_before = arm.read_joints_raw()
    initial_diff = [t - c for t, c in zip(target, cur_before)]
    sign = [1 if d > 0 else (-1 if d < 0 else 0) for d in initial_diff]

    if dry_run:
        return MoveResult(
            label=label,
            target_raw=list(target),
            final_raw=list(cur_before),
            elapsed_s=0.0,
            status="skipped",
            detail="dry-run",
        )

    print(f"  → 发送目标 {target}  speed={speed} acc={acc}", flush=True)
    arm.move_joints_raw(
        target, speeds=[speed] * n, accs=[acc] * n,
        bypass_constraints=True,    # 探索时绕开规则
    )

    t0 = time.monotonic()
    deadline = t0 + timeout_s
    last_pos = list(cur_before)
    last_pos_change_t = t0
    overload_since: Dict[int, float] = {}
    peak_load: Dict[str, int] = {j.name: 0 for j in arm.config.joints}

    while True:
        now = time.monotonic()
        if now >= deadline:
            arm.emergency_stop()
            final = arm.read_joints_raw()
            return MoveResult(
                label=label,
                target_raw=list(target),
                final_raw=final,
                elapsed_s=now - t0,
                status="timeout",
                detail=f"超时 {timeout_s:.1f}s 仍未到位，已 emergency_stop",
                peak_load=peak_load,
            )

        try:
            fbs = arm.read_feedback()
        except Exception as e:
            arm.emergency_stop()
            final = arm.read_joints_raw()
            return MoveResult(
                label=label,
                target_raw=list(target),
                final_raw=final,
                elapsed_s=now - t0,
                status="overload",
                detail=f"读反馈失败：{e}（可能丢包）",
                peak_load=peak_load,
            )

        cur_pos = [fb.position for fb in fbs]

        for i, fb in enumerate(fbs):
            jn = arm.config.joints[i].name
            if abs(fb.load) > peak_load[jn]:
                peak_load[jn] = abs(fb.load)

        for i, fb in enumerate(fbs):
            if abs(fb.load) > LOAD_STOP:
                overload_since.setdefault(i, now)
                if (now - overload_since[i]) * 1000 >= STALL_HOLD_MS:
                    arm.emergency_stop()
                    return MoveResult(
                        label=label,
                        target_raw=list(target),
                        final_raw=cur_pos,
                        elapsed_s=now - t0,
                        status="overload",
                        detail=f"J{i+1}({arm.config.joints[i].name}) load={fb.load} "
                               f"持续 ≥{STALL_HOLD_MS}ms，已停",
                        peak_load=peak_load,
                    )
            else:
                overload_since.pop(i, None)

        for i in range(n):
            if sign[i] == 0:
                continue
            actual = cur_pos[i] - cur_before[i]
            if actual * sign[i] < -WRONG_DIR_RAW:
                arm.emergency_stop()
                return MoveResult(
                    label=label,
                    target_raw=list(target),
                    final_raw=cur_pos,
                    elapsed_s=now - t0,
                    status="wrong_dir",
                    detail=f"J{i+1}({arm.config.joints[i].name}) 朝错误方向走了 "
                           f"{actual} raw（目标方向 {sign[i]}），已停",
                    peak_load=peak_load,
                )

        max_pos_change = max(abs(c - p) for c, p in zip(cur_pos, last_pos))
        if max_pos_change >= NO_MOVE_RAW:
            last_pos = list(cur_pos)
            last_pos_change_t = now

        diff_to_target = [abs(t - c) for t, c in zip(target, cur_pos)]
        all_in_pos = all(d <= POS_TOLERANCE_RAW for d in diff_to_target)
        any_moving = any(fb.moving for fb in fbs)

        if all_in_pos and not any_moving:
            return MoveResult(
                label=label,
                target_raw=list(target),
                final_raw=cur_pos,
                elapsed_s=now - t0,
                status="ok",
                detail=f"全部到位（最大偏差 {max(diff_to_target)} raw）",
                peak_load=peak_load,
            )

        if (now - last_pos_change_t) * 1000 >= NO_MOVE_HOLD_MS:
            offenders = [
                (i, diff_to_target[i])
                for i in range(n)
                if diff_to_target[i] > POS_TOLERANCE_RAW
            ]
            if offenders:
                arm.emergency_stop()
                names = ", ".join(
                    f"J{i+1}({arm.config.joints[i].name}) 还差 {d} raw"
                    for i, d in offenders
                )
                return MoveResult(
                    label=label,
                    target_raw=list(target),
                    final_raw=cur_pos,
                    elapsed_s=now - t0,
                    status="stall",
                    detail=f"位置 {NO_MOVE_HOLD_MS}ms 不变但未到位 → 堵转：{names}",
                    peak_load=peak_load,
                )

        for i, fb in enumerate(fbs):
            if fb.temperature > TEMP_PAUSE_C:
                arm.emergency_stop()
                return MoveResult(
                    label=label,
                    target_raw=list(target),
                    final_raw=cur_pos,
                    elapsed_s=now - t0,
                    status="overload",
                    detail=f"J{i+1}({arm.config.joints[i].name}) "
                           f"温度 {fb.temperature}℃ > {TEMP_PAUSE_C}℃，已停",
                    peak_load=peak_load,
                )

        time.sleep(POLL_PERIOD_S)


# ---------------------------------------------------------------------------
# Phase 0：只读勘察
# ---------------------------------------------------------------------------

def phase0_inspect(arm: Arm) -> PhaseReport:
    rep = PhaseReport(phase="phase0_inspect", started_at=time.time())
    print("\n[Phase 0] 只读勘察（不动臂、不开扭矩）")
    print("=" * 70)

    print("\n* Ping 全部舵机...")
    pings = arm.ping_all()
    bad = [sid for sid, ok in pings.items() if not ok]
    if bad:
        msg = f"以下 ID 没响应：{bad}"
        rep.notes.append(msg)
        print(f"  !! {msg}")
    else:
        print(f"  全部 {len(pings)} 颗舵机在线 ✓")

    print("\n* 读完整反馈...")
    snaps = _read_snapshot(arm)
    rep.snapshots = snaps
    _print_snapshot(snaps)

    print("\n* 与命名姿态的距离（按关节顺序，单位 raw）")
    cur = [s.raw for s in snaps]
    poses = arm.config.poses
    if not poses:
        print("  config 里没有 poses")
    for name, pos in poses.items():
        diff = [t - c for t, c in zip(pos, cur)]
        max_abs = max(abs(d) for d in diff)
        print(f"  {name:<8} → Δ {diff}   |max| = {max_abs}")

    print("\n* 健康摘要")
    max_temp = max(s.temp_c for s in snaps)
    max_load = max(abs(s.load) for s in snaps)
    min_v = min(s.voltage for s in snaps)
    notes = []
    if max_temp > TEMP_PAUSE_C:
        notes.append(f"最高温度 {max_temp}℃ > {TEMP_PAUSE_C}℃，建议先降温")
    if max_load > LOAD_WARN:
        notes.append(f"静态最大 |load| = {max_load} > {LOAD_WARN}，可能已经在受力")
    if min_v < 6.5:
        notes.append(f"最低电压 {min_v}V，电源可能不足（建议 ≥7V）")
    if not notes:
        notes.append("基础指标全部正常")
    for n in notes:
        print(f"  • {n}")
        rep.notes.extend(notes)

    rep.finished_at = time.time()
    return rep


# ---------------------------------------------------------------------------
# Phase 1：末端单轴 ±10° 试探
# ---------------------------------------------------------------------------

def phase1_endeffector(
    arm: Arm,
    *,
    dry_run: bool,
    speed: int,
    acc: int,
) -> PhaseReport:
    rep = PhaseReport(phase="phase1_endeffector", started_at=time.time())
    print("\n[Phase 1] 末端单轴小幅试探（J6 → J5 → J4 各 ±10°）")
    print("=" * 70)

    cur = arm.read_joints_raw()
    rep.snapshots = _read_snapshot(arm)

    if not dry_run:
        print("\n* 开扭矩...")
        arm.set_torque(True)
        cur = arm.read_joints_raw()
        n = len(cur)
        arm.move_joints_raw(cur, speeds=[speed]*n, accs=[acc]*n,
                            bypass_constraints=True)

    DEG_PER_RAW = 360.0 / 4096.0
    delta_raw = int(round(PHASE1_DELTA_DEG / DEG_PER_RAW))

    for idx in (5, 4, 3):
        j = arm.config.joints[idx]
        base_pos = list(arm.read_joints_raw())
        center = base_pos[idx]
        print(f"\n* J{idx+1} ({j.name})  当前 raw={center}")

        for sign, label in ((+1, "正向"), (-1, "负向")):
            tgt_pos = list(base_pos)
            tgt_raw = center + sign * delta_raw
            tgt_raw = max(j._raw_lo, min(j._raw_hi, tgt_raw))
            tgt_pos[idx] = tgt_raw
            r = safe_move(
                arm, tgt_pos,
                speed=speed, acc=acc, timeout_s=PHASE1_TIMEOUT_S,
                label=f"J{idx+1}_{label}_{PHASE1_DELTA_DEG}deg",
                dry_run=dry_run,
            )
            print(f"  {label} {PHASE1_DELTA_DEG}° → raw={tgt_raw}  "
                  f"[{r.status}] {r.detail}  peak_load={r.peak_load.get(j.name, 0)}")
            rep.moves.append(r)

            r2 = safe_move(
                arm, base_pos,
                speed=speed, acc=acc, timeout_s=PHASE1_TIMEOUT_S,
                label=f"J{idx+1}_{label}_return",
                dry_run=dry_run,
            )
            print(f"  回到中心 → [{r2.status}]")
            rep.moves.append(r2)

    rep.finished_at = time.time()
    return rep


# ---------------------------------------------------------------------------
# Phase 2：按 J6→J1 顺序逐关节走到目标姿态
# ---------------------------------------------------------------------------

def phase2_path(
    arm: Arm,
    target_pose: str,
    *,
    dry_run: bool,
    speed: int,
    acc: int,
) -> PhaseReport:
    rep = PhaseReport(phase=f"phase2_path_{target_pose}", started_at=time.time())
    print(f"\n[Phase 2] 路径探索：当前位置 → {target_pose}")
    print("=" * 70)

    if target_pose not in arm.config.poses:
        raise ValueError(
            f"未知姿态 '{target_pose}'，可选：{sorted(arm.config.poses)}"
        )
    target = list(arm.config.poses[target_pose])
    n = len(target)

    rep.snapshots = _read_snapshot(arm)
    cur = [s.raw for s in rep.snapshots]
    diff = [t - c for t, c in zip(target, cur)]
    print(f"  当前 raw: {cur}")
    print(f"  目标 raw: {target}")
    print(f"  Δ raw   : {diff}")

    if not dry_run:
        print("\n* 开扭矩...")
        arm.set_torque(True)
        arm.move_joints_raw(cur, speeds=[speed]*n, accs=[acc]*n,
                            bypass_constraints=True)

    pending: List[int] = list(range(n - 1, -1, -1))
    succeeded: List[int] = []
    failed: List[int] = []

    for round_id in (1, 2):
        if not pending:
            break
        print(f"\n--- 第 {round_id} 轮：尝试关节 {[i+1 for i in pending]} ---")
        retry: List[int] = []
        for idx in pending:
            j = arm.config.joints[idx]
            cur_now = arm.read_joints_raw() if not dry_run else list(cur)
            if abs(target[idx] - cur_now[idx]) <= POS_TOLERANCE_RAW:
                print(f"  J{idx+1}({j.name}) 已在目标 ±{POS_TOLERANCE_RAW} raw 内，跳过")
                succeeded.append(idx)
                continue

            stage_target = list(cur_now)
            stage_target[idx] = target[idx]
            r = safe_move(
                arm, stage_target,
                speed=speed, acc=acc, timeout_s=PHASE2_TIMEOUT_S,
                label=f"R{round_id}_J{idx+1}_to_{target_pose}",
                dry_run=dry_run,
            )
            print(f"  J{idx+1}({j.name}) raw {cur_now[idx]} → {target[idx]}  "
                  f"[{r.status}] {r.detail}")
            print(f"      peak_load: {r.peak_load}")
            rep.moves.append(r)
            if r.status == "ok" or r.status == "skipped":
                succeeded.append(idx)
            else:
                retry.append(idx)
                if r.status in ("overload", "wrong_dir"):
                    print(f"  !! 触发硬保护，本关节本轮放弃")
        pending = retry

    if pending:
        failed = pending
        rep.notes.append(
            f"以下关节未能到达目标：{[i+1 for i in failed]}"
        )
        print(f"\n!! 未到位关节：J{[i+1 for i in failed]}")
    else:
        print(f"\n✓ 全部 6 关节均已到达 {target_pose}")

    final = arm.read_joints_raw() if not dry_run else cur
    rep.notes.append(f"final raw = {final}")
    rep.finished_at = time.time()
    return rep


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _confirm(prompt: str) -> bool:
    try:
        ans = input(prompt + " 输入 yes 继续，其他任意键取消：").strip().lower()
    except EOFError:
        return False
    return ans in ("yes", "y")


def _save_report(rep: PhaseReport, log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.localtime(rep.started_at))
    fp = log_dir / f"explore_{rep.phase}_{stamp}.json"

    def _convert(obj: Any) -> Any:
        if hasattr(obj, "__dataclass_fields__"):
            return asdict(obj)
        return obj

    with open(fp, "w", encoding="utf-8") as f:
        json.dump(asdict(rep), f, ensure_ascii=False, indent=2)
    return fp


def main() -> int:
    global _active_arm

    ap = argparse.ArgumentParser(
        description="自动探索机械臂运动边界（堵转检测 + 慢速渐进）",
    )
    ap.add_argument("--phase", type=int, choices=(0, 1, 2), required=True)
    ap.add_argument("--target", choices=("home", "ready", "safe"), default=None,
                    help="phase 2 的目标姿态")
    ap.add_argument("--speed", type=int, default=DEFAULT_SPEED,
                    help=f"探索速度（步/秒，硬上限 {SAFE_MAX_SPEED}）")
    ap.add_argument("--acc", type=int, default=DEFAULT_ACC,
                    help=f"加速度（硬上限 {SAFE_MAX_ACC}）")
    ap.add_argument("--dry-run", action="store_true",
                    help="不真发命令，只读 + 打印")
    ap.add_argument("--confirm", action="store_true",
                    help="跳过交互式确认（用于无人值守）")
    ap.add_argument("--log-dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "logs",
                    help="JSON 报告输出目录")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    if args.phase == 2 and args.target is None:
        ap.error("--phase 2 需要 --target {home,ready,safe}")

    speed = min(args.speed, SAFE_MAX_SPEED)
    acc = min(args.acc, SAFE_MAX_ACC)

    print("=" * 70)
    print(f"  探索机械臂运动边界  phase={args.phase}  "
          f"target={args.target or '-'}  dry_run={args.dry_run}")
    print(f"  速度限制 ≤ {SAFE_MAX_SPEED}（本次 {speed}），加速度 ≤ {SAFE_MAX_ACC}（本次 {acc}）")
    print(f"  load 警告 {LOAD_WARN} / 立停 {LOAD_STOP}（持续 {STALL_HOLD_MS}ms）")
    print(f"  位置堵转 {NO_MOVE_RAW}raw / {NO_MOVE_HOLD_MS}ms")
    print("  ★ 任意时刻 Ctrl+C 可立刻紧急停止")
    print("=" * 70)

    if args.phase >= 1 and not args.dry_run and not args.confirm:
        if not _confirm("\n确认开始执行（机械臂将真实运动）？"):
            print("已取消")
            return 0

    signal.signal(signal.SIGINT, _emergency_handler)
    signal.signal(signal.SIGTERM, _emergency_handler)

    cfg = ArmConfig.from_yaml(CONFIG_PATH)
    rep: Optional[PhaseReport] = None
    with Arm(cfg) as arm:
        _active_arm = arm
        try:
            if args.phase == 0:
                rep = phase0_inspect(arm)
            elif args.phase == 1:
                rep = phase1_endeffector(
                    arm, dry_run=args.dry_run, speed=speed, acc=acc,
                )
            else:
                rep = phase2_path(
                    arm, args.target,
                    dry_run=args.dry_run, speed=speed, acc=acc,
                )
        finally:
            if args.phase >= 1 and not args.dry_run:
                print("\n* 收尾：emergency_stop（卸扭矩 + 冻结目标）")
                try:
                    arm.emergency_stop()
                except Exception as e:
                    print(f"emergency_stop 失败：{e}")
            _active_arm = None

    if rep is not None:
        fp = _save_report(rep, args.log_dir)
        print(f"\n✓ 报告已保存：{fp}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
