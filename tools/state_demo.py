"""整机状态机演示：上电 → enable(走 ready) → 演示动作 → disable(回 safe)。

完整跑一遍 :class:`arm.arm.Arm` 的两态机：

    DISABLED ──enable()──> ENABLED ──disable()──> DISABLED

中间在 ``ready`` 姿态基础上做一次温和的"逐关节运动"——J1..J5 依次小幅
摆动后回 ready，用来直观验证扭矩、目标控制、wait_until_idle 都通了。

用法：
    python -m tools.state_demo
    python -m tools.state_demo --skip-wave        # 只跑 enable / disable
    python -m tools.state_demo --joints 3         # 只演示 J1..J3
    python -m tools.state_demo --stop-to home_raw # 用 move_home() 而非 safe pose 收尾
    python -m tools.state_demo --stop-to none     # 直接放扭矩，不走任何姿态

使用前提：
    config/arm_config.yaml 里同时定义了 ``ready`` 和 ``safe`` 两个 pose。
    机械臂周围有足够空间，演示动作不会撞到东西。
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from arm.arm import Arm, ArmConfig, ArmState


# ---------------------------------------------------------------------------
# 小工具
# ---------------------------------------------------------------------------

def _print_state(arm: Arm, banner: str) -> None:
    """打印一行状态摘要。"""
    print(f"\n── {banner} ─────────────────────────────────────────")
    print(f"  state = {arm.state.name}")
    try:
        positions = arm.read_joints_raw()
    except Exception as e:
        print(f"  读位置失败：{e}")
        return
    for j, raw in zip(arm.config.joints, positions):
        deg = j.raw_to_deg(raw)
        deg_str = f"{deg:+6.1f}°" if deg is not None else "    -  "
        pct = j.raw_to_pct(raw)
        print(
            f"  {j.name:<11} ID={j.id}  raw={raw:4d}  pct={pct*100:+6.1f}%  {deg_str}"
        )


def _confirm(prompt: str, *, yes: bool) -> bool:
    """交互式确认；``--yes`` 时直接返回 True。"""
    if yes:
        print(f"{prompt} [自动 yes]")
        return True
    try:
        ans = input(f"{prompt} [y/N] ").strip().lower()
    except EOFError:
        return False
    return ans in ("y", "yes")


# ---------------------------------------------------------------------------
# 演示动作：J1..Jk 逐关节摆动一次
# ---------------------------------------------------------------------------

def exercise_demo(
    arm: Arm,
    *,
    joint_count: int = 5,
    hold_s: float = 0.5,
    speed: int = 800,
    acc: int = 40,
) -> None:
    """让 J1 .. J``joint_count`` 依次小幅摆动，再回 ready。

    每个关节摆动方向根据它当前在 ready 位置的百分比自动选择：

        |ready_pct| > 0.2  →  朝 home（pct=0）走，再回 ready
        否则               →  朝 +30% 走，再回 ready

    这样保证 J1..J5 都能看到可见运动，且永远在软限位内。
    """
    ready_raw = list(arm.read_joints_raw())
    n = len(arm.config.joints)
    k = min(joint_count, n)

    for i in range(k):
        j = arm.config.joints[i]
        ready_pct = j.raw_to_pct(ready_raw[i])
        target_pct = 0.0 if abs(ready_pct) > 0.2 else 0.3
        target_raw = j.pct_to_raw(target_pct)

        sequence = [
            (f"摆到 pct={target_pct:+.0%} (raw={target_raw})", target_raw),
            ("回 ready", ready_raw[i]),
        ]
        for label, raw_val in sequence:
            cmd = list(ready_raw)
            cmd[i] = raw_val
            print(f"  J{i+1} {j.name:<12} → {label}")
            arm.move_joints_raw(cmd, speeds=[speed] * n, accs=[acc] * n)
            arm.wait_until_idle(timeout=5.0)
            time.sleep(hold_s)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="机械臂状态机演示（enable / 动作 / disable）")
    ap.add_argument(
        "--config",
        default="config/arm_config.yaml",
        help="机械臂配置文件路径",
    )
    ap.add_argument("--port", default=None, help="覆盖配置中的串口")
    ap.add_argument(
        "--stop-to",
        default="safe",
        choices=("safe", "ready", "home_raw", "none"),
        help="收尾时的目标：'safe'(默认) / 'ready' / 'home_raw'(走 move_home) / 'none'(直接放扭矩)",
    )
    ap.add_argument("--skip-wave", action="store_true", help="跳过逐关节运动演示")
    ap.add_argument(
        "--joints",
        type=int,
        default=5,
        help="演示前 N 个关节（默认 5，即 J1..J5；J6 一般是夹爪不参与）",
    )
    ap.add_argument("-y", "--yes", action="store_true", help="跳过所有交互确认")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"✗ 配置文件不存在：{cfg_path}", file=sys.stderr)
        return 2
    cfg = ArmConfig.from_yaml(cfg_path)
    if args.port:
        cfg.port = args.port

    # 必备 pose 校验
    needed = {"ready"} if args.stop_to != "safe" else {"ready", "safe"}
    missing = sorted(needed - set(cfg.poses))
    if missing:
        print(f"✗ 配置缺少 pose：{missing}（请先在 config/arm_config.yaml 标定）", file=sys.stderr)
        return 2

    print("================================================================")
    print("  机械臂状态机演示")
    print(f"  config    : {cfg_path}")
    print(f"  port      : {cfg.port}")
    print(f"  stop_to   : {args.stop_to}")
    print(f"  动作演示  : {'否' if args.skip_wave else f'是 (J1..J{args.joints} 逐关节)'}")
    print("================================================================")

    with Arm(cfg) as arm:
        assert arm.state == ArmState.DISABLED
        _print_state(arm, "初始（DISABLED，扭矩 OFF）")

        if not _confirm("即将 enable()，机械臂会缓慢走到 ready。继续？", yes=args.yes):
            print("已取消。")
            return 0

        # 1) enable → ready
        print("\n[1/3] enable() ...")
        arm.enable(speed=600, acc=30)
        _print_state(arm, "已 enable（ENABLED，已在 ready）")

        # 2) 演示动作
        if not args.skip_wave:
            print(f"\n[2/3] exercise demo (J1..J{args.joints}) ...")
            try:
                exercise_demo(arm, joint_count=args.joints)
            except Exception as e:
                print(f"  ⚠ 动作失败：{e} —— 直接进入收尾")
            _print_state(arm, "演示结束（仍 ENABLED）")
        else:
            print("\n[2/3] (跳过动作演示)")

        # 3) 收尾
        print(f"\n[3/3] 收尾：stop-to={args.stop_to}")
        try:
            if args.stop_to == "none":
                arm.disable(to=None)
            elif args.stop_to == "home_raw":
                arm.move_home(speed=400, acc=30)
                arm.wait_until_idle(timeout=10.0)
                arm.disable(to=None)
            else:
                arm.disable(to=args.stop_to, speed=400, acc=30)
        except Exception as e:
            print(f"  ✗ 收尾失败：{e}，执行紧急停止")
            try:
                arm.emergency_stop()
            except Exception:
                pass
            return 1

        _print_state(arm, f"已 disable（DISABLED，扭矩 OFF，目标={args.stop_to}）")

    print("\n演示完成。")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n^C 已中断。建议手动 `python -m tools.jog` 进入空载模式后再操作。")
        sys.exit(130)
