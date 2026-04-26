"""键盘点动测试工具。

终端实时控制单关节，用于快速验证机械结构、关节方向、限位是否合理。

按键：
    1..6     选择关节（也可用 ←/→ 切换）
    q / a    增大 / 减小当前关节位置（粗调，每次 ±50 raw ≈ ±4.4°）
    w / s    增大 / 减小当前关节位置（细调，每次 ±5 raw ≈ ±0.44°）
    h        全部关节回零位（home）
    t        切换扭矩（开/关）—— 关闭后可手动摆动
    r        刷新一次反馈
    SPACE    紧急停止（放扭矩）
    ESC / x  退出

用法：
    python -m tools.jog
    python -m tools.jog --port /dev/cu.usbserial-XYZ
    python -m tools.jog --config config/arm_config.yaml
"""

from __future__ import annotations

import argparse
import os
import select
import sys
import termios
import tty
from contextlib import contextmanager
from pathlib import Path

from arm.arm import Arm, ArmConfig
from arm.servo import POSITION_RESOLUTION


# ---------------------------------------------------------------------------
# 终端原始模式 + 单字符读取（仅 Unix；Windows 暂不支持）
# ---------------------------------------------------------------------------

@contextmanager
def cbreak_mode():
    if not sys.stdin.isatty():
        raise RuntimeError("jog 需要在交互式终端运行")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout: float = 0.1) -> str:
    """非阻塞读取一个按键；超时返回空字符串。处理方向键 (ESC '[' 'C'/'D')。"""
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ""
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    # ESC 序列：再读 2 字节
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not rlist:
        return "\x1b"
    seq = sys.stdin.read(2)
    if seq == "[C":
        return "→"
    if seq == "[D":
        return "←"
    if seq == "[A":
        return "↑"
    if seq == "[B":
        return "↓"
    return "\x1b"


# ---------------------------------------------------------------------------
# 主循环
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="键盘点动测试")
    ap.add_argument("--port", default=None, help="覆盖配置中的串口")
    ap.add_argument(
        "--config",
        default="config/arm_config.yaml",
        help="机械臂配置文件路径",
    )
    return ap.parse_args()


def render_status(arm: Arm, sel: int, torque_on: bool, msg: str) -> None:
    """重绘状态（用 ANSI escape 原地刷新）。"""
    sys.stdout.write("\x1b[2J\x1b[H")
    print("=" * 76)
    print("  键盘点动  |  扭矩:", "ON " if torque_on else "OFF", " |  ESC/x 退出")
    print("=" * 76)
    try:
        positions = arm.read_joints_raw()
        pcts = [j.raw_to_pct(p) for j, p in zip(arm.config.joints, positions)]
    except Exception as e:
        print(f"  读位置失败：{e}")
        positions = [0] * len(arm.config.joints)
        pcts = [0.0] * len(arm.config.joints)

    for i, (j, raw, pct) in enumerate(zip(arm.config.joints, positions, pcts)):
        marker = "▶" if i == sel else " "
        bar_pos = int(round(pct * 30))
        bar = "[" + "█" * bar_pos + "·" * (30 - bar_pos) + "]"
        deg = j.raw_to_deg(raw)
        deg_str = f"{deg:+6.1f}°" if deg is not None else "    -  "
        print(
            f" {marker} J{i+1} {j.name:<11} ID={j.id}  raw={raw:4d} "
            f"{bar} {pct*100:5.1f}%  {deg_str}"
        )
    print("-" * 76)
    print("  1..6/←→ 选关节   q/a 粗调  w/s 细调   h 回 home")
    print("  t 切扭矩         r 刷新     空格 急停   ESC/x 退出")
    if msg:
        print(f"\n  {msg}")
    sys.stdout.flush()


STEP_COARSE = 50
STEP_FINE = 5


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"✗ 配置文件不存在：{cfg_path}", file=sys.stderr)
        return 2
    cfg = ArmConfig.from_yaml(cfg_path)
    if args.port:
        cfg.port = args.port

    sel = 0
    torque_on = False
    msg = "已就绪"

    with Arm(cfg) as arm:
        # 初始读一次，构造目标位置（避免动作"跳"）
        try:
            targets = arm.read_joints_raw()
        except Exception as e:
            print(f"✗ 启动时读位置失败：{e}", file=sys.stderr)
            return 1

        # 默认放扭矩，便于先手动摆姿势
        arm.set_torque(False)

        with cbreak_mode():
            render_status(arm, sel, torque_on, msg)
            while True:
                key = read_key(timeout=0.15)
                if not key:
                    render_status(arm, sel, torque_on, msg)
                    continue

                if key in ("\x1b", "x", "X"):
                    break

                if key in "123456":
                    idx = int(key) - 1
                    if idx < len(arm.config.joints):
                        sel = idx
                        msg = f"已选 J{sel+1} ({arm.config.joints[sel].name})"
                elif key == "←":
                    sel = (sel - 1) % len(arm.config.joints)
                    msg = f"已选 J{sel+1}"
                elif key == "→":
                    sel = (sel + 1) % len(arm.config.joints)
                    msg = f"已选 J{sel+1}"
                elif key in ("q", "a", "w", "s"):
                    if not torque_on:
                        msg = "⚠ 扭矩未开启，按 t 打开扭矩后再点动"
                    else:
                        step = STEP_COARSE if key in ("q", "a") else STEP_FINE
                        sign = +1 if key in ("q", "w") else -1
                        targets[sel] = max(
                            0,
                            min(POSITION_RESOLUTION - 1, targets[sel] + sign * step),
                        )
                        try:
                            arm.servos[sel].write_position(
                                targets[sel],
                                speed=arm.config.joints[sel].max_speed,
                                acc=arm.config.joints[sel].max_acc,
                            )
                            msg = f"J{sel+1} → raw {targets[sel]}"
                        except Exception as e:
                            msg = f"写位置失败：{e}"
                elif key == "h":
                    try:
                        arm.move_home(speed=600, acc=30)
                        targets = [j.home_raw for j in arm.config.joints]
                        msg = "→ 回零位中..."
                    except Exception as e:
                        msg = f"回零失败：{e}"
                elif key == "t":
                    torque_on = not torque_on
                    try:
                        arm.set_torque(torque_on)
                        msg = "扭矩已" + ("打开" if torque_on else "关闭（可手动摆动）")
                        if torque_on:
                            # 打开扭矩前同步目标位置 = 当前位置，避免突跳
                            targets = arm.read_joints_raw()
                    except Exception as e:
                        msg = f"切扭矩失败：{e}"
                elif key == "r":
                    msg = "已刷新"
                elif key == " ":
                    try:
                        arm.emergency_stop()
                        torque_on = False
                        msg = "⚠ 紧急停止：扭矩已放开"
                    except Exception as e:
                        msg = f"急停失败：{e}"

                render_status(arm, sel, torque_on, msg)

        # 退出前默认放扭矩
        try:
            arm.set_torque(False)
        except Exception:
            pass
    print("\n已退出。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
