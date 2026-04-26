"""把当前每颗舵机的位置标定为零位（写入 config 的 home_raw）。

仅改软件层标定，**不写舵机 EPROM**，可随时重新标定。

用法：
    python -m tools.calibrate_home --port /dev/serial0
    python -m tools.calibrate_home --dry-run    # 只读取并打印，不修改文件

执行流程：
    1) ping 全部关节，确认在线
    2) 读取每颗当前 raw 位置
    3) 显示新旧 home_raw 对比
    4) 备份原 yaml 到 .bak，写入新值（用文本替换，保留注释/顺序）
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

from arm.arm import Arm, ArmConfig
from arm.bus import autodetect_port


def update_yaml_home_raw(text: str, name_to_raw: dict[str, int]) -> str:
    """对每个 joint name 找到其后面最近的 home_raw 行并替换数值。

    依赖 yaml 的标准格式：
        - name: foo
          ...
          home_raw: 1234
    """
    out = text
    for name, raw in name_to_raw.items():
        pattern = re.compile(
            r"(-\s*name:\s*" + re.escape(name) + r"\b[\s\S]*?home_raw:\s*)\d+",
        )
        new_out, n = pattern.subn(lambda m, r=raw: f"{m.group(1)}{r}", out, count=1)
        if n != 1:
            raise RuntimeError(f"yaml 中找不到关节 {name!r} 的 home_raw 字段")
        out = new_out
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="软件层零位标定（写入 home_raw）")
    ap.add_argument("--port", default=None, help="串口设备（默认按平台自动检测）")
    ap.add_argument(
        "--config",
        default="config/arm_config.yaml",
        help="配置文件路径（默认 config/arm_config.yaml）",
    )
    ap.add_argument("-y", "--yes", action="store_true", help="跳过交互确认")
    ap.add_argument("--dry-run", action="store_true", help="只读取并打印，不写入")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"✗ 配置文件不存在: {cfg_path}", file=sys.stderr)
        return 2

    cfg = ArmConfig.from_yaml(cfg_path)
    if args.port:
        cfg.port = args.port
    elif cfg.port == "auto":
        cfg.port = autodetect_port() or "/dev/serial0"

    print(f"→ 串口 {cfg.port} @ {cfg.baudrate} bps")
    print(f"  配置文件 {cfg_path}")
    print()

    with Arm(cfg) as arm:
        ping = arm.ping_all()
        offline = [sid for sid, ok in ping.items() if not ok]
        if offline:
            print(f"✗ 以下舵机不在线: {offline}", file=sys.stderr)
            return 1
        print(f"  已 ping 通：{sorted(ping.keys())}")
        print()

        try:
            raws = arm.read_joints_raw()
        except Exception as e:
            print(f"✗ 读取位置失败: {e}", file=sys.stderr)
            return 1

    name_to_raw: dict[str, int] = {}
    print(
        f"  {'关节':<14s} {'ID':>3s}"
        f" {'旧 home_raw':>12s} {'新 home_raw':>12s}  delta"
    )
    for j, r in zip(cfg.joints, raws):
        delta = r - j.home_raw
        print(
            f"  {j.name:<14s} {j.id:>3d}"
            f" {j.home_raw:>12d} {r:>12d}  {delta:+d}"
        )
        name_to_raw[j.name] = r
    print()

    if args.dry_run:
        print("• --dry-run，未写入文件")
        return 0

    if not args.yes:
        confirm = input("  确认写入到配置文件？[y/N] ").strip().lower()
        if confirm not in ("y", "yes"):
            print("已取消")
            return 0

    original = cfg_path.read_text(encoding="utf-8")
    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")
    print(f"  已备份到 {backup}")

    try:
        new_text = update_yaml_home_raw(original, name_to_raw)
    except RuntimeError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 1

    cfg_path.write_text(new_text, encoding="utf-8")
    print(f"✓ 已更新 {cfg_path}")
    print()
    print("  现在所有关节当前位置 = 0°，可用：")
    print("    python -m tools.scan          # 验证")
    print("    arm.move_home()               # 软件回零")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
