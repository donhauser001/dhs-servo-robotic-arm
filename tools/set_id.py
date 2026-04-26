"""安全地设置（修改）一颗舵机的 ID。

⚠ **警告**：调用前请确保总线上**只接了这一颗**要改 ID 的舵机！
出厂默认 ID 都是 1，多颗同时挂在总线上时改 ID 会冲突。

用法：
    # 把当前唯一在线的舵机（默认探测 ID=1）改成 ID=2
    python -m tools.set_id --to 2

    # 显式指定原 ID
    python -m tools.set_id --from 5 --to 6

    # 自定义串口
    python -m tools.set_id --port /dev/cu.usbserial-XYZ --to 3

典型流程（首次给 6 颗舵机分配 1..6）：
    1) 只接第 1 颗（ID=1，默认）→ python -m tools.set_id --to 1   # 确认它在
    2) 拔掉，换上第 2 颗（仍是出厂 ID=1）→ python -m tools.set_id --to 2
    3) 拔掉，换上第 3 颗 → python -m tools.set_id --to 3
    4) ... 依此类推到 ID=6
    5) 把 6 颗全部串起来再 `python -m tools.scan` 验证
"""

from __future__ import annotations

import argparse
import sys
import time

from arm.bus import Bus, autodetect_port
from arm.servo import Servo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="修改总线舵机的 ID（永久写入 EPROM）")
    ap.add_argument("--port", default=None, help="串口设备（默认自动检测）")
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument(
        "--from",
        dest="from_id",
        type=int,
        default=None,
        help="当前舵机 ID；省略时自动扫描 0..253 找到唯一一颗",
    )
    ap.add_argument("--to", dest="to_id", type=int, required=True, help="目标 ID（0..253）")
    ap.add_argument("-y", "--yes", action="store_true", help="跳过二次确认")
    return ap.parse_args()


def find_only_servo(bus: Bus) -> int:
    """全段扫描，要求总线上只有一颗舵机；返回它的 ID。"""
    found: list[int] = []
    for sid in range(0, 254):
        if bus.ping(sid):
            found.append(sid)
            if len(found) > 1:
                break
    if not found:
        raise RuntimeError("未发现任何舵机（请检查总线供电与接线）")
    if len(found) > 1:
        raise RuntimeError(
            f"总线上同时存在多颗舵机 {found}：请先断开其它舵机，只留一颗后再操作"
        )
    return found[0]


def main() -> int:
    args = parse_args()
    if not 0 <= args.to_id <= 253:
        print("✗ --to 必须在 0..253 范围内", file=sys.stderr)
        return 2

    port = args.port or autodetect_port()
    if not port:
        print("✗ 未找到串口", file=sys.stderr)
        return 2

    print(f"→ 串口 {port} @ {args.baud} bps")
    with Bus(port=port, baudrate=args.baud) as bus:
        if args.from_id is None:
            print("• 自动扫描中...")
            current_id = find_only_servo(bus)
            print(f"  发现舵机 ID = {current_id}")
        else:
            current_id = args.from_id
            if not bus.ping(current_id):
                print(f"✗ ID={current_id} 的舵机未响应", file=sys.stderr)
                return 1

        if current_id == args.to_id:
            print(f"✓ 当前 ID 已经是 {args.to_id}，无需修改")
            return 0

        if not args.yes:
            print(
                f"\n  即将把 ID {current_id} 改为 {args.to_id}（永久写入 EPROM）"
            )
            confirm = input("  请确认总线上只接了这一颗舵机，继续？[y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("已取消")
                return 0

        servo = Servo(bus, current_id)
        servo.set_id(args.to_id)
        time.sleep(0.05)

        # 验证
        if bus.ping(args.to_id):
            print(f"✓ 修改成功：ID = {args.to_id}")
            return 0
        print(f"✗ 修改后 ping 不通：ID = {args.to_id}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
