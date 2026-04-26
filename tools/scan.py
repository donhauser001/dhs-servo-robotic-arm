"""扫描总线上的舵机。

用法：
    python -m tools.scan                       # 自动选串口，扫描 ID 1..20
    python -m tools.scan --port /dev/cu.usbserial-XYZ
    python -m tools.scan --range 0 253         # 全 ID 段扫描（慢）
    python -m tools.scan --port COM4 --baud 1000000
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import List

from arm.bus import Bus, autodetect_port
from arm.protocol import ProtocolError, TimeoutError_
from arm.servo import Servo


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="扫描 SMS_STS 总线舵机")
    ap.add_argument("--port", default=None, help="串口设备（默认自动检测）")
    ap.add_argument("--baud", type=int, default=1_000_000, help="波特率（默认 1Mbps）")
    ap.add_argument(
        "--range",
        nargs=2,
        type=int,
        default=[1, 20],
        metavar=("LO", "HI"),
        help="扫描的 ID 区间，含两端（默认 1..20）",
    )
    ap.add_argument("--timeout", type=float, default=0.05, help="单次响应超时（秒）")
    ap.add_argument("--verbose", "-v", action="store_true", help="显示完整反馈")
    return ap.parse_args()


def fmt_table(headers: List[str], rows: List[List[str]]) -> str:
    """简单的对齐表格输出。"""
    cols = len(headers)
    widths = [len(h) for h in headers]
    for row in rows:
        for i in range(cols):
            widths[i] = max(widths[i], len(str(row[i])))
    sep = "  "

    def fmt_row(r: List[str]) -> str:
        return sep.join(str(r[i]).ljust(widths[i]) for i in range(cols))

    out = [fmt_row(headers), sep.join("-" * w for w in widths)]
    out.extend(fmt_row(r) for r in rows)
    return "\n".join(out)


def main() -> int:
    args = parse_args()
    port = args.port or autodetect_port()
    if not port:
        print("✗ 未找到可用串口。请确认：", file=sys.stderr)
        print("  1. 驱动板已通过 USB 连接（USB 档），或树莓派 GPIO UART 已启用（ESP32 档）", file=sys.stderr)
        print("  2. macOS 用户可能需要装 CH343 驱动：https://www.wch.cn/downloads/CH343SER_MAC_ZIP.html", file=sys.stderr)
        print("  3. 用 `ls /dev/cu.* /dev/tty.usb*` 确认设备名后用 --port 指定", file=sys.stderr)
        return 2

    lo, hi = args.range
    print(f"→ 串口 {port} @ {args.baud} bps，扫描 ID {lo}..{hi}\n")

    found_rows: List[List[str]] = []
    t0 = time.monotonic()
    try:
        with Bus(port=port, baudrate=args.baud, timeout=args.timeout) as bus:
            for sid in range(lo, hi + 1):
                ok = bus.ping(sid)
                if not ok:
                    continue
                # 在线，读完整反馈
                row: List[str] = [str(sid)]
                try:
                    fb = Servo(bus, sid).feedback()
                    row += [
                        f"{fb.position}",
                        f"{fb.position_deg:+.1f}°",
                        f"{fb.voltage:.1f} V",
                        f"{fb.temperature} ℃",
                        f"{fb.load:+d}",
                        f"{fb.current_ma:.0f} mA" if fb.current_ma is not None else "—",
                        "运动中" if fb.moving else "静止",
                    ]
                except (TimeoutError_, ProtocolError) as e:
                    row += [f"反馈失败: {e}", "", "", "", "", "", ""]
                found_rows.append(row)
                if args.verbose:
                    print(f"  ID {sid:3d}: {row[1:]}")
    except Exception as e:
        print(f"✗ 通信错误：{e}", file=sys.stderr)
        return 1

    elapsed = time.monotonic() - t0
    print()
    if not found_rows:
        print(f"⚠  未发现任何舵机（耗时 {elapsed:.2f}s）")
        print("  排查思路：")
        print("    • 总线供电（XT60 / DC 接口）是否接好 12V")
        print("    • 6P 排线方向是否正确，舵机两个接口都可串联")
        print("    • 出厂 ID 全为 1，多颗同时连接会冲突，请逐颗扫描或先改 ID")
        print("    • 用 `python -m tools.scan --range 0 253` 全段扫描看看")
        return 1

    headers = ["ID", "位置", "角度", "电压", "温度", "负载", "电流", "状态"]
    print(fmt_table(headers, found_rows))
    print(f"\n✓ 共发现 {len(found_rows)} 颗舵机（耗时 {elapsed:.2f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
