"""批量按映射表改 ID（用于把临时两位数 ID 整理回 1..N）。

用法（示例：把 11..16 → 1..6）：
    python -m tools.reset_ids --map 11:1,12:2,13:3,14:4,15:5,16:6 -y

对每个 src→dst 对：
    1) 必须确认 src 在线
    2) 必须确认 dst **不**在线（避免目标 ID 已被占用）
    3) 执行 set_id

修改顺序很重要：如果有"环形交换"（A→B 同时 B→A），脚本会通过临时 ID 两次跳跃完成。
本工具会先做完整的预检（dry-run），确认安全后再实际写入。
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Dict, List, Tuple

from arm.bus import Bus, autodetect_port
from arm.servo import Servo


def parse_map(s: str) -> Dict[int, int]:
    out: Dict[int, int] = {}
    for pair in s.split(","):
        pair = pair.strip()
        if not pair:
            continue
        src, dst = pair.split(":")
        src_i, dst_i = int(src), int(dst)
        if not (0 <= src_i <= 253 and 0 <= dst_i <= 253):
            raise ValueError(f"id out of range in pair: {pair}")
        if src_i in out:
            raise ValueError(f"duplicate src id: {src_i}")
        out[src_i] = dst_i
    return out


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="批量改 SMS_STS 舵机 ID")
    ap.add_argument("--port", default=None)
    ap.add_argument("--baud", type=int, default=1_000_000)
    ap.add_argument(
        "--map",
        required=True,
        help="ID 映射，格式 src:dst,src:dst,...   例如 11:1,12:2,13:3",
    )
    ap.add_argument("-y", "--yes", action="store_true", help="跳过确认")
    return ap.parse_args()


def find_safe_temp_id(in_use: set[int]) -> int:
    """挑一个肯定没被占用的临时 ID。"""
    for cand in range(200, 254):
        if cand not in in_use:
            return cand
    raise RuntimeError("找不到空闲临时 ID（200..253 都被占用）")


def main() -> int:
    args = parse_args()
    try:
        mapping = parse_map(args.map)
    except ValueError as e:
        print(f"✗ {e}", file=sys.stderr)
        return 2

    port = args.port or autodetect_port()
    if not port:
        print("✗ 未找到串口", file=sys.stderr)
        return 2

    print(f"→ 串口 {port} @ {args.baud} bps")
    print(f"  映射：{', '.join(f'{s}→{d}' for s, d in mapping.items())}")
    print()

    with Bus(port=port, baudrate=args.baud) as bus:
        # ---- 预检 ----
        print("• 预检...")
        online: set[int] = set()
        for sid in range(0, 254):
            if bus.ping(sid):
                online.add(sid)
        print(f"  当前在线：{sorted(online)}")

        for src, dst in mapping.items():
            if src not in online:
                print(f"✗ src ID {src} 不在线", file=sys.stderr)
                return 1

        # 检测目标 ID 冲突：
        # 如果 dst 已经在线，且 dst 不是任何一个 src（即不是被改走的），就冲突
        srcs = set(mapping.keys())
        for src, dst in mapping.items():
            if dst in online and dst not in srcs:
                print(
                    f"✗ 目标 ID {dst} 已被一颗未参与映射的舵机占用，无法继续",
                    file=sys.stderr,
                )
                return 1

        # ---- 排序：避免覆盖（先改"目标空闲"的，后改可能冲突的） ----
        # 拓扑：把 src→dst 当作边，从入度 0 的节点开始改。
        # 简化处理：循环中如果有环，用一个临时 ID 中转。
        plan: List[Tuple[int, int]] = []
        remaining = dict(mapping)
        in_use_after = set(online)
        while remaining:
            # 找一个 dst 当前不在线（或它就是某个 src 但已被改走）的对
            progress = False
            for src in list(remaining):
                dst = remaining[src]
                if dst not in in_use_after or dst == src:
                    plan.append((src, dst))
                    in_use_after.discard(src)
                    in_use_after.add(dst)
                    del remaining[src]
                    progress = True
                    break
            if not progress:
                # 有环：随便挑一对，先改到临时 ID
                src = next(iter(remaining))
                temp = find_safe_temp_id(in_use_after | set(remaining.values()))
                plan.append((src, temp))
                # 更新：原 src→dst 改为 temp→dst
                final_dst = remaining.pop(src)
                remaining[temp] = final_dst
                in_use_after.discard(src)
                in_use_after.add(temp)

        print("\n• 计划执行顺序：")
        for src, dst in plan:
            print(f"  {src:3d} → {dst:3d}")

        if not args.yes:
            confirm = input("\n  确认执行？[y/N] ").strip().lower()
            if confirm not in ("y", "yes"):
                print("已取消")
                return 0

        # ---- 执行 ----
        print()
        for src, dst in plan:
            if src == dst:
                continue
            print(f"  {src:3d} → {dst:3d} ... ", end="", flush=True)
            try:
                servo = Servo(bus, src)
                servo.set_id(dst)
                time.sleep(0.05)
                if bus.ping(dst):
                    print("✓")
                else:
                    print("✗ 验证失败")
                    return 1
            except Exception as e:
                print(f"✗ {e}")
                return 1

        # ---- 最终验证 ----
        print("\n• 最终扫描验证：")
        final_online = sorted(sid for sid in range(0, 254) if bus.ping(sid))
        print(f"  在线 IDs：{final_online}")

        expected = sorted(mapping.values()) + sorted(online - set(mapping.keys()))
        expected_unique = sorted(set(expected))
        if set(final_online) == set(expected_unique):
            print("✓ 全部 ID 与预期一致")
            return 0
        print(f"⚠ 与预期 {expected_unique} 不一致", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
