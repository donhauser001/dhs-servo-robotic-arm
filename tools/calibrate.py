"""交互式标定：home (0%) + max (+100%) + 可选 low (-100%)。

每关节按需标 2 或 3 个点：
    home → 0%      （初始位 / 上电默认）
    max  → +100%   （正向极限）
    low  → -100%   （负向极限，仅双向关节，例如旋转关节才需要）

工作流：
    1) 启动时关闭所有关节扭矩，可手动摆动
    2) home 通常已在配置里（之前标过）。如要重标，把关节摆到位按 ``h``
    3) 把关节摆到正向极限按 ``m`` 记 max
    4) 旋转类关节再摆到反向极限按 ``l`` 记 low（抬升类跳过）
    5) 按 ``L``（大写）可清除 low 标定（把双向关节改回单向）
    6) 全部标完后按 ``s`` 保存（自动备份 .bak）

按键：
    1..6 / ←/→   选关节
    h             记 ★home★（0%）
    m             记 ★max★（+100%）
    l             记 ★low★（-100%，仅双向关节用）
    L             清除 low（变回单向）
    t             切换扭矩
    r             刷新读取
    s             保存退出
    q / x / ESC   放弃退出
    SPACE         紧急停止

用法：
    python -m tools.calibrate
    python -m tools.calibrate --port /dev/serial0
"""

from __future__ import annotations

import argparse
import re
import select
import sys
import termios
import tty
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from arm.arm import Arm, ArmConfig, JointConfig


# ---------------------------------------------------------------------------
# 终端输入
# ---------------------------------------------------------------------------

@contextmanager
def cbreak_mode():
    if not sys.stdin.isatty():
        raise RuntimeError("calibrate 需要在交互式终端运行")
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def read_key(timeout: float = 0.15) -> str:
    rlist, _, _ = select.select([sys.stdin], [], [], timeout)
    if not rlist:
        return ""
    ch = sys.stdin.read(1)
    if ch != "\x1b":
        return ch
    rlist, _, _ = select.select([sys.stdin], [], [], 0.05)
    if not rlist:
        return "\x1b"
    seq = sys.stdin.read(2)
    return {"[C": "→", "[D": "←", "[A": "↑", "[B": "↓"}.get(seq, "\x1b")


# ---------------------------------------------------------------------------
# 标定状态
# ---------------------------------------------------------------------------

@dataclass
class JointCal:
    name: str
    id: int
    home_raw: Optional[int] = None
    max_raw: Optional[int] = None
    low_raw: Optional[int] = None         # 配则双向，None 则单向

    @classmethod
    def from_config(cls, jc: JointConfig) -> "JointCal":
        return cls(
            name=jc.name,
            id=jc.id,
            home_raw=jc.home_raw,
            max_raw=jc.max_raw,
            low_raw=jc.low_raw,
        )

    def is_complete(self) -> bool:
        if self.home_raw is None or self.max_raw is None or self.home_raw == self.max_raw:
            return False
        # low 是可选的；如果设了，必须和 max 在 home 异侧
        if self.low_raw is not None:
            if self.low_raw == self.home_raw:
                return False
            pos_dir = 1 if self.max_raw > self.home_raw else -1
            neg_dir = 1 if self.low_raw > self.home_raw else -1
            if pos_dir == neg_dir:
                return False
        return True

    def is_modified(self, jc: JointConfig) -> bool:
        return (
            self.home_raw != jc.home_raw
            or self.max_raw != jc.max_raw
            or self.low_raw != jc.low_raw
        )


# ---------------------------------------------------------------------------
# 渲染
# ---------------------------------------------------------------------------

def _bar(pct: float, width: int = 30) -> str:
    """在 -1..+1 范围内渲染条；中点是 home。"""
    pct = max(-1.0, min(1.0, pct))
    half = width // 2
    cells = ["·"] * width
    cells[half] = "│"  # home 标记
    if pct >= 0:
        end = half + int(round(pct * half))
        for i in range(half, end + 1):
            cells[i] = "█"
    else:
        end = half + int(round(pct * half))
        for i in range(end, half + 1):
            cells[i] = "█"
    return "[" + "".join(cells) + "]"


def _fmt_raw(v: Optional[int]) -> str:
    return f"{v:4d}" if v is not None else "  - "


def _calc_pct(raw: int, cal: JointCal) -> float:
    """根据 cal 当前标定算 pct（-1..+1，单向时 0..+1）。未标完返回 0。"""
    if cal.home_raw is None or cal.max_raw is None or cal.home_raw == cal.max_raw:
        return 0.0
    delta = raw - cal.home_raw
    if delta == 0:
        return 0.0
    pos_delta = cal.max_raw - cal.home_raw
    same_side_as_max = (delta > 0) == (pos_delta > 0)
    if same_side_as_max:
        return min(1.0, abs(delta) / abs(pos_delta))
    if cal.low_raw is None:
        return 0.0
    neg_delta = cal.low_raw - cal.home_raw
    if neg_delta == 0:
        return 0.0
    return -min(1.0, abs(delta) / abs(neg_delta))


def render(
    sel: int,
    cals: List[JointCal],
    raws: List[int],
    torque_on: bool,
    msg: str,
) -> None:
    sys.stdout.write("\x1b[2J\x1b[H")
    print("=" * 86)
    print(
        "  关节标定  |  扭矩:", "ON " if torque_on else "OFF",
        " |  h=home  m=max  l=low  s=保存  q=放弃",
    )
    print("=" * 86)
    print(
        f"  {'':2s}{'关节':<12s}{'当前':>6s}   "
        f"{'low(-100%)':>10s}  {'home':>5s}  {'max(+100%)':>10s}  "
        f"{'位置 (-100%←0%→+100%)':<36s}"
    )

    for i, (cal, raw) in enumerate(zip(cals, raws)):
        marker = "▶" if i == sel else " "
        pct = _calc_pct(raw, cal)
        bar = _bar(pct)
        pct_label = f"{pct * 100:+6.1f}%"
        kind = "↔" if cal.low_raw is not None else "→"   # 双向 / 单向
        done = "✓" if cal.is_complete() else " "
        print(
            f" {marker} J{i+1} {cal.name:<9s}{raw:>6d} {kind}  "
            f"{_fmt_raw(cal.low_raw)}     {_fmt_raw(cal.home_raw)}     {_fmt_raw(cal.max_raw)}    "
            f"{bar} {pct_label} {done}"
        )

    print("-" * 86)
    print("  按键：")
    print("    1..6 / ←→ 选关节       h 记 home   m 记 max(+100%)   l 记 low(-100%)")
    print("    L 清除 low (变单向)    t 切扭矩    r 刷新     空格 急停     s 保存   q/x 放弃")
    if msg:
        print(f"\n  {msg}")
    sys.stdout.flush()


# ---------------------------------------------------------------------------
# YAML 写入
# ---------------------------------------------------------------------------

def update_yaml(text: str, cals: List[JointCal]) -> str:
    """对每个关节的 yaml 段，按需替换/插入 home_raw / max_raw / low_raw。

    - low_raw is None 时：如果原 yaml 有该字段，把它移除；没有就什么也不做
    - low_raw 有值时：替换或插入
    """
    out = text
    for cal in cals:
        m = re.search(
            r"(^[ \t]*-[ \t]*name:[ \t]*" + re.escape(cal.name) + r"\b.*$)",
            out,
            re.MULTILINE,
        )
        if not m:
            raise RuntimeError(f"yaml 中找不到关节 {cal.name!r}")
        seg_start = m.start()
        rest = out[m.end():]
        m2 = re.search(r"^[ \t]*-[ \t]*name:|^\S", rest, re.MULTILINE)
        seg_end = m.end() + (m2.start() if m2 else len(rest))
        seg = out[seg_start:seg_end]

        seg = _set_field(seg, "home_raw", cal.home_raw)
        seg = _set_field(seg, "max_raw", cal.max_raw)
        if cal.low_raw is not None:
            seg = _set_field(seg, "low_raw", cal.low_raw)
        else:
            seg = _remove_field(seg, "low_raw")

        out = out[:seg_start] + seg + out[seg_end:]
    return out


def _set_field(seg: str, key: str, value) -> str:
    """已存在 ``key:`` 行 → 替换值；否则插入到 id 行后。
    传 None 跳过。"""
    if value is None:
        return seg
    pattern = re.compile(r"^([ \t]*)" + re.escape(key) + r":[ \t]*[^\n#]*", re.MULTILINE)
    if pattern.search(seg):
        return pattern.sub(lambda m: f"{m.group(1)}{key}: {value}", seg, count=1)
    id_pat = re.compile(r"^([ \t]*)id:[^\n]*\n", re.MULTILINE)
    m = id_pat.search(seg)
    if not m:
        return seg.rstrip() + f"\n    {key}: {value}\n"
    indent = m.group(1)
    insert_at = m.end()
    return seg[:insert_at] + f"{indent}{key}: {value}\n" + seg[insert_at:]


def _remove_field(seg: str, key: str) -> str:
    """删除整行 ``    key: value...\n``。找不到则原样返回。"""
    pattern = re.compile(r"^[ \t]*" + re.escape(key) + r":[^\n]*\n", re.MULTILINE)
    return pattern.sub("", seg, count=1)


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="交互式两点标定（home + max）")
    ap.add_argument("--port", default=None)
    ap.add_argument("--config", default="config/arm_config.yaml")
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

    cals = [JointCal.from_config(j) for j in cfg.joints]
    sel = 0
    torque_on = False
    msg = "已就绪：默认放扭矩，可手摆。home 已加载，把每颗摆到行程另一端按 m 记录"

    with Arm(cfg) as arm:
        try:
            arm.set_torque(False)
        except Exception as e:
            print(f"✗ 关闭扭矩失败：{e}", file=sys.stderr)
            return 1

        try:
            raws = arm.read_joints_raw()
        except Exception as e:
            print(f"✗ 启动时读位置失败：{e}", file=sys.stderr)
            return 1

        with cbreak_mode():
            render(sel, cals, raws, torque_on, msg)
            saved = False
            while True:
                key = read_key()
                if not key:
                    try:
                        raws = arm.read_joints_raw()
                    except Exception:
                        pass
                    render(sel, cals, raws, torque_on, msg)
                    continue

                if key in ("q", "Q", "x", "X", "\x1b"):
                    msg = "已放弃修改"
                    break

                if key in "123456":
                    idx = int(key) - 1
                    if idx < len(cals):
                        sel = idx
                        msg = f"已选 J{sel+1} ({cals[sel].name})"
                elif key == "←":
                    sel = (sel - 1) % len(cals)
                elif key == "→":
                    sel = (sel + 1) % len(cals)
                elif key == "h":
                    cals[sel].home_raw = raws[sel]
                    msg = f"J{sel+1} home_raw = {raws[sel]}（0%）"
                elif key == "m":
                    cals[sel].max_raw = raws[sel]
                    msg = f"J{sel+1} max_raw = {raws[sel]}（+100%）"
                elif key == "l":
                    cals[sel].low_raw = raws[sel]
                    # 校验：必须和 max 在 home 异侧
                    if not cals[sel].is_complete():
                        msg = (f"⚠ J{sel+1} low_raw = {raws[sel]} 不合法：必须与 max 在 home 的两侧。"
                               f"已记录但保存时会拒绝。")
                    else:
                        msg = f"J{sel+1} low_raw = {raws[sel]}（-100%，双向）"
                elif key == "L":
                    cals[sel].low_raw = None
                    msg = f"J{sel+1} 已清除 low（变回单向，pct ∈ [0, 1]）"
                elif key == "t":
                    torque_on = not torque_on
                    try:
                        arm.set_torque(torque_on)
                        msg = "扭矩 " + ("打开（小心）" if torque_on else "关闭（可手摆）")
                    except Exception as e:
                        msg = f"切扭矩失败：{e}"
                elif key == "r":
                    msg = "已刷新"
                elif key == " ":
                    try:
                        arm.emergency_stop()
                        torque_on = False
                        msg = "⚠ 紧急停止"
                    except Exception as e:
                        msg = f"急停失败：{e}"
                elif key in ("s", "S"):
                    msg = _save_to_yaml(cfg_path, cals, cfg)
                    saved = True
                    break

                try:
                    raws = arm.read_joints_raw()
                except Exception:
                    pass
                render(sel, cals, raws, torque_on, msg)

        try:
            arm.set_torque(False)
        except Exception:
            pass

    print()
    print(msg)
    return 0


def _save_to_yaml(cfg_path: Path, cals: List[JointCal], cfg: ArmConfig) -> str:
    incomplete = [c.name for c in cals if not c.is_complete()]
    if incomplete:
        return f"✗ 这些关节还没标完：{incomplete}（按 m 记录每个 max，或 q 放弃）"

    modified = [c for c, jc in zip(cals, cfg.joints) if c.is_modified(jc)]
    if not modified:
        return "✓ 没有改动，未写入"

    original = cfg_path.read_text(encoding="utf-8")
    backup = cfg_path.with_suffix(cfg_path.suffix + ".bak")
    backup.write_text(original, encoding="utf-8")

    try:
        new_text = update_yaml(original, cals)
    except RuntimeError as e:
        return f"✗ 写入失败：{e}"

    cfg_path.write_text(new_text, encoding="utf-8")
    names = ", ".join(c.name for c in modified)
    return f"✓ 已保存（{names}），原文件备份到 {backup.name}"


if __name__ == "__main__":
    raise SystemExit(main())
