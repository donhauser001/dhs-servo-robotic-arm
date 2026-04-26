"""ArmConfig.poses + ArmState / enable / disable / goto_pose 单元测试。

不依赖硬件 —— 用 ``unittest.mock`` 替换 ``Bus``。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arm.arm import Arm, ArmConfig, ArmState, JointConfig


# ---------- helpers ----------

def _make_joints(n: int = 6) -> list[JointConfig]:
    """构造 n 个 home=2048, max=4000, low=100 的双向关节。"""
    return [
        JointConfig(name=f"j{i+1}", id=i+1, home_raw=2048, max_raw=4000, low_raw=100)
        for i in range(n)
    ]


def _make_arm(poses: dict[str, list[int]] | None = None, n_joints: int = 6) -> tuple[Arm, MagicMock]:
    """构造一个 6 关节的 Arm，bus / servo 全部 mock。

    返回 ``(arm, bus_mock)``，其中 ``bus_mock.transact.call_args_list`` 可以
    用来断言调用顺序。
    """
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    bus.read_u8 = MagicMock(return_value=0)          # MOVING=0 → 立即 idle
    bus.read_u16 = MagicMock(return_value=2048)      # 当前位置 = home

    cfg = ArmConfig(joints=_make_joints(n_joints), poses=poses or {})
    arm = Arm(cfg, bus=bus)

    for s in arm.servos:
        s.read_position = MagicMock(return_value=2048)
        s.write_position = MagicMock()
        s.torque = MagicMock()
    return arm, bus


# ---------- ArmConfig.poses 解析 ----------

def test_yaml_loads_poses(tmp_path: Path) -> None:
    yaml_text = """
joints:
  - {name: j1, id: 1, home_raw: 2048, max_raw: 4000}
  - {name: j2, id: 2, home_raw: 2048, max_raw: 4000}
poses:
  safe:  [4029, 2301]
  ready: [2051, 1193]
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert cfg.poses["safe"] == [4029, 2301]
    assert cfg.poses["ready"] == [2051, 1193]


def test_yaml_without_poses_section_ok(tmp_path: Path) -> None:
    yaml_text = """
joints:
  - {name: j1, id: 1, home_raw: 2048, max_raw: 4000}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert cfg.poses == {}


def test_yaml_pose_with_non_list_emits_warning(tmp_path: Path) -> None:
    yaml_text = """
joints:
  - {name: j1, id: 1, home_raw: 2048, max_raw: 4000}
poses:
  good: [100]
  bad:  not_a_list
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.warns(UserWarning, match="bad"):
        cfg = ArmConfig.from_yaml(p)
    assert "good" in cfg.poses
    assert "bad" not in cfg.poses


# ---------- ArmState 初始值 ----------

def test_default_state_is_disabled() -> None:
    arm, _ = _make_arm()
    assert arm.state == ArmState.DISABLED


# ---------- goto_pose 错误处理 ----------

def test_goto_pose_unknown_raises() -> None:
    arm, _ = _make_arm(poses={"safe": [2048] * 6})
    with pytest.raises(ValueError, match="unknown pose"):
        arm.goto_pose("nonexistent", wait=False)


def test_goto_pose_wrong_length_raises() -> None:
    arm, _ = _make_arm(poses={"weird": [1, 2, 3]})
    with pytest.raises(ValueError, match="期望 6"):
        arm.goto_pose("weird", wait=False)


def test_goto_pose_known_sends_sync_write() -> None:
    target = [2048, 2048, 2048, 2048, 2048, 2048]
    arm, bus = _make_arm(poses={"any": target})
    arm.goto_pose("any", wait=False)
    assert bus.transact.call_count >= 1   # 至少发了 1 次 sync_write


# ---------- enable / disable 流程 ----------

def test_enable_without_ready_raises() -> None:
    arm, _ = _make_arm(poses={"safe": [2048] * 6})
    with pytest.raises(RuntimeError, match="ready"):
        arm.enable()


def test_enable_changes_state_and_calls_torque() -> None:
    arm, bus = _make_arm(poses={
        "safe":  [2048] * 6,
        "ready": [3000] * 6,
    })
    assert arm.state == ArmState.DISABLED
    arm.enable()
    assert arm.state == ArmState.ENABLED
    # 至少：1 次"先把 GOAL 设为当前 raw"+ 1 次 set_torque + 1 次走 ready_pose
    assert bus.transact.call_count >= 3


def test_disable_changes_state_and_calls_torque() -> None:
    arm, bus = _make_arm(poses={
        "safe":  [2048] * 6,
        "ready": [3000] * 6,
    })
    arm.enable()
    bus.transact.reset_mock()
    arm.disable()
    assert arm.state == ArmState.DISABLED
    # disable 至少包含：走 safe_pose + 关扭矩
    assert bus.transact.call_count >= 2


def test_disable_to_none_skips_movement() -> None:
    """``disable(to=None)`` 不发运动指令，仅关扭矩。"""
    arm, bus = _make_arm(poses={
        "safe":  [2048] * 6,
        "ready": [3000] * 6,
    })
    arm.enable()
    bus.transact.reset_mock()
    arm.disable(to=None)
    assert arm.state == ArmState.DISABLED
    # 仅一次：关扭矩
    assert bus.transact.call_count == 1


def test_disable_to_unknown_raises() -> None:
    arm, _ = _make_arm(poses={"safe": [2048] * 6, "ready": [3000] * 6})
    arm.enable()
    with pytest.raises(ValueError, match="unknown pose"):
        arm.disable(to="nonexistent")


def test_disable_default_safe_missing_raises() -> None:
    """默认 ``to="safe"``；config 中无 safe 姿态时报错。"""
    arm, _ = _make_arm(poses={"ready": [3000] * 6})
    arm.enable()
    with pytest.raises(ValueError, match="safe"):
        arm.disable()


def test_enable_idempotent_state_stays_enabled() -> None:
    """enable 已经是 ENABLED 时再次调用，状态保持，仍然走到 ready_pose。"""
    arm, _ = _make_arm(poses={
        "safe":  [2048] * 6,
        "ready": [3000] * 6,
    })
    arm.enable()
    arm.enable()
    assert arm.state == ArmState.ENABLED


