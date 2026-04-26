"""关节前置联动（JointPrerequisite）单元测试。

不依赖硬件 —— 用 ``unittest.mock`` 替换 ``Bus``。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from arm.arm import (
    Arm,
    ArmConfig,
    JointConfig,
    JointPrerequisite,
)


# ---------- helpers ----------

def _j3_like(name: str = "elbow", id: int = 3) -> JointConfig:
    """带度数映射的 elbow 关节。"""
    return JointConfig(
        name=name, id=id,
        home_raw=2048, max_raw=232, low_raw=2445,
        min_deg=34.9, max_deg=-159.8,
    )


def _wrist_like(name: str, id: int) -> JointConfig:
    """带 ±175° 度数映射的对称关节。"""
    return JointConfig(
        name=name, id=id,
        home_raw=2048, max_raw=4037, low_raw=59,
        min_deg=-175.0, max_deg=175.0,
    )


def _make_arm(prerequisites, current_raws):
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    bus.read_u8 = MagicMock(return_value=0)   # MOVING=0：始终视为已到位

    joints = [_j3_like(), _wrist_like("wrist_pitch", 4), _wrist_like("wrist_roll", 5)]
    cfg = ArmConfig(joints=joints, prerequisites=prerequisites)
    arm = Arm(cfg, bus=bus)

    for i, s in enumerate(arm.servos):
        s.read_position = MagicMock(return_value=current_raws[i])
        s.write_position = MagicMock()
        s.torque = MagicMock()
    return arm, bus


# ---------- 数据类 ----------

def test_joint_prerequisite_dataclass_basic():
    p = JointPrerequisite(
        target="wrist_roll", target_op=">", target_deg=30,
        gate="elbow", gate_op=">", gate_deg=50,
    )
    assert p.target == "wrist_roll"
    assert p.target_op == ">"
    assert p.target_deg == 30
    assert p.gate == "elbow"
    assert p.gate_op == ">"
    assert p.gate_deg == 50
    assert p.buffer_deg == 2.0      # 默认值


def test_joint_prerequisite_is_frozen():
    p = JointPrerequisite(target="a", target_op=">", target_deg=0,
                          gate="b", gate_op=">", gate_deg=0)
    with pytest.raises(Exception):
        p.target = "x"          # type: ignore[misc]


# ---------- yaml 解析 ----------

def test_yaml_loads_prerequisites(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_roll, id: 5, home_raw: 2048, max_raw: 4037, low_raw: 59, min_deg: -175, max_deg: 175}
prerequisites:
  - target: wrist_roll
    target_op: '>'
    target_deg: 30
    gate: elbow
    gate_op: '>'
    gate_deg: 50
    buffer_deg: 3
  - target: wrist_roll
    target_op: '<'
    target_deg: -30
    gate: elbow
    gate_op: '<'
    gate_deg: -50
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert len(cfg.prerequisites) == 2
    assert cfg.prerequisites[0].buffer_deg == 3
    assert cfg.prerequisites[1].target_op == "<"
    assert cfg.prerequisites[1].gate_op == "<"
    assert cfg.prerequisites[1].buffer_deg == 2.0     # 默认


def test_yaml_prereq_unknown_target_warns(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
prerequisites:
  - {target: not_a_joint, target_op: '>', target_deg: 0, gate: elbow, gate_op: '>', gate_deg: 0}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.warns(UserWarning, match="not_a_joint"):
        cfg = ArmConfig.from_yaml(p)
    assert cfg.prerequisites == []


def test_yaml_prereq_target_eq_gate_warns(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
prerequisites:
  - {target: elbow, target_op: '>', target_deg: 0, gate: elbow, gate_op: '>', gate_deg: 0}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.warns(UserWarning, match="不能相同"):
        cfg = ArmConfig.from_yaml(p)
    assert cfg.prerequisites == []


# ---------- _apply_prerequisites 行为 ----------

def test_prereq_not_triggered_does_nothing():
    """target 目标不在触发区间 → 不读 gate、不动作。"""
    arm, bus = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=50,
        )],
        current_raws=[2048, 2048, 2048],     # 全在 home
    )
    # wrist_roll 想去 +10° → 不触发（不 > 30°）
    target_raw_wrist = arm.config.joints[2].deg_to_raw(10.0)
    out = arm._apply_prerequisites([2048, 2048, target_raw_wrist], None, None)
    assert out == [2048, 2048, target_raw_wrist]
    arm.servos[0].read_position.assert_not_called()    # 没读 gate
    arm.servos[0].write_position.assert_not_called()   # 没自动转


def test_prereq_triggered_and_gate_satisfied_no_move():
    """target 触发，但 gate 当前已满足 + 用户 gate 目标也满足 → 不自动转。"""
    j_elbow = _j3_like()
    cur_elbow_raw = j_elbow.deg_to_raw(20.0)   # +20° → 满足 > 0°
    arm, bus = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)   # 触发
    target_elbow = arm.config.joints[0].deg_to_raw(15.0)   # 用户也想让 elbow 在安全区
    out = arm._apply_prerequisites([target_elbow, 2048, target_wrist], None, None)
    assert out == [target_elbow, 2048, target_wrist]   # 啥都没改
    arm.servos[0].write_position.assert_not_called()


def test_prereq_triggered_and_gate_unsatisfied_auto_moves():
    """target 触发 + gate 当前不满足 → 自动通过 move_joints_raw 把 gate 转到 safe，
    并把用户原 gate 目标也改写为 safe_raw。"""
    j_elbow = _j3_like()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)   # -30° → 不满足 > 0°
    arm, bus = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0, buffer_deg=2.0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)    # 触发
    target_elbow_user = arm.config.joints[0].deg_to_raw(-10.0)  # 用户也想让 elbow 在违反区
    safe_raw = arm.config.joints[0].deg_to_raw(2.0)         # 0 + buffer

    out = arm._apply_prerequisites(
        [target_elbow_user, 2048, target_wrist], None, None,
    )

    # 干预事件被记录
    kinds = [iv["kind"] for iv in arm.recent_interventions]
    assert kinds.count("prerequisite") >= 1
    # bus.transact 被调（move_joints_raw 内部 sync_write）
    assert bus.transact.call_count >= 1
    # 用户原 gate 目标也被改写到 safe_raw
    assert out[0] == safe_raw
    assert out[2] == target_wrist


def test_prereq_with_op_lt():
    """target_op='<' & gate_op='<' 同样工作。"""
    j_elbow = _j3_like()
    cur_elbow_raw = j_elbow.deg_to_raw(0.0)   # 0° → 不满足 < -50°
    arm, bus = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op="<", target_deg=-30,
            gate="elbow", gate_op="<", gate_deg=-50, buffer_deg=2.0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(-60.0)   # < -30° 触发
    safe_raw = arm.config.joints[0].deg_to_raw(-52.0)       # -50 - buffer

    out = arm._apply_prerequisites([2048, 2048, target_wrist], None, None)
    # 自动驱动 gate 走的是 sync_write（不是 single write_position）
    assert bus.transact.call_count >= 1
    # 用户原 gate 目标 = home (0°)，违反 < -50° → 改写为 safe_raw
    assert out[0] == safe_raw
    # 干预事件记录里能看到 prerequisite
    assert any(iv["kind"] == "prerequisite" for iv in arm.recent_interventions)


def test_prereq_skipped_when_target_no_deg_mapping():
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    j_no_deg = JointConfig(name="t", id=4, home_raw=2048, max_raw=4000, low_raw=100)
    j_gate = _j3_like(name="g", id=3)
    cfg = ArmConfig(
        joints=[j_no_deg, j_gate],
        prerequisites=[JointPrerequisite(
            target="t", target_op=">", target_deg=0,
            gate="g", gate_op=">", gate_deg=0,
        )],
    )
    arm = Arm(cfg, bus=bus)
    for s in arm.servos:
        s.read_position = MagicMock(return_value=2048)
        s.write_position = MagicMock()

    with pytest.warns(UserWarning, match="target 没有度数映射"):
        out = arm._apply_prerequisites([2500, 2048], None, None)
    assert out == [2500, 2048]
    arm.servos[1].write_position.assert_not_called()


def test_bypass_constraints_skips_prerequisites():
    """``bypass_constraints=True`` 同时跳过 prerequisites（不读不写 gate）。"""
    j_elbow = _j3_like()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)
    arm, bus = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)
    arm.move_joints_raw([cur_elbow_raw, 2048, target_wrist], bypass_constraints=True)
    arm.servos[0].write_position.assert_not_called()


def test_no_prerequisites_zero_overhead():
    """空 prerequisites → 不读不写 gate。"""
    arm, bus = _make_arm(prerequisites=[], current_raws=[2048, 2048, 2048])
    arm.move_joints_raw([2200, 2200, 2200])
    for s in arm.servos:
        s.write_position.assert_not_called()
        s.read_position.assert_not_called()
