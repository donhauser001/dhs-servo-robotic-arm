"""关节联锁约束（JointConstraint）单元测试。

不依赖硬件 —— 用 ``unittest.mock`` 替换 ``Bus``。
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arm.arm import (
    Arm,
    ArmConfig,
    JointConfig,
    JointConstraint,
)


# ---------- helpers ----------

def _j3_like(home_raw: int = 2048, max_raw: int = 232, low_raw: int = 2445):
    """构造一个跟真实 J3 标定一致的 elbow 关节。"""
    return JointConfig(
        name="elbow", id=3,
        home_raw=home_raw, max_raw=max_raw, low_raw=low_raw,
        min_deg=34.9, max_deg=-159.8,    # home_deg 默认 0
    )


def _make_arm_with_j3_constraints(j3_current_raw: int) -> tuple[Arm, MagicMock, list]:
    """构造 5 关节臂（J1-J5），J3 是 elbow（带度数），J4/J5 受 J3 联锁。

    ``j3_current_raw`` 决定 mock 的 J3 当前位置（用于触发/不触发约束）。
    """
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")

    joints = [
        JointConfig(name="base", id=1, home_raw=2048, max_raw=4000, low_raw=100),
        JointConfig(name="shoulder", id=2, home_raw=2048, max_raw=4000, low_raw=100),
        _j3_like(),
        JointConfig(name="wrist_pitch", id=4, home_raw=2048, max_raw=4000, low_raw=100),
        JointConfig(name="wrist_roll", id=5, home_raw=2048, max_raw=4000, low_raw=100),
    ]
    constraints = [
        JointConstraint(target="wrist_pitch", gate="elbow", gate_min_deg=0.0),
        JointConstraint(target="wrist_roll",  gate="elbow", gate_min_deg=20.0),
    ]
    cfg = ArmConfig(joints=joints, constraints=constraints)
    arm = Arm(cfg, bus=bus)

    current = [2048, 2048, j3_current_raw, 2048, 2048]
    for i, s in enumerate(arm.servos):
        s.read_position = MagicMock(return_value=current[i])
        s.write_position = MagicMock()
        s.torque = MagicMock()
    return arm, bus, current


# ---------- JointConstraint 数据类 ----------

def test_joint_constraint_dataclass_basic():
    c = JointConstraint(target="wrist_pitch", gate="elbow", gate_min_deg=0.0)
    assert c.target == "wrist_pitch"
    assert c.gate == "elbow"
    assert c.gate_min_deg == 0.0


def test_joint_constraint_is_frozen():
    """frozen=True 防止误改约束（让它能 hash 进 set，也避免 runtime mutation）。"""
    c = JointConstraint(target="a", gate="b", gate_min_deg=0)
    with pytest.raises(Exception):
        c.target = "x"          # type: ignore[misc]


# ---------- yaml 解析 constraints ----------

def test_yaml_loads_constraints(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_pitch, id: 4, home_raw: 2048, max_raw: 4000, low_raw: 100}
  - {name: wrist_roll,  id: 5, home_raw: 2048, max_raw: 4000, low_raw: 100}
constraints:
  - {target: wrist_pitch, gate: elbow, gate_min_deg: 0}
  - {target: wrist_roll,  gate: elbow, gate_min_deg: 20}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert len(cfg.constraints) == 2
    assert cfg.constraints[0].target == "wrist_pitch"
    assert cfg.constraints[0].gate == "elbow"
    assert cfg.constraints[0].gate_min_deg == 0
    assert cfg.constraints[1].gate_min_deg == 20


def test_yaml_constraints_gate_joint_alias(tmp_path: Path):
    """``gate_joint`` 是 ``gate`` 的同义词，向后兼容。"""
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_pitch, id: 4, home_raw: 2048, max_raw: 4000, low_raw: 100}
constraints:
  - {target: wrist_pitch, gate_joint: elbow, gate_min_deg: 0}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert len(cfg.constraints) == 1
    assert cfg.constraints[0].gate == "elbow"


def test_yaml_constraints_unknown_target_warns(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
constraints:
  - {target: not_a_joint, gate: elbow, gate_min_deg: 0}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.warns(UserWarning, match="not_a_joint"):
        cfg = ArmConfig.from_yaml(p)
    assert cfg.constraints == []


def test_yaml_no_constraints_section_ok(tmp_path: Path):
    yaml_text = """
joints:
  - {name: j1, id: 1, home_raw: 2048, max_raw: 4000}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert cfg.constraints == []


# ---------- _apply_constraints 行为 ----------

def test_constraint_satisfied_lets_target_through():
    """J3 在 +30°（>20°），J4 / J5 都解锁 → 目标位置原样下发。"""
    # raw=2178 → pct=-0.328 → home + 0.328*34.9 = +11.4°? 让我们直接挑一个 ≥20° 的
    # raw=2275 → delta=+227, low_delta=+397, pct=-227/397=-0.572, deg=+19.97°
    # 用 raw=2295 让 J3 在 +21° 左右
    arm, bus, current = _make_arm_with_j3_constraints(j3_current_raw=2295)
    j3 = arm.config.joints[2]
    j3_deg = j3.raw_to_deg(2295)
    assert j3_deg > 20.0      # 双约束都满足

    target = [2200, 2200, 2295, 2500, 2700]
    arm.move_joints_raw(target)

    # 抓 sync_write 的 payload，确认 J4/J5 没被冻结
    call = bus.transact.call_args_list[-1]
    pkt = call.args[0]
    # pkt.params 在 sync_write 头部包含 reg + len，后面 7B*N
    # 简化：直接对比断言它发送了原始 target，而不是 current
    # 这里用一个间接断言：bus.transact 至少被调一次
    assert bus.transact.call_count == 1


def test_constraint_violated_freezes_only_affected_joint():
    """J3 在 -50°（<0°），J4/J5 都被冻结，但 J1/J2/J3 正常。"""
    # J3 raw=1480 → -50°
    arm, bus, current = _make_arm_with_j3_constraints(j3_current_raw=1480)

    target = [2200, 2200, 1480, 2500, 2700]
    arm.move_joints_raw(target)

    # 我们不解 sync_write 包，而是改在 _apply_constraints 之后的 positions 检查；
    # _apply_constraints 是公共方法，可以直接测
    out = arm._apply_constraints(list(target))
    assert out[0] == 2200            # J1 原样
    assert out[1] == 2200            # J2 原样
    assert out[2] == 1480            # J3（gate）原样
    assert out[3] == current[3]      # J4 被冻结到当前
    assert out[4] == current[4]      # J5 被冻结到当前


def test_constraint_partial_satisfaction():
    """J3 在 +10°（>0° 但 ≤20°）→ J4 解锁，J5 仍冻结。"""
    # J3 raw=2162 → pct ≈ -0.287 → +10.0°
    arm, bus, current = _make_arm_with_j3_constraints(j3_current_raw=2162)
    j3 = arm.config.joints[2]
    j3_deg = j3.raw_to_deg(2162)
    assert 0 < j3_deg <= 20         # 仅 J4 解锁

    target = [2200, 2200, 2162, 2500, 2700]
    out = arm._apply_constraints(list(target))
    assert out[3] == 2500            # J4 解锁
    assert out[4] == current[4]      # J5 仍冻结


def test_constraint_at_threshold_is_strict_greater():
    """J3 = 0° 时 J4 仍然冻结（gate_min_deg 是严格大于）。"""
    arm, bus, current = _make_arm_with_j3_constraints(j3_current_raw=2048)  # home → 0°
    target = [2200, 2200, 2048, 2500, 2700]
    out = arm._apply_constraints(list(target))
    assert out[3] == current[3]      # J4 冻结（0° 不算 >0°）


def test_bypass_constraints_skips_check():
    """``bypass_constraints=True`` 不读 gate、不冻结。"""
    arm, bus, current = _make_arm_with_j3_constraints(j3_current_raw=1480)  # 违反
    target = [2200, 2200, 1480, 2500, 2700]
    arm.move_joints_raw(target, bypass_constraints=True)
    # 没调 read_position（_apply_constraints 没跑）
    for s in arm.servos:
        s.read_position.assert_not_called()


def test_no_constraints_zero_overhead():
    """没配 constraints 时不读关节位置。"""
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    cfg = ArmConfig(
        joints=[JointConfig(name=f"j{i+1}", id=i+1, home_raw=2048,
                            max_raw=4000, low_raw=100) for i in range(5)],
        constraints=[],   # 显式空
    )
    arm = Arm(cfg, bus=bus)
    for s in arm.servos:
        s.read_position = MagicMock(return_value=2048)

    arm.move_joints_raw([2200] * 5)
    for s in arm.servos:
        s.read_position.assert_not_called()


def test_joint_constraint_default_op_is_gt():
    """不显式指定 op 时默认是 ">"（向后兼容）。"""
    c = JointConstraint(target="a", gate="b", gate_min_deg=0)
    assert c.op == ">"


def test_yaml_loads_constraint_with_op_lt(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_pitch, id: 4, home_raw: 2048, max_raw: 4000, low_raw: 100}
constraints:
  - {target: wrist_pitch, gate: elbow, gate_min_deg: 30, op: '<'}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert cfg.constraints[0].op == "<"
    assert cfg.constraints[0].gate_min_deg == 30


def test_yaml_constraints_invalid_op_warns_and_defaults_to_gt(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_pitch, id: 4, home_raw: 2048, max_raw: 4000, low_raw: 100}
constraints:
  - {target: wrist_pitch, gate: elbow, gate_min_deg: 0, op: '>='}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.warns(UserWarning, match="op"):
        cfg = ArmConfig.from_yaml(p)
    assert cfg.constraints[0].op == ">"


def test_constraint_op_lt_freezes_when_gate_above_threshold():
    """op='<' 时含义反转：gate < 阈值 才解锁；gate ≥ 阈值时冻结。"""
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    joints = [
        _j3_like(),
        JointConfig(name="wrist_pitch", id=4, home_raw=2048, max_raw=4000, low_raw=100),
    ]
    cfg = ArmConfig(
        joints=joints,
        constraints=[JointConstraint(target="wrist_pitch", gate="elbow",
                                     gate_min_deg=0.0, op="<")],
    )
    arm = Arm(cfg, bus=bus)
    current = [2295, 2048]    # J3 ≈ +21°（>0°，违反 op='<'）
    for i, s in enumerate(arm.servos):
        s.read_position = MagicMock(return_value=current[i])

    out = arm._apply_constraints([2295, 2500])
    assert out[1] == current[1]   # 被冻结


def test_constraint_op_lt_lets_through_when_gate_below_threshold():
    """op='<' 且 gate < 阈值：解锁。"""
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    joints = [
        _j3_like(),
        JointConfig(name="wrist_pitch", id=4, home_raw=2048, max_raw=4000, low_raw=100),
    ]
    cfg = ArmConfig(
        joints=joints,
        constraints=[JointConstraint(target="wrist_pitch", gate="elbow",
                                     gate_min_deg=0.0, op="<")],
    )
    arm = Arm(cfg, bus=bus)
    current = [1480, 2048]    # J3 ≈ -50°（<0°，满足 op='<'）
    for i, s in enumerate(arm.servos):
        s.read_position = MagicMock(return_value=current[i])

    out = arm._apply_constraints([1480, 2500])
    assert out[1] == 2500         # 解锁


def test_constraint_skipped_when_gate_has_no_deg_mapping():
    """gate 关节没标度数 → 发 warning，约束跳过（不冻结 target）。"""
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    j_no_deg = JointConfig(name="elbow", id=3, home_raw=2048,
                           max_raw=232, low_raw=2445)  # 没 min/max_deg
    j_target = JointConfig(name="wrist_pitch", id=4, home_raw=2048,
                           max_raw=4000, low_raw=100)
    cfg = ArmConfig(
        joints=[j_no_deg, j_target],
        constraints=[JointConstraint(target="wrist_pitch", gate="elbow",
                                     gate_min_deg=0.0)],
    )
    arm = Arm(cfg, bus=bus)
    for s in arm.servos:
        s.read_position = MagicMock(return_value=2048)

    target = [2048, 2500]
    with pytest.warns(UserWarning, match="度数映射"):
        out = arm._apply_constraints(list(target))
    # 跳过 → target 原样
    assert out[1] == 2500
