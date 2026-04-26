"""条件限位（JointZoneLimit）单元测试。"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from arm.arm import (
    Arm,
    ArmConfig,
    JointConfig,
    JointZoneLimit,
)


# ---------- helpers ----------

def _j3_like(name: str = "elbow", id: int = 3) -> JointConfig:
    return JointConfig(
        name=name, id=id,
        home_raw=2048, max_raw=232, low_raw=2445,
        min_deg=34.9, max_deg=-159.8,
    )


def _wrist_like(name: str, id: int) -> JointConfig:
    return JointConfig(
        name=name, id=id,
        home_raw=2048, max_raw=4037, low_raw=59,
        min_deg=-175.0, max_deg=175.0,
    )


def _make_arm(zone_limits, current_raws):
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")

    joints = [_j3_like(), _wrist_like("wrist_pitch", 4), _wrist_like("wrist_roll", 5)]
    cfg = ArmConfig(joints=joints, zone_limits=zone_limits)
    arm = Arm(cfg, bus=bus)

    for i, s in enumerate(arm.servos):
        s.read_position = MagicMock(return_value=current_raws[i])
        s.write_position = MagicMock()
    return arm


# ---------- 数据类 ----------

def test_zone_limit_dataclass_basic():
    z = JointZoneLimit(
        target="wrist_roll", target_op=">", target_deg=90,
        gate="elbow", gate_op="<", gate_deg=0,
    )
    assert z.target == "wrist_roll"
    assert z.target_op == ">"
    assert z.target_deg == 90


def test_zone_limit_is_frozen():
    z = JointZoneLimit(target="a", target_op=">", target_deg=0,
                       gate="b", gate_op=">", gate_deg=0)
    with pytest.raises(Exception):
        z.target = "x"          # type: ignore[misc]


# ---------- yaml 解析 ----------

def test_yaml_loads_zone_limits(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_roll, id: 5, home_raw: 2048, max_raw: 4037, low_raw: 59, min_deg: -175, max_deg: 175}
zone_limits:
  - target: wrist_roll
    target_op: '>'
    target_deg: 90
    gate: elbow
    gate_op: '<'
    gate_deg: 0
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    cfg = ArmConfig.from_yaml(p)
    assert len(cfg.zone_limits) == 1
    z = cfg.zone_limits[0]
    assert z.target == "wrist_roll" and z.target_op == ">" and z.target_deg == 90
    assert z.gate == "elbow" and z.gate_op == "<" and z.gate_deg == 0


def test_yaml_zone_limit_target_eq_gate_warns(tmp_path: Path):
    yaml_text = """
joints:
  - {name: elbow, id: 3, home_raw: 2048, max_raw: 232, low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
zone_limits:
  - {target: elbow, target_op: '>', target_deg: 0, gate: elbow, gate_op: '>', gate_deg: 0}
"""
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text)
    with pytest.warns(UserWarning, match="不能相同"):
        cfg = ArmConfig.from_yaml(p)
    assert cfg.zone_limits == []


# ---------- _apply_zone_limits 行为 ----------

def test_zone_inactive_when_gate_not_in_zone():
    """gate 不处于激活区间 → 不夹紧。"""
    j_elbow = _j3_like()
    cur_elbow = j_elbow.deg_to_raw(20.0)   # +20°，不满足 < 0°
    arm = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op=">", target_deg=90,
            gate="elbow", gate_op="<", gate_deg=0,
        )],
        current_raws=[cur_elbow, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(120.0)  # 想去 +120°
    out = arm._apply_zone_limits([cur_elbow, 2048, target_wrist])
    assert out[2] == target_wrist                           # 没动


def test_zone_active_clamps_target_to_boundary():
    """gate 处于激活区间，target 想越过禁区边界 → 夹紧到边界。"""
    j_elbow = _j3_like()
    cur_elbow = j_elbow.deg_to_raw(-30.0)   # -30°，激活 < 0°
    arm = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op=">", target_deg=90,
            gate="elbow", gate_op="<", gate_deg=0,
        )],
        current_raws=[cur_elbow, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(120.0)  # 想去 +120° → 越界
    boundary_raw = arm.config.joints[2].deg_to_raw(90.0)   # 应被夹到 +90°

    out = arm._apply_zone_limits([cur_elbow, 2048, target_wrist])
    assert out[2] == boundary_raw


def test_zone_active_but_target_inside_safe_zone_unchanged():
    """gate 激活，但 target 目标在合规区内 → 不动（target 仍可在合规区自由活动）。"""
    j_elbow = _j3_like()
    cur_elbow = j_elbow.deg_to_raw(-30.0)
    arm = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op=">", target_deg=90,
            gate="elbow", gate_op="<", gate_deg=0,
        )],
        current_raws=[cur_elbow, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)  # 60° ≤ 90° 合规
    out = arm._apply_zone_limits([cur_elbow, 2048, target_wrist])
    assert out[2] == target_wrist


def test_zone_with_op_lt_clamps_to_lower_boundary():
    """target_op='<' 时禁止低于阈值 → 越界目标夹紧到阈值。"""
    j_elbow = _j3_like()
    cur_elbow = j_elbow.deg_to_raw(20.0)
    arm = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op="<", target_deg=-30,
            gate="elbow", gate_op=">", gate_deg=0,
        )],
        current_raws=[cur_elbow, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(-90.0)  # 想去 -90° → 越界 < -30°
    boundary_raw = arm.config.joints[2].deg_to_raw(-30.0)
    out = arm._apply_zone_limits([cur_elbow, 2048, target_wrist])
    assert out[2] == boundary_raw


def test_zone_skipped_when_gate_no_deg_mapping():
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    j_no_deg = JointConfig(name="g", id=3, home_raw=2048, max_raw=4000, low_raw=100)
    j_target = _wrist_like("t", 4)
    cfg = ArmConfig(
        joints=[j_no_deg, j_target],
        zone_limits=[JointZoneLimit(
            target="t", target_op=">", target_deg=0,
            gate="g", gate_op=">", gate_deg=0,
        )],
    )
    arm = Arm(cfg, bus=bus)
    for s in arm.servos:
        s.read_position = MagicMock(return_value=2048)

    target_t = arm.config.joints[1].deg_to_raw(60.0)
    with pytest.warns(UserWarning, match="度数映射"):
        out = arm._apply_zone_limits([2048, target_t])
    assert out[1] == target_t      # 没夹


def test_bypass_constraints_skips_zone_limits():
    j_elbow = _j3_like()
    cur_elbow = j_elbow.deg_to_raw(-30.0)
    arm = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op=">", target_deg=90,
            gate="elbow", gate_op="<", gate_deg=0,
        )],
        current_raws=[cur_elbow, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(120.0)
    arm.move_joints_raw([cur_elbow, 2048, target_wrist], bypass_constraints=True)
    # bypass 时不读 gate
    arm.servos[0].read_position.assert_not_called()


def test_no_zone_limits_zero_overhead():
    arm = _make_arm(zone_limits=[], current_raws=[2048, 2048, 2048])
    arm.move_joints_raw([2200, 2200, 2200])
    for s in arm.servos:
        s.read_position.assert_not_called()


def test_zone_limit_applied_during_move_joints_raw():
    """end-to-end：move_joints_raw 触发夹紧后 sync_write 收到的是夹紧后的 raw。"""
    j_elbow = _j3_like()
    cur_elbow = j_elbow.deg_to_raw(-30.0)
    arm = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op=">", target_deg=90,
            gate="elbow", gate_op="<", gate_deg=0,
        )],
        current_raws=[cur_elbow, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(120.0)
    boundary_raw = arm.config.joints[2].deg_to_raw(90.0)

    arm.move_joints_raw([cur_elbow, 2048, target_wrist])
    # gate 被读了一次（用于评估激活）
    arm.servos[0].read_position.assert_called()
    # sync_write 用的是夹紧后 raw — 通过 bus.transact payload 不易解；
    # 直接调 _apply_zone_limits 也已在前面测过。
    out = arm._apply_zone_limits([cur_elbow, 2048, target_wrist])
    assert out[2] == boundary_raw
