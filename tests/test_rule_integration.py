"""三种联动规则的集成测试：

* prereq 自动驱动 gate → 联锁 / 限位由"违反"翻转为"满足"
* enabled=False 时跳过该规则
* 多规则方向矛盾在 from_yaml 时给出 warning
* 干预事件被 ring buffer 捕获
* emergency_stop 绕过所有规则
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
    JointPrerequisite,
    JointZoneLimit,
)


# ---------- helpers ----------

def _j_elbow() -> JointConfig:
    return JointConfig(
        name="elbow", id=3,
        home_raw=2048, max_raw=232, low_raw=2445,
        min_deg=34.9, max_deg=-159.8,
    )


def _j_wrist(name: str, id: int) -> JointConfig:
    return JointConfig(
        name=name, id=id,
        home_raw=2048, max_raw=4037, low_raw=59,
        min_deg=-175.0, max_deg=175.0,
    )


def _make_arm(*, constraints=(), prerequisites=(), zone_limits=(), current_raws=None):
    bus = MagicMock()
    bus.transact = MagicMock(return_value=b"")
    bus.read_u8 = MagicMock(return_value=0)   # MOVING=0：wait_until_idle 立刻 True

    joints = [_j_elbow(), _j_wrist("wrist_pitch", 4), _j_wrist("wrist_roll", 5)]
    cfg = ArmConfig(
        joints=joints,
        constraints=list(constraints),
        prerequisites=list(prerequisites),
        zone_limits=list(zone_limits),
    )
    arm = Arm(cfg, bus=bus)

    raws = list(current_raws if current_raws is not None else [2048, 2048, 2048])
    for i, s in enumerate(arm.servos):
        s.read_position = MagicMock(return_value=raws[i])
        s.write_position = MagicMock()
        s.torque = MagicMock()

    # 让 read_joints_raw 也跟着 raws 一致
    arm.read_joints_raw = MagicMock(return_value=raws)
    return arm, bus, raws


# ---------- prereq 解锁 constraint ----------

def test_prereq_drives_gate_then_constraint_no_longer_blocks():
    """场景：constraint 要 elbow > 0 才能动 wrist_roll；prereq 同样要 elbow > 0。
    用户初始 elbow=-30°，但调用 prereq 后 elbow 已被改写为 +2°（满足）。
    """
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)   # 不满足
    arm, bus, _ = _make_arm(
        constraints=[JointConstraint(
            target="wrist_roll", gate="elbow",
            gate_min_deg=0, op=">",
        )],
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0, buffer_deg=2.0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)
    safe_elbow = arm.config.joints[0].deg_to_raw(2.0)

    # 调 move_joints_raw → 触发 prereq → gate 被改 → constraint 不锁
    arm.move_joints_raw([2048, 2048, target_wrist])

    # bus.transact 至少被调 2 次：prereq 内部驱动 + 真实 sync_write
    assert bus.transact.call_count >= 2

    # 检查最后一次 sync_write 的 elbow 目标 = safe_elbow
    last_call_args = bus.transact.call_args_list[-1]
    # 不强解析 packet 内部，仅校验干预事件
    kinds = [iv["kind"] for iv in arm.recent_interventions]
    assert "prerequisite" in kinds


def test_disabled_prereq_skipped():
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)
    arm, bus, _ = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0,
            enabled=False,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)
    out = arm._apply_prerequisites([2048, 2048, target_wrist], None, None)
    # 没有触发，positions 原样返回
    assert out == [2048, 2048, target_wrist]
    assert not arm.recent_interventions


def test_disabled_constraint_skipped():
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)   # 不满足
    arm, bus, _ = _make_arm(
        constraints=[JointConstraint(
            target="wrist_roll", gate="elbow",
            gate_min_deg=0, op=">",
            enabled=False,
        )],
        current_raws=[cur_elbow_raw, 2048, 2050],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)
    out = arm._apply_constraints([cur_elbow_raw, 2048, target_wrist])
    # 禁用了就不会被锁
    assert out[2] == target_wrist


def test_disabled_zone_limit_skipped():
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)
    arm, bus, _ = _make_arm(
        zone_limits=[JointZoneLimit(
            target="wrist_roll", target_op=">", target_deg=90,
            gate="elbow", gate_op="<", gate_deg=0,
            enabled=False,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(120.0)   # > 90，本应被夹紧
    out = arm._apply_zone_limits([cur_elbow_raw, 2048, target_wrist])
    assert out[2] == target_wrist   # 没被夹


# ---------- emergency_stop 绕过规则 ----------

def test_emergency_stop_bypasses_all_rules():
    """紧急停止绕过 prerequisites（避免阻塞）和所有其它规则。"""
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)
    arm, bus, raws = _make_arm(
        constraints=[JointConstraint(
            target="wrist_roll", gate="elbow",
            gate_min_deg=0, op=">",
        )],
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )

    # set_torque 也用 bus.transact，不该 raise
    arm.emergency_stop()
    # 没有任何规则触发干预（紧急停止本身不计为 intervention）
    assert len(arm.recent_interventions) == 0


# ---------- wait_until_idle 超时 raise ----------

def test_prereq_wait_timeout_raises_runtime_error(monkeypatch):
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)
    arm, bus, _ = _make_arm(
        prerequisites=[JointPrerequisite(
            target="wrist_roll", target_op=">", target_deg=30,
            gate="elbow", gate_op=">", gate_deg=0,
        )],
        current_raws=[cur_elbow_raw, 2048, 2048],
    )
    # 让 wait_until_idle 一直返回 False（模拟到位失败）
    arm.wait_until_idle = MagicMock(return_value=False)

    target_wrist = arm.config.joints[2].deg_to_raw(60.0)
    with pytest.raises(RuntimeError, match="未到位"):
        arm._apply_prerequisites([2048, 2048, target_wrist], None, None)


# ---------- 干预事件 ring buffer 容量 ----------

def test_intervention_ring_buffer_caps_at_20():
    j_elbow = _j_elbow()
    cur_elbow_raw = j_elbow.deg_to_raw(-30.0)
    arm, bus, _ = _make_arm(
        constraints=[JointConstraint(
            target="wrist_roll", gate="elbow",
            gate_min_deg=0, op=">",
        )],
        current_raws=[cur_elbow_raw, 2048, 2050],
    )
    target_wrist = arm.config.joints[2].deg_to_raw(60.0)
    for _ in range(30):
        arm._apply_constraints([cur_elbow_raw, 2048, target_wrist])
    assert len(arm.recent_interventions) == 20
    assert all(iv["kind"] == "constraint" for iv in arm.recent_interventions)


# ---------- 跨规则一致性体检 ----------

def test_validate_warns_on_constraint_prereq_direction_conflict(tmp_path: Path):
    """同一对 (target, gate) 上 constraint 与 prereq 条件不一致时给 warning。"""
    yaml_text = """
joints:
  - {name: elbow,       id: 3, home_raw: 2048, max_raw: 232,  low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_roll,  id: 5, home_raw: 2048, max_raw: 4037, low_raw: 59,  min_deg: -175.0, max_deg: 175.0}

constraints:
  - {target: wrist_roll, gate: elbow, gate_min_deg: 0,   op: ">"}

prerequisites:
  - {target: wrist_roll, target_op: ">", target_deg: 30,
     gate:   elbow,      gate_op:   ">", gate_deg:   50,
     buffer_deg: 2}
""".strip()
    p = tmp_path / "c.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    with pytest.warns(UserWarning, match="规则冲突"):
        ArmConfig.from_yaml(p)


def test_validate_warns_on_zone_limit_empty_zone(tmp_path: Path):
    """同 target 上 禁>10 与 禁<20 同时激活时合规区间为空。"""
    yaml_text = """
joints:
  - {name: elbow,       id: 3, home_raw: 2048, max_raw: 232,  low_raw: 2445, min_deg: 34.9, max_deg: -159.8}
  - {name: wrist_roll,  id: 5, home_raw: 2048, max_raw: 4037, low_raw: 59,  min_deg: -175.0, max_deg: 175.0}

zone_limits:
  - {target: wrist_roll, target_op: ">", target_deg: 10, gate: elbow, gate_op: ">", gate_deg: 0}
  - {target: wrist_roll, target_op: "<", target_deg: 20, gate: elbow, gate_op: "<", gate_deg: 0}
""".strip()
    p = tmp_path / "z.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    with pytest.warns(UserWarning, match="zone_limit 矛盾"):
        ArmConfig.from_yaml(p)


def test_validate_warns_on_prereq_dependency_cycle(tmp_path: Path):
    """A 的前置是 B，B 的前置是 A → 死锁，给 warning。"""
    yaml_text = """
joints:
  - {name: a, id: 1, home_raw: 2048, max_raw: 4037, low_raw: 59, min_deg: -175.0, max_deg: 175.0}
  - {name: b, id: 2, home_raw: 2048, max_raw: 4037, low_raw: 59, min_deg: -175.0, max_deg: 175.0}

prerequisites:
  - {target: a, target_op: ">", target_deg: 0, gate: b, gate_op: ">", gate_deg: 0}
  - {target: b, target_op: ">", target_deg: 0, gate: a, gate_op: ">", gate_deg: 0}
""".strip()
    p = tmp_path / "cycle.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    with pytest.warns(UserWarning, match="依赖环"):
        ArmConfig.from_yaml(p)
