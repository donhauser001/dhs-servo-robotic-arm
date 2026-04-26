"""SMS_STS 协议层单元测试（不依赖硬件）。

跑法：
    pytest -q tests/
"""

from __future__ import annotations

import pytest

from arm import protocol as p
from arm.protocol import (
    BROADCAST_ID,
    HEADER,
    Instruction,
    Packet,
    Reg,
    StatusError,
    checksum,
    from_le16,
    parse_response,
    to_le16,
)


# ---------- 校验和 ----------

def test_checksum_basic():
    # 飞特官方 PING 例子: ID=1, LEN=2, INST=0x01 → CHK = ~(1+2+1) = ~4 = 0xFB
    assert checksum([1, 2, 1]) == 0xFB

def test_checksum_overflow():
    # 求和会超过 0xFF 时应取低 8 位再取反
    assert checksum([0xFF, 0xFF, 0xFF]) == 0x02


# ---------- 16-bit 编解码 ----------

@pytest.mark.parametrize("value, expected", [
    (0,        b"\x00\x00"),
    (4095,     b"\xff\x0f"),
    (300,      b"\x2c\x01"),
    (-1,       b"\x01\x80"),
    (-1000,    b"\xe8\x83"),
])
def test_to_le16(value, expected):
    assert to_le16(value) == expected

def test_le16_roundtrip_unsigned():
    for v in (0, 1, 100, 4095, 0x7FFF):
        b = to_le16(v)
        assert from_le16(b[0], b[1]) == v

def test_le16_signed_bit15():
    # 速度 / 负载是 bit15=方向位的特殊有符号
    assert from_le16(0xE8, 0x83, signed_bit15=True) == -1000
    assert from_le16(0x64, 0x00, signed_bit15=True) == 100


# ---------- 数据包构造与解码 ----------

def test_packet_ping_encode():
    pkt = Packet.ping(1)
    raw = pkt.encode()
    assert raw[:2] == HEADER
    assert raw[2] == 1               # ID
    assert raw[3] == 2                # LEN = 1 (INST) + 1 (CHK) = 2
    assert raw[4] == Instruction.PING
    # 校验
    assert raw[5] == checksum(raw[2:5])

def test_packet_read_position():
    pkt = Packet.read(3, Reg.PRESENT_POSITION, 2)
    raw = pkt.encode()
    assert raw[2] == 3
    assert raw[3] == 4                # LEN = 4
    assert raw[4] == Instruction.READ
    assert raw[5] == Reg.PRESENT_POSITION
    assert raw[6] == 2

def test_packet_write_position():
    pkt = Packet.write(2, Reg.GOAL_POSITION, to_le16(2048))
    raw = pkt.encode()
    assert raw[2] == 2
    assert raw[4] == Instruction.WRITE
    assert raw[5] == Reg.GOAL_POSITION
    assert raw[6:8] == b"\x00\x08"     # 2048 = 0x0800 LE

def test_packet_sync_write():
    items = [
        (1, bytes([50]) + to_le16(2000) + to_le16(0) + to_le16(1500)),
        (2, bytes([50]) + to_le16(3000) + to_le16(0) + to_le16(1500)),
    ]
    pkt = Packet.sync_write(Reg.ACC, 7, items)
    raw = pkt.encode()
    assert raw[2] == BROADCAST_ID
    assert raw[4] == Instruction.SYNC_WRITE
    # 参数布局：addr(1) data_len(1) [id, payload×7] × N
    assert raw[5] == Reg.ACC
    assert raw[6] == 7
    assert raw[7] == 1                 # 第 1 颗 ID
    assert raw[15] == 2                # 第 2 颗 ID（7+1=8 字节后）

def test_parse_response_ok():
    # 模拟 ID=5 的舵机回包：ERR=0, 参数 = 位置 2047 (LE)
    body = bytes([5, 4, 0]) + to_le16(2047)
    raw = HEADER + body + bytes([checksum(body)])
    pkt = parse_response(raw)
    assert pkt.sid == 5
    assert pkt.inst_or_err == 0
    assert from_le16(pkt.params[0], pkt.params[1]) == 2047

def test_parse_response_bad_checksum():
    body = bytes([5, 4, 0]) + to_le16(2047)
    raw = HEADER + body + bytes([0x00])  # 错误校验和
    with pytest.raises(p.ChecksumError):
        parse_response(raw)

def test_parse_response_truncated():
    raw = HEADER + bytes([5, 4, 0])  # 缺参数和校验
    with pytest.raises(p.ProtocolError):
        parse_response(raw)


# ---------- 关节百分比 / 角度 ↔ raw 映射 ----------

def test_pct_to_raw_basic():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500)
    assert j.pct_to_raw(0.0) == 1000
    assert j.pct_to_raw(1.0) == 3500
    # 0.5 应在中间
    assert j.pct_to_raw(0.5) == round(1000 + 0.5 * (3500 - 1000))


def test_pct_to_raw_reversed_direction():
    """home > max 时（即 max 在 home 的"减小"方向），公式应自然处理。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=4079, max_raw=2700)
    assert j.pct_to_raw(0.0) == 4079
    assert j.pct_to_raw(1.0) == 2700
    # 0% 端 raw 较大，100% 端较小，pct 增大对应 raw 减小
    assert j.pct_to_raw(0.5) < j.pct_to_raw(0.0)
    assert j.pct_to_raw(1.0) < j.pct_to_raw(0.5)


def test_raw_to_pct_basic():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500)
    assert j.raw_to_pct(1000) == 0.0
    assert j.raw_to_pct(3500) == 1.0
    assert abs(j.raw_to_pct(2250) - 0.5) < 1e-9


def test_raw_to_pct_reversed_direction():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=4079, max_raw=2700)
    assert j.raw_to_pct(4079) == 0.0
    assert j.raw_to_pct(2700) == 1.0


def test_pct_clamps_out_of_range():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500)
    assert j.pct_to_raw(-0.5) == j.pct_to_raw(0.0)
    assert j.pct_to_raw(1.5) == j.pct_to_raw(1.0)
    # 超出标定范围的 raw 也夹紧
    assert j.raw_to_pct(500) == 0.0
    assert j.raw_to_pct(99999) == 1.0
    # 反方向标定时也得夹紧
    j2 = JointConfig(name="t", id=1, home_raw=4079, max_raw=2700)
    assert j2.raw_to_pct(4090) == 0.0   # 超 home 端
    assert j2.raw_to_pct(100) == 1.0    # 超 max 端


def test_pct_roundtrip():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1500, max_raw=3800)
    for raw in (1500, 2000, 2500, 3000, 3800):
        pct = j.raw_to_pct(raw)
        assert j.pct_to_raw(pct) == raw


def test_home_pct_is_zero():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=4079, max_raw=2700)
    assert j.home_pct == 0.0


def test_deg_mapping_optional():
    from arm.arm import JointConfig
    j_no_deg = JointConfig(name="t", id=1, home_raw=2047, max_raw=4095)
    assert j_no_deg.has_deg_mapping() is False
    assert j_no_deg.raw_to_deg(2047) is None
    with pytest.raises(ValueError):
        j_no_deg.deg_to_raw(0.0)


def test_deg_mapping_linear():
    """min_deg ↔ home_raw（0%），max_deg ↔ max_raw（100%）。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500,
                    min_deg=0.0, max_deg=180.0)
    assert j.deg_to_raw(0.0) == 1000      # home / 0% / min_deg
    assert j.deg_to_raw(180.0) == 3500     # max / 100% / max_deg
    assert j.deg_to_raw(90.0) == round(1000 + 0.5 * (3500 - 1000))
    assert abs(j.raw_to_deg(1000) - 0.0) < 1e-6
    assert abs(j.raw_to_deg(3500) - 180.0) < 1e-6


def test_deg_clamps_out_of_range():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500,
                    min_deg=0.0, max_deg=180.0)
    assert j.deg_to_raw(-30) == j.deg_to_raw(0)
    assert j.deg_to_raw(360) == j.deg_to_raw(180)


def test_invalid_home_max_equal():
    from arm.arm import JointConfig
    with pytest.raises(ValueError):
        JointConfig(name="t", id=1, home_raw=2000, max_raw=2000)


# ---------- 双向关节（low_raw） ----------

def test_bidirectional_pct_to_raw():
    """home 在中间，max 一端 +100%，low 另一端 -100%。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=500)
    assert j.is_bidirectional is True
    assert j.pct_to_raw(0.0) == 2000
    assert j.pct_to_raw(+1.0) == 3500
    assert j.pct_to_raw(-1.0) == 500
    # 半程
    assert j.pct_to_raw(+0.5) == round(2000 + 0.5 * (3500 - 2000))
    assert j.pct_to_raw(-0.5) == round(2000 + 0.5 * (500 - 2000))


def test_bidirectional_raw_to_pct():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=500)
    assert j.raw_to_pct(2000) == 0.0
    assert j.raw_to_pct(3500) == 1.0
    assert j.raw_to_pct(500) == -1.0
    # 中间值
    assert abs(j.raw_to_pct(2750) - 0.5) < 1e-9
    assert abs(j.raw_to_pct(1250) - (-0.5)) < 1e-9


def test_bidirectional_clamps_at_minus_one_and_plus_one():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=500)
    assert j.pct_to_raw(-2.0) == 500
    assert j.pct_to_raw(+2.0) == 3500
    assert j.raw_to_pct(0) == -1.0      # 超出 low 侧
    assert j.raw_to_pct(99999) == 1.0   # 超出 max 侧


def test_unidirectional_negative_pct_clamped_to_zero():
    """单向关节传负值应被夹到 0%（home_raw）。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500)
    assert j.is_bidirectional is False
    assert j.pct_to_raw(-0.5) == 2000
    assert j.pct_to_raw(-1.0) == 2000
    # raw 在 home 反侧时 pct 应该是 0
    assert j.raw_to_pct(500) == 0.0


def test_bidirectional_reversed_direction():
    """home 数值最大、max 在低值方向、low 在高值方向也合法。"""
    from arm.arm import JointConfig
    # home=2048, max=300（往低走 = 正方向），low=3700（往高走 = 负方向）
    j = JointConfig(name="t", id=1, home_raw=2048, max_raw=300, low_raw=3700)
    assert j.pct_to_raw(0.0) == 2048
    assert j.pct_to_raw(+1.0) == 300
    assert j.pct_to_raw(-1.0) == 3700
    assert j.raw_to_pct(300) == 1.0
    assert j.raw_to_pct(3700) == -1.0


def test_bidirectional_invalid_low_same_side_as_max():
    """low 和 max 必须在 home 的不同侧，否则配置非法。"""
    from arm.arm import JointConfig
    with pytest.raises(ValueError, match="不同侧"):
        # home=2000, max=3500（往大）, low=2500（也往大）→ 错
        JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=2500)


def test_bidirectional_invalid_low_eq_home():
    from arm.arm import JointConfig
    with pytest.raises(ValueError):
        JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=2000)


def test_bidirectional_deg_mapping():
    """双向关节的 deg：min_deg ↔ low（-100%），max_deg ↔ max（+100%）。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=500,
                    min_deg=-90.0, max_deg=+90.0)
    assert j.deg_to_raw(0.0) == 2000      # home / 0%
    assert j.deg_to_raw(+90.0) == 3500    # max / +100%
    assert j.deg_to_raw(-90.0) == 500     # low / -100%
    assert abs(j.raw_to_deg(2000) - 0.0) < 1e-6
    assert abs(j.raw_to_deg(3500) - 90.0) < 1e-6
    assert abs(j.raw_to_deg(500) - (-90.0)) < 1e-6


def test_unidirectional_deg_mapping_unchanged():
    """单向关节 deg 仍然是 min↔home, max↔max（与之前一致）。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500,
                    min_deg=0.0, max_deg=180.0)
    assert j.deg_to_raw(0.0) == 1000
    assert j.deg_to_raw(180.0) == 3500


# ---------- 3 点分段线性度数映射（home_deg） ----------

def test_home_deg_default_is_zero():
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=500,
                    min_deg=-90.0, max_deg=+90.0)
    assert j.home_deg == 0.0


def test_asymmetric_bidirectional_deg_anchored_at_home():
    """非对称范围（J3 风格）：home_deg=0 → home 对到物理 0°，不靠中点。"""
    from arm.arm import JointConfig
    # J3 真实参数：home=2048(0°), max=232(-159.8°), low=2445(+34.9°)
    j = JointConfig(name="elbow", id=3,
                    home_raw=2048, max_raw=232, low_raw=2445,
                    min_deg=34.9, max_deg=-159.8)
    # 三个锚点
    assert j.raw_to_deg(2048) == pytest.approx(0.0, abs=1e-6)
    assert j.raw_to_deg(232) == pytest.approx(-159.8, abs=1e-6)
    assert j.raw_to_deg(2445) == pytest.approx(34.9, abs=1e-6)
    # 反向：home_deg ↔ home_raw
    assert j.deg_to_raw(0.0) == 2048
    assert j.deg_to_raw(-159.8) == 232
    assert j.deg_to_raw(34.9) == 2445
    # 越 max 侧极限 → 夹到 max_raw
    assert j.deg_to_raw(-200) == 232
    # 越 low 侧极限 → 夹到 low_raw
    assert j.deg_to_raw(50) == 2445


def test_asymmetric_deg_piecewise_linear_each_side():
    """home 两侧分段独立线性，斜率不同。"""
    from arm.arm import JointConfig
    j = JointConfig(name="elbow", id=3,
                    home_raw=2048, max_raw=232, low_raw=2445,
                    min_deg=34.9, max_deg=-159.8)
    # max 侧：pct=+0.5 ↔ home 到 max 中点（int round 误差 ≈ 0.05°）
    half_max_raw = j.pct_to_raw(0.5)
    assert j.raw_to_deg(half_max_raw) == pytest.approx(-159.8 / 2, abs=0.1)
    # low 侧：pct=-0.5 ↔ home 到 low 中点
    half_low_raw = j.pct_to_raw(-0.5)
    assert j.raw_to_deg(half_low_raw) == pytest.approx(34.9 / 2, abs=0.1)


def test_home_deg_nonzero():
    """home_deg 可以不为 0：例如关节安装时偏移了 -45°。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=2000, max_raw=3500, low_raw=500,
                    min_deg=-90.0, max_deg=+90.0, home_deg=-45.0)
    # home 现在对到 -45°
    assert j.raw_to_deg(2000) == pytest.approx(-45.0, abs=1e-6)
    # max 仍然是 +90°
    assert j.raw_to_deg(3500) == pytest.approx(90.0, abs=1e-6)
    assert j.raw_to_deg(500) == pytest.approx(-90.0, abs=1e-6)
    # 反向也对
    assert j.deg_to_raw(-45.0) == 2000


def test_unidirectional_min_deg_optional():
    """单向关节 has_deg_mapping 只看 max_deg（min_deg 不参与）。"""
    from arm.arm import JointConfig
    j = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500, max_deg=180.0)
    assert j.has_deg_mapping() is True
    assert j.raw_to_deg(1000) == pytest.approx(0.0, abs=1e-6)   # home_deg 默认 0
    assert j.raw_to_deg(3500) == pytest.approx(180.0, abs=1e-6)
    assert j.deg_to_raw(0.0) == 1000
    assert j.deg_to_raw(180.0) == 3500


def test_pct_min_max_attributes():
    """关节自带 pct_min / pct_max 让应用层能查询合法范围。"""
    from arm.arm import JointConfig
    uni = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500)
    assert uni.pct_min == 0.0 and uni.pct_max == 1.0
    bi = JointConfig(name="t", id=1, home_raw=1000, max_raw=3500, low_raw=200)
    assert bi.pct_min == -1.0 and bi.pct_max == 1.0


def test_yaml_loads_low_raw(tmp_path):
    """yaml 含 low_raw 应被加载为双向关节。"""
    from arm.arm import ArmConfig
    cfg_text = """
bus:
  port: auto
  baudrate: 1000000
joints:
  - name: rotor
    id: 1
    home_raw: 2000
    max_raw: 3500
    low_raw: 500
    max_speed: 1500
    max_acc: 50
  - name: lifter
    id: 2
    home_raw: 1000
    max_raw: 4000
    max_speed: 1200
    max_acc: 30
"""
    p = tmp_path / "bi.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    cfg = ArmConfig.from_yaml(p)
    assert cfg.joints[0].is_bidirectional is True
    assert cfg.joints[0].low_raw == 500
    assert cfg.joints[1].is_bidirectional is False
    assert cfg.joints[1].low_raw is None


def test_yaml_ignores_legacy_fields(tmp_path):
    """从老 yaml 加载时，未知字段（direction / min_raw / pct_invert）应被忽略并发警告。"""
    import warnings
    from arm.arm import ArmConfig
    cfg_text = """
bus:
  port: auto
  baudrate: 1000000
joints:
  - name: legacy
    id: 1
    home_raw: 1000
    max_raw: 4095
    min_raw: 0               # 老字段（v1 三点标定）
    pct_invert: true         # 老字段
    direction: -1            # 老字段（v0）
    min_angle_deg: -90       # 老字段
    max_angle_deg: +90       # 老字段
    max_speed: 1500
    max_acc: 50
"""
    p = tmp_path / "legacy.yaml"
    p.write_text(cfg_text, encoding="utf-8")
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        cfg = ArmConfig.from_yaml(p)
        assert any("legacy" in str(x.message) for x in w)
    assert cfg.joints[0].name == "legacy"
    assert cfg.joints[0].home_raw == 1000
    assert cfg.joints[0].max_raw == 4095
