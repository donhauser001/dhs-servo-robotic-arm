"""SMS_STS 协议层（飞特 / Feetech ST 系列总线舵机）。

兼容型号：ST3020 / ST3025 / ST3235 / ST3215 / ST3032 等所有走 SMS_STS 协议的舵机。
通信参数：1Mbps 默认，TTL 半双工，小端字节序。

数据包结构：
    Request:  0xFF 0xFF  ID  LEN  INST  [PARAM...]  CHKSUM
    Response: 0xFF 0xFF  ID  LEN  ERR   [PARAM...]  CHKSUM
其中：
    LEN    = 参数字节数 + 2（INST/ERR + CHKSUM）
    CHKSUM = ~(ID + LEN + INST + PARAM...) & 0xFF
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Iterable, Sequence


HEADER = b"\xff\xff"
BROADCAST_ID = 0xFE


class Instruction(IntEnum):
    """SMS_STS 指令字。"""

    PING = 0x01
    READ = 0x02
    WRITE = 0x03
    REG_WRITE = 0x04   # 寄存器写（缓存，等 ACTION 触发）
    ACTION = 0x05
    SYNC_READ = 0x82
    SYNC_WRITE = 0x83
    RESET = 0x06


class Reg(IntEnum):
    """SMS_STS 内存表关键地址（按舵机数据手册整理）。

    标注 `(2)` 的为 16-bit 小端，其余 8-bit。
    标注 `EPROM` 的写入前需先解锁（写 0 到 LOCK），写完上锁，掉电保存。
    """

    # ---- EPROM (掉电保存) ----
    MODEL_L = 0x03            # (2) 型号
    ID = 0x05                 #     舵机 ID
    BAUD_RATE = 0x06          #     波特率档位（0=1Mbps, 1=500k, ... 7=4800）
    RETURN_DELAY = 0x07
    RESPONSE_LEVEL = 0x08
    MIN_ANGLE_LIMIT = 0x09    # (2) 最小角度限位
    MAX_ANGLE_LIMIT = 0x0B    # (2) 最大角度限位
    MAX_TEMP_LIMIT = 0x0D
    MAX_VOLTAGE = 0x0E
    MIN_VOLTAGE = 0x0F
    MAX_TORQUE = 0x10         # (2)
    PHASE = 0x12
    UNLOAD_CONDITION = 0x13
    LED_ALARM_CONDITION = 0x14
    KP = 0x15                 # 位置环 P
    KD = 0x16                 # 位置环 D
    KI = 0x17                 # 位置环 I
    MIN_STARTUP_FORCE = 0x18  # (2)
    CW_DEAD = 0x1A
    CCW_DEAD = 0x1B
    PROTECT_CURRENT = 0x1C    # (2)
    ANGULAR_RESOLUTION = 0x1E
    OFFSET = 0x1F             # (2) 位置校准
    OPERATION_MODE = 0x21     # 0=位置, 1=电机/连续旋转
    PROTECT_TORQUE = 0x22
    PROTECT_TIME = 0x23
    OVERLOAD_TORQUE = 0x24

    # ---- SRAM (掉电不保存) ----
    TORQUE_ENABLE = 0x28      # 0=放扭矩, 1=锁扭矩, 128=校零位
    ACC = 0x29                # 启停加速度 0-150
    GOAL_POSITION = 0x2A      # (2) 目标位置 0-4095
    GOAL_TIME = 0x2C          # (2)
    GOAL_SPEED = 0x2E         # (2) 目标速度 0-3073（步/秒，50 步/秒 ≈ 0.732 RPM）
    LOCK = 0x37               # EPROM 锁：0=解锁, 1=上锁

    # ---- 反馈区（只读）----
    PRESENT_POSITION = 0x38   # (2)
    PRESENT_SPEED = 0x3A      # (2) 高位为方向位
    PRESENT_LOAD = 0x3C       # (2) 高位为方向位
    PRESENT_VOLTAGE = 0x3E    # 单位 0.1V
    PRESENT_TEMPERATURE = 0x3F  # ℃
    STATUS = 0x41
    MOVING = 0x42
    PRESENT_CURRENT = 0x45    # (2) 单位 6.5mA


class Error(IntEnum):
    """状态包 ERR 字节的位标志。"""

    VOLTAGE = 0x01
    ANGLE = 0x02
    OVERHEAT = 0x04
    OVER_RANGE = 0x08
    CHECKSUM = 0x10
    OVERLOAD = 0x20
    INSTRUCTION = 0x40


class ProtocolError(RuntimeError):
    """协议层异常基类。"""


class ChecksumError(ProtocolError):
    pass


class TimeoutError_(ProtocolError):
    pass


class StatusError(ProtocolError):
    def __init__(self, sid: int, err: int):
        flags = [e.name for e in Error if err & e.value]
        super().__init__(f"servo #{sid} status error 0x{err:02X} [{', '.join(flags) or 'unknown'}]")
        self.sid = sid
        self.err = err


# ---------------------------------------------------------------------------
# 工具函数：校验和、字节序
# ---------------------------------------------------------------------------

def checksum(payload: Iterable[int]) -> int:
    """SMS_STS 校验和：所有字节求和取反低 8 位（不含包头 0xFF 0xFF）。"""
    return (~sum(payload)) & 0xFF


def to_le16(value: int) -> bytes:
    """16-bit 有符号 → 小端 2 字节。

    SMS_STS 用最高位作为符号位（speed/load），所以这里按舵机原始格式编码：
    负数则置位 bit15。
    """
    v = int(value)
    if v < 0:
        v = (-v) & 0x7FFF | 0x8000
    else:
        v = v & 0xFFFF
    return bytes([v & 0xFF, (v >> 8) & 0xFF])


def from_le16(lo: int, hi: int, signed_bit15: bool = False) -> int:
    """小端 2 字节 → 整数。

    signed_bit15=True 时按 SMS_STS 的"符号位 = bit15"格式解码（speed/load）。
    """
    raw = (hi << 8) | lo
    if signed_bit15:
        if raw & 0x8000:
            return -(raw & 0x7FFF)
        return raw
    return raw


# ---------------------------------------------------------------------------
# 数据包构造
# ---------------------------------------------------------------------------

@dataclass
class Packet:
    """协议数据包（请求或响应）。"""

    sid: int
    inst_or_err: int
    params: bytes = b""

    def encode(self) -> bytes:
        length = len(self.params) + 2
        body = bytes([self.sid, length, self.inst_or_err]) + self.params
        return HEADER + body + bytes([checksum(body)])

    # ---- 常用请求构造器 ----

    @classmethod
    def ping(cls, sid: int) -> "Packet":
        return cls(sid, Instruction.PING)

    @classmethod
    def read(cls, sid: int, addr: int, length: int) -> "Packet":
        return cls(sid, Instruction.READ, bytes([addr, length]))

    @classmethod
    def write(cls, sid: int, addr: int, data: bytes) -> "Packet":
        return cls(sid, Instruction.WRITE, bytes([addr]) + data)

    @classmethod
    def reg_write(cls, sid: int, addr: int, data: bytes) -> "Packet":
        return cls(sid, Instruction.REG_WRITE, bytes([addr]) + data)

    @classmethod
    def action(cls) -> "Packet":
        return cls(BROADCAST_ID, Instruction.ACTION)

    @classmethod
    def sync_write(cls, addr: int, data_len: int, items: Sequence[tuple[int, bytes]]) -> "Packet":
        """构造同步写包：一次给多个舵机各写 data_len 字节到 addr。

        items: [(servo_id, payload_bytes), ...]，payload_bytes 长度必须等于 data_len。
        """
        params = bytearray([addr, data_len])
        for sid, payload in items:
            if len(payload) != data_len:
                raise ValueError(f"sync_write payload for #{sid} must be {data_len} bytes")
            params.append(sid)
            params.extend(payload)
        return cls(BROADCAST_ID, Instruction.SYNC_WRITE, bytes(params))


def parse_response(buf: bytes) -> Packet:
    """解析一帧状态包，校验通过则返回 Packet（其中 inst_or_err 为错误字节）。

    抛出 ChecksumError / ProtocolError。
    """
    if len(buf) < 6:
        raise ProtocolError(f"response too short: {buf!r}")
    if buf[0:2] != HEADER:
        raise ProtocolError(f"bad header: {buf[:2]!r}")
    sid = buf[2]
    length = buf[3]
    expected_total = 4 + length
    if len(buf) < expected_total:
        raise ProtocolError(f"truncated frame: got {len(buf)} need {expected_total}")
    err = buf[4]
    params = bytes(buf[5 : 4 + length - 1])
    chk = buf[4 + length - 1]
    if chk != checksum(buf[2 : 4 + length - 1]):
        raise ChecksumError(f"checksum mismatch: {buf!r}")
    return Packet(sid=sid, inst_or_err=err, params=params)


__all__ = [
    "HEADER",
    "BROADCAST_ID",
    "Instruction",
    "Reg",
    "Error",
    "ProtocolError",
    "ChecksumError",
    "TimeoutError_",
    "StatusError",
    "Packet",
    "checksum",
    "to_le16",
    "from_le16",
    "parse_response",
]
