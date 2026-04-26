"""单舵机抽象。

封装一颗 SMS_STS 舵机的所有常用操作：位置读写、扭矩、反馈、ID/限位/中位等 EPROM 设置。
所有操作通过共享的 :class:`Bus` 实例完成。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import protocol as p
from .bus import Bus
from .protocol import Reg


# 一些可读常量
POSITION_RESOLUTION = 4096  # 0..4095 = 360°
DEG_PER_STEP = 360.0 / POSITION_RESOLUTION


@dataclass
class Feedback:
    """舵机一次完整反馈数据快照。"""

    position: int          # 0..4095
    speed: int             # 步/秒，正负表示方向
    load: int              # -1000..+1000，负载百分比×10
    voltage: float         # V
    temperature: int       # ℃
    moving: bool
    current_ma: Optional[float] = None  # mA（只有 ST3020/ST3025 有该反馈，ST3235 没有）

    @property
    def position_deg(self) -> float:
        return (self.position - POSITION_RESOLUTION // 2) * DEG_PER_STEP


class Servo:
    """单颗 SMS_STS 舵机的高层封装。"""

    def __init__(self, bus: Bus, sid: int):
        self.bus = bus
        self.sid = sid

    # ---- 探活 ----

    def ping(self) -> bool:
        return self.bus.ping(self.sid)

    # ---- 位置读写 ----

    def read_position(self) -> int:
        """读取当前位置（0..4095）。"""
        return self.bus.read_u16(self.sid, Reg.PRESENT_POSITION)

    def write_position(
        self,
        position: int,
        speed: int = 1500,
        acc: int = 50,
    ) -> None:
        """控制舵机以指定速度/加速度转到目标位置。

        position: 0..4095
        speed:    0..3073 步/秒
        acc:      0..150（值越小越柔和）
        """
        position = max(0, min(POSITION_RESOLUTION - 1, int(position)))
        speed = max(0, min(3073, int(speed)))
        acc = max(0, min(150, int(acc)))
        # 一次写 7 字节：ACC(1) + GOAL_POSITION(2) + GOAL_TIME(2) + GOAL_SPEED(2)
        # 这是 SCServo 库 WritePosEx 的标准写法（地址 0x29 起）
        payload = (
            bytes([acc])
            + p.to_le16(position)
            + p.to_le16(0)         # GOAL_TIME = 0：不限时
            + p.to_le16(speed)
        )
        self.bus.write_bytes(self.sid, Reg.ACC, payload)

    # ---- 扭矩 ----

    def torque(self, on: bool) -> None:
        """打开/关闭扭矩。关闭后可以用手转动舵机（用于示教）。"""
        self.bus.write_u8(self.sid, Reg.TORQUE_ENABLE, 1 if on else 0)

    def is_torque_on(self) -> bool:
        return self.bus.read_u8(self.sid, Reg.TORQUE_ENABLE) != 0

    # ---- 反馈 ----

    def feedback(self) -> Feedback:
        """一次性读取主要反馈数据。"""
        # 0x38..0x42 共 11 字节：position(2) speed(2) load(2) voltage(1) temp(1) status(1) ?(1) moving(1)
        b = self.bus.read_bytes(self.sid, Reg.PRESENT_POSITION, 11)
        position = p.from_le16(b[0], b[1])
        speed = p.from_le16(b[2], b[3], signed_bit15=True)
        load_raw = p.from_le16(b[4], b[5], signed_bit15=True)
        voltage = b[6] * 0.1
        temperature = b[7]
        moving = bool(b[10])

        # 电流（如果舵机支持，则在 0x45 起 2 字节，单位约 6.5 mA/LSB）
        current_ma: Optional[float]
        try:
            cur_raw = self.bus.read_u16(self.sid, Reg.PRESENT_CURRENT, signed_bit15=True)
            current_ma = cur_raw * 6.5
        except Exception:
            current_ma = None

        return Feedback(
            position=position,
            speed=speed,
            load=load_raw,
            voltage=voltage,
            temperature=temperature,
            moving=moving,
            current_ma=current_ma,
        )

    # ---- EPROM 写（ID / 限位 / 中位偏移） ----

    def unlock_eprom(self) -> None:
        """解锁 EPROM（写 0 到 LOCK 寄存器）。写完关键参数后必须 lock。"""
        self.bus.write_u8(self.sid, Reg.LOCK, 0)

    def lock_eprom(self) -> None:
        self.bus.write_u8(self.sid, Reg.LOCK, 1)

    def set_id(self, new_id: int) -> None:
        """**永久**改变本舵机 ID。

        调用后 self.sid 自动更新到 new_id。注意：调用时总线上**只能**接这一颗舵机！
        """
        if not 0 <= new_id <= 253:
            raise ValueError(f"invalid id: {new_id}")
        self.unlock_eprom()
        self.bus.write_u8(self.sid, Reg.ID, new_id)
        old = self.sid
        self.sid = new_id  # 后续的 lock 必须用新 ID
        try:
            self.lock_eprom()
        except Exception:
            self.sid = old
            raise

    def set_middle_position(self) -> None:
        """**特殊功能**：把当前位置设为中位（写 128 到 TORQUE_ENABLE）。

        相当于把当前角度作为机械零位，永久写入 EPROM。建议先把关节摆到机械零位再调用。
        """
        self.bus.write_u8(self.sid, Reg.TORQUE_ENABLE, 128)

    def get_offset(self) -> int:
        """读当前位置校正偏移 OFFSET（步，-2047..+2047）。"""
        raw = self.bus.read_u16(self.sid, Reg.OFFSET)
        if raw & 0x800:
            return -(raw & 0x7FF)
        return raw & 0x7FF

    def set_offset(self, offset: int) -> None:
        """设置位置校正偏移 OFFSET（永久写入 EPROM）。

        取值范围：-2047..+2047 步。Feetech SMS_STS 协议中 OFFSET 是 12-bit
        有符号数（bit 11 = 符号位），故编码与 ``to_le16`` 不同。

        作用（实测公式）：
            reported_raw = (physical_raw - offset) mod 4096
            即：offset 越大 → 读数越小；offset 为负 → 读数变大。
            写 GOAL_POSITION 时，舵机内部按同样的偏移做反向换算
            （goal 走到的物理位置 = goal + offset），因此应用层只需关心
            "已偏移后的 raw"，物理行为保持一致。

        典型用法：让当前物理 home 报告为 2048（中位），避免行程跨越 0/4095。
            offset = physical_home - 2048
        """
        if not -2047 <= offset <= 2047:
            raise ValueError(f"offset out of range [-2047, 2047]: {offset}")
        if offset >= 0:
            encoded = offset & 0x7FF
        else:
            encoded = 0x800 | ((-offset) & 0x7FF)
        payload = bytes([encoded & 0xFF, (encoded >> 8) & 0xFF])
        self.unlock_eprom()
        self.bus.write_bytes(self.sid, Reg.OFFSET, payload)
        self.lock_eprom()

    def set_angle_limits(self, min_pos: int, max_pos: int) -> None:
        """设置软限位（永久）。0..4095。设为 0/0 则取消限位（电机模式）。"""
        self.unlock_eprom()
        self.bus.write_u16(self.sid, Reg.MIN_ANGLE_LIMIT, min_pos)
        self.bus.write_u16(self.sid, Reg.MAX_ANGLE_LIMIT, max_pos)
        self.lock_eprom()

    # ---- 工作模式 ----

    def set_position_mode(self) -> None:
        """切换为位置模式（默认）。"""
        self.unlock_eprom()
        self.bus.write_u8(self.sid, Reg.OPERATION_MODE, 0)
        self.lock_eprom()

    def set_motor_mode(self) -> None:
        """切换为电机模式（连续旋转，可用 GOAL_SPEED 控制方向和速度）。"""
        self.unlock_eprom()
        self.bus.write_u8(self.sid, Reg.OPERATION_MODE, 1)
        self.lock_eprom()


__all__ = ["Servo", "Feedback", "POSITION_RESOLUTION", "DEG_PER_STEP"]
