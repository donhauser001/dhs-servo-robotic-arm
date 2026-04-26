"""半双工串口总线（SMS_STS over TTL）。

负责：
    - 自动选择串口（macOS / Linux / 树莓派 / Windows 各有约定）
    - 加锁的事务式读写（一次只能有一个请求在飞）
    - 流式解析响应：自动跳过 self-echo 和噪声，找到第一个合法的状态包
"""

from __future__ import annotations

import glob
import logging
import platform
import threading
import time
from typing import Optional

import serial

from . import protocol as p
from .protocol import (
    BROADCAST_ID,
    ChecksumError,
    HEADER,
    Packet,
    ProtocolError,
    StatusError,
    TimeoutError_,
    checksum,
    parse_response,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 串口自动发现
# ---------------------------------------------------------------------------

def _candidate_ports() -> list[str]:
    """按当前平台列出可能的串口候选（按优先级排序）。"""
    sys = platform.system()
    candidates: list[str] = []
    if sys == "Darwin":
        # macOS：优先 CH343/CH340/CP210x，之后通用 usbserial
        for pat in (
            "/dev/cu.wchusbserial*",
            "/dev/cu.usbserial-*",
            "/dev/cu.SLAB_USBtoUART*",
            "/dev/cu.usbmodem*",
        ):
            candidates.extend(sorted(glob.glob(pat)))
    elif sys == "Linux":
        # 树莓派：GPIO 串口 /dev/serial0；USB 档：ttyUSB*
        for pat in (
            "/dev/ttyUSB*",
            "/dev/ttyACM*",
            "/dev/serial0",
            "/dev/ttyAMA*",
            "/dev/ttyS*",
        ):
            candidates.extend(sorted(glob.glob(pat)))
    elif sys == "Windows":
        try:
            from serial.tools import list_ports
            candidates = [p.device for p in list_ports.comports()]
        except Exception:
            candidates = [f"COM{i}" for i in range(1, 30)]
    return candidates


def autodetect_port() -> Optional[str]:
    """挑出最有可能是舵机驱动板的串口（不实际打开，只列出候选第一个）。"""
    cands = _candidate_ports()
    return cands[0] if cands else None


# ---------------------------------------------------------------------------
# Bus
# ---------------------------------------------------------------------------

class Bus:
    """SMS_STS 半双工总线。线程安全。

    用法：
        with Bus("/dev/cu.usbserial-XYZ") as bus:
            pkt = bus.transact(Packet.ping(1))
    """

    def __init__(
        self,
        port: Optional[str] = None,
        baudrate: int = 1_000_000,
        timeout: float = 0.05,
        retries: int = 2,
    ) -> None:
        if port is None or port == "auto":
            port = autodetect_port()
            if port is None:
                raise RuntimeError(
                    "未找到可用串口；请显式传入 port=... 或检查 USB 是否连接、驱动是否安装"
                )
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.retries = retries
        self._ser: Optional[serial.Serial] = None
        self._lock = threading.RLock()

    # ---- 生命周期 ----

    def open(self) -> "Bus":
        if self._ser is not None and self._ser.is_open:
            return self
        self._ser = serial.Serial(
            port=self.port,
            baudrate=self.baudrate,
            bytesize=8,
            parity=serial.PARITY_NONE,
            stopbits=1,
            timeout=self.timeout,
            write_timeout=0.5,
        )
        time.sleep(0.05)
        self._ser.reset_input_buffer()
        self._ser.reset_output_buffer()
        log.info("opened serial %s @ %d bps", self.port, self.baudrate)
        return self

    def close(self) -> None:
        if self._ser is not None and self._ser.is_open:
            self._ser.close()
        self._ser = None

    def __enter__(self) -> "Bus":
        return self.open()

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    @property
    def is_open(self) -> bool:
        return self._ser is not None and self._ser.is_open

    # ---- 低层收发 ----

    def _ensure_open(self) -> serial.Serial:
        if self._ser is None or not self._ser.is_open:
            self.open()
        assert self._ser is not None
        return self._ser

    def _write_raw(self, data: bytes) -> None:
        ser = self._ensure_open()
        ser.reset_input_buffer()
        ser.write(data)
        ser.flush()

    def _read_response(self, expected_total: float = 0.05) -> Packet:
        """流式读取一个响应包：跳过 echo / 噪声，定位 0xFF 0xFF 包头后解析。

        expected_total: 等待整个响应到齐的时间预算（秒）。
        """
        ser = self._ensure_open()
        deadline = time.monotonic() + expected_total
        buf = bytearray()

        # 1) 找包头 0xFF 0xFF（同时跳过 echo）
        while time.monotonic() < deadline:
            chunk = ser.read(64)
            if chunk:
                buf.extend(chunk)
                idx = buf.find(HEADER)
                if idx >= 0:
                    # 把包头之前的字节（self-echo + 噪声）丢掉
                    del buf[:idx]
                    break
            else:
                if buf:
                    # 已经收到一些字节但还没找到包头：再等一下
                    continue
        else:
            raise TimeoutError_(f"no header within {expected_total*1000:.0f} ms; got={bytes(buf)!r}")

        # 2) 至少需要 6 字节才能解析
        while len(buf) < 6 and time.monotonic() < deadline:
            chunk = ser.read(6 - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < 6:
            raise TimeoutError_(f"truncated header: {bytes(buf)!r}")

        length = buf[3]
        total = 4 + length
        # 3) 等齐剩余字节
        while len(buf) < total and time.monotonic() < deadline:
            chunk = ser.read(total - len(buf))
            if chunk:
                buf.extend(chunk)
        if len(buf) < total:
            raise TimeoutError_(f"truncated frame: have {len(buf)} need {total}: {bytes(buf)!r}")

        return parse_response(bytes(buf[:total]))

    # ---- 事务接口 ----

    def transact(
        self,
        pkt: Packet,
        expect_response: Optional[bool] = None,
        raise_on_error: bool = True,
    ) -> Optional[Packet]:
        """发送一个数据包，按需读取响应。

        默认行为：广播 ID（0xFE）和 ACTION/SYNC_WRITE 不读响应；其他都读。
        """
        if expect_response is None:
            expect_response = pkt.sid != BROADCAST_ID

        with self._lock:
            last_err: Optional[Exception] = None
            for attempt in range(self.retries + 1):
                try:
                    self._write_raw(pkt.encode())
                    if not expect_response:
                        return None
                    resp = self._read_response()
                    # 状态包的 sid 必须等于请求的 sid（除非广播）
                    if resp.sid != pkt.sid and pkt.sid != BROADCAST_ID:
                        raise ProtocolError(f"sid mismatch: req={pkt.sid} resp={resp.sid}")
                    if raise_on_error and resp.inst_or_err != 0:
                        raise StatusError(resp.sid, resp.inst_or_err)
                    return resp
                except (TimeoutError_, ChecksumError, ProtocolError) as e:
                    last_err = e
                    log.debug("transact attempt %d/%d failed: %s", attempt + 1, self.retries + 1, e)
                    time.sleep(0.005)
            assert last_err is not None
            raise last_err

    # ---- 高层便捷读写 ----

    def ping(self, sid: int) -> bool:
        """返回 True 表示该 ID 在线。"""
        try:
            self.transact(Packet.ping(sid))
            return True
        except (TimeoutError_, ProtocolError):
            return False

    def read_bytes(self, sid: int, addr: int, length: int) -> bytes:
        resp = self.transact(Packet.read(sid, addr, length))
        assert resp is not None
        if len(resp.params) != length:
            raise ProtocolError(
                f"read length mismatch: requested {length}, got {len(resp.params)} "
                f"from servo #{sid} addr 0x{addr:02X}"
            )
        return resp.params

    def write_bytes(self, sid: int, addr: int, data: bytes) -> None:
        self.transact(Packet.write(sid, addr, data))

    def read_u8(self, sid: int, addr: int) -> int:
        return self.read_bytes(sid, addr, 1)[0]

    def write_u8(self, sid: int, addr: int, value: int) -> None:
        self.write_bytes(sid, addr, bytes([value & 0xFF]))

    def read_u16(self, sid: int, addr: int, signed_bit15: bool = False) -> int:
        b = self.read_bytes(sid, addr, 2)
        return p.from_le16(b[0], b[1], signed_bit15=signed_bit15)

    def write_u16(self, sid: int, addr: int, value: int) -> None:
        self.write_bytes(sid, addr, p.to_le16(value))


__all__ = ["Bus", "autodetect_port"]
