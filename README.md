# 舵机机械臂 2 号

> 6 自由度总线舵机机械臂控制软件
> 树莓派 5 + Bus Servo Driver HAT (A) + 飞特 ST3020/ST3025/ST3235

## 文档

- [硬件介绍](docs/硬件介绍.md) — 部件、接线、协议、风险

## 快速开始

### 1. 安装依赖

```bash
# 推荐用虚拟环境
python3 -m venv .venv
source .venv/bin/activate

# 必装
pip install -e .

# 可选：Web 控制面板
pip install -e ".[web]"

# 可选：开发与测试
pip install -e ".[dev]"
```

### 2. 跑单元测试（不需要硬件）

```bash
pytest -q tests/
```

### 3. 扫描总线舵机

```bash
# 自动选串口，扫描 ID 1..20
python -m tools.scan

# 显式指定串口
python -m tools.scan --port /dev/cu.usbserial-XYZ

# 全段扫描（首次接线、不知道 ID 时用）
python -m tools.scan --range 0 253
```

### 4. 给舵机分配 ID（首次必做）

舵机出厂 ID 都是 `1`，必须**逐颗**改成 1..6：

```bash
# 1) 总线只接 1 颗舵机 → 改成 ID=2
python -m tools.set_id --to 2

# 2) 拔掉，换上下一颗 → 改成 ID=3
python -m tools.set_id --to 3

# ... 直到 ID=6 全部分配完成

# 把 6 颗串起来再扫描验证
python -m tools.scan
```

### 5. 键盘点动测试

```bash
python -m tools.jog
```

按键速查：

| 键 | 作用 |
| :---: | :--- |
| `1`–`6` | 选关节 |
| `←` / `→` | 切换关节 |
| `q` / `a` | 当前关节位置 ±50（粗调，约 ±4.4°）|
| `w` / `s` | 当前关节位置 ±5（细调，约 ±0.44°）|
| `t` | 切换扭矩开/关（关闭后可手动摆动）|
| `h` | 全部关节回零位 |
| `r` | 刷新反馈 |
| `空格` | 紧急停止（放扭矩）|
| `ESC` / `x` | 退出 |

## 工程结构

```
舵机机械臂/
├── README.md                      项目说明
├── pyproject.toml                 依赖管理
├── docs/硬件介绍.md                硬件文档
├── config/arm_config.yaml         机械臂配置（关节、限位、零位）
├── arm/                           核心 Python 包
│   ├── protocol.py                SMS_STS 协议封装
│   ├── bus.py                     半双工串口总线
│   ├── servo.py                   单舵机抽象
│   └── arm.py                     6 DOF 整体控制器
├── tools/                         命令行工具
│   ├── scan.py                    扫描舵机
│   ├── set_id.py                  设置 ID
│   └── jog.py                     键盘点动
└── tests/test_protocol.py         协议层单元测试
```

## 平台支持

| 系统 | 串口默认 | 备注 |
| :--- | :--- | :--- |
| **macOS** | `/dev/cu.wchusbserial*` 或 `/dev/cu.usbserial-*` | CH343 驱动可能要装：[CH343SER_MAC](https://www.wch.cn/downloads/CH343SER_MAC_ZIP.html) |
| **Linux / 树莓派 (USB 档)** | `/dev/ttyUSB0` | 需要 `dialout` 组：`sudo usermod -a -G dialout $USER` |
| **Linux / 树莓派 (ESP32 档)** | `/dev/serial0` | 先启用串口 + 释放蓝牙，见硬件文档 §3.1 |
| **Windows** | `COM3` 等 | `pyserial` 自动处理 |

## 已知限制（v0.1）

- `tools/jog.py` 暂不支持 Windows（依赖 `termios`），后续会用 `msvcrt` 兼容
- 还没有运动学（正/逆解）—— 在 `arm/kinematics.py` 中规划
- 还没有 Web 控制面板 —— 在 `web/` 中规划
- 还没有示教 / 回放 —— 在 `tools/teach.py` `tools/replay.py` 中规划
