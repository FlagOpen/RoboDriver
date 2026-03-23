# robodriver-teleoperator-pico-ultra4-dora

## 项目说明

本项目将 Pico Ultra4 VR 控制器的遥操作集成进 dora dataflow 框架，支持：
- **Pico Ultra4** VR 控制器（通过 XRoboToolkit SDK）
- **Piper** 机械臂（CAN 总线控制，含 IK 求解）
- **RealSense D405** 手腕相机
- **Orbbec Gemini 335** 顶部相机

核心节点 `dora_node_piper.py` 将 `PiperTeleopController` 适配为 dora 事件驱动模式，在 50Hz tick 下执行 IK + 控制循环。

---

## 快速启动

### 前置条件

```bash
# 1. 启动 CAN 总线
sudo ip link set can0 up type can bitrate 1000000

# 2. 确认 XRoboToolkit PC Service 正在运行（监听 127.0.0.1:60061）
ss -tlnp | grep 60061
# 应看到 RoboticsService 进程

# 3. Pico 头显连接同一局域网，打开 XRoboToolkit App
```

### 启动遥操

```bash
cd RoboDriver/robodriver/teleoperators/robodriver-teleoperator-pico-ultra4-dora

# 清理残留进程（重要！避免相机设备被占用）
pkill -f "dora run\|dora_node_piper\|dora_camera\|dora_arm_piper" 2>/dev/null; sleep 1

# 重置 Orbbec 相机 USB（重要！dora 被强杀后 Orbbec SDK 的 UVC 锁不会自动释放）
# 不加这步会报 uvc_open res-6 / Device or resource busy
sudo usbreset 2bc5:0800; sleep 1

# 激活虚拟环境并启动
source .venv/bin/activate
dora run dora/dataflow.yml
```

### 操作说明

- 按住 **右手 grip 键** 激活控制（`right_arm is activated` 出现后生效）
- 移动手柄控制 Piper 末端位姿（IK 实时求解）
- 松开 grip 键暂停控制
- **右手扳机键** 控制夹爪开合
- `Ctrl+C` 停止

---

## 依赖安装（首次配置）

XR Robotics中下载XRoboToolkit-PC-Service、XRoboToolkit-Teleop-Sample-Python，pico中安装xrobotoolkits
```bash
# Pico 遥操依赖(本地安装)
pip install XRoboToolkit-Teleop-Sample-Python/dependencies/XRoboToolkit-PC-Service-Pybind/
pip install -e XRoboToolkit-Teleop-Sample-Python/
pip install placo meshcat

# 验证
python -c "import xrobotoolkit_sdk, xrobotoolkit_teleop, placo, meshcat; print('all ok')"
```

---

### 步骤 4：创建新的统一虚拟环境

```bash
# 创建 Python 3.10 虚拟环境
uv venv .venv -p 3.10

# 激活虚拟环境
source .venv/bin/activate
```

### 步骤 5：安装依赖

```bash
# 安装基础依赖
uv pip install -e .

# 安装硬件依赖（RealSense、Orbbec、Piper）
uv pip install -e .[hardware]
```

这将安装以下硬件驱动：
- `pyrealsense2` (2.56.5.9235) - Intel RealSense D405 相机驱动
- `sb-pyorbbecsdk` (1.3.1) - Orbbec Gemini 335 相机驱动
- `piper-sdk` (0.6.1) - Piper 机械臂 SDK

### 步骤 6：验证安装

```bash
# 验证 Python 可执行文件位置
python -c "import sys; print('Python:', sys.executable)"

# 验证硬件 SDK 是否正确安装
python -c "import pyrealsense2; print('✅ RealSense SDK 已安装')"
python -c "import piper_sdk; print('✅ Piper SDK 已安装')"
python -c "import pyorbbecsdk; print('✅ Orbbec SDK 已安装')"
```

预期输出：
```
Python: /home/dora/RoboDriver/robodriver/teleoperators/robodriver-teleoperator-pico-ultra4-dora/.venv/bin/python
✅ RealSense SDK 已安装
✅ Piper SDK 已安装
✅ Orbbec SDK 已安装
```


## 启动数据采集（仅支持单臂遥操）
## 1.激活piper机械臂
```bash
cd SDK/piper_sdk/piper_sdk
bash can_config.sh
```
- pc端打开XRobotoolkits-pc-server，pico中启动xrobotoolkits(安装后在资源库中，与pc端连接到一个网段上，Head，Controller、Hand、Send、Switch W/A Button,Trcking、Vision都选择上)
### 1. 激活虚拟环境

```bash
cd /home/dora/RoboDriver/robodriver/teleoperators/robodriver-teleoperator-pico-ultra4-dora
source .venv/bin/activate
```

### 2. 启动 Dora 服务

```bash
dora up
```

### 3. 启动数据流

```bash
dora start dora/dataflow.yml
```

## 相关文档

- [RoboDriver 主文档](../../README.md)
- [Dora 框架文档](https://dora-rs.ai/)
- [RealSense SDK 文档](https://github.com/IntelRealSense/librealsense)
- [Orbbec SDK 文档](https://github.com/orbbec/pyorbbecsdk)
- [Piper SDK 文档](https://github.com/agilexrobotics/piper_sdk)
