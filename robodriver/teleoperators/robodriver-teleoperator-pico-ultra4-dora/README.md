# robodriver-teleoperator-pico-ultra4-dora

## 项目说明

本项目将 Pico Ultra4 VR 控制器的遥操作集成进 dora dataflow 框架，支持：
- **Pico Ultra4** VR 控制器（通过 XRoboToolkit SDK）
- **Piper** 机械臂（CAN 总线控制，含 IK 求解）
- **RealSense D405** 手腕相机（使用的相机序列号，需要修改）
- **Orbbec Gemini 335** 顶部相机

核心节点 `dora_node_piper.py` 将 `PiperTeleopController` 适配为 dora 事件驱动模式，在 50Hz tick 下执行 IK + 控制循环。

## 步骤1：前置条件
### 激活can通信
- 安装piper的SDK [git@github.com:agilexrobotics/piper_sdk.git]
- XR Robotics中下载XRoboToolkit-PC-Service、XRoboToolkit-Teleop-Sample-Python，pico中安装xrobotoolkits[]
```bash
# Pico 遥操依赖(本地安装)
pip install XRoboToolkit-Teleop-Sample-Python/dependencies/XRoboToolkit-PC-Service-Pybind/
pip install -e XRoboToolkit-Teleop-Sample-Python/
pip install placo meshcat

# 验证
python -c "import xrobotoolkit_sdk, xrobotoolkit_teleop, placo, meshcat; print('all ok')"
```

### 依赖安装（首次配置）

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

## 步骤2：创建新的统一虚拟环境

```bash
# 创建 Python 3.10 虚拟环境
uv venv .venv -p 3.10

# 激活虚拟环境
source .venv/bin/activate
```

## 步骤3：安装依赖

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

### 步骤4：验证安装

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


## 启动数据采集（双臂遥操）
### 1.激活piper机械臂
```bash
# 查找can对应的USB port
bash find_all_can_port.sh 
# 激活can0、can1
bash can_activate.sh can0 1000000 USB-port  # bash can_activate.sh can0 1000000 3-8.4.4:1.0
bash can_activate.sh can1 1000000 USB-port  # bash can_activate.sh can1 1000000 3-8.4.3:1.0
```
- pc端打开XRobotoolkits-pc-server，pico中启动xrobotoolkits(安装后在资源库中，与pc端连接到一个网段上，Head，Controller、Hand、Send、Switch W/A Button,Trcking、Vision都选择上)
### 2. 激活虚拟环境

```bash
cd /home/dora/RoboDriver/robodriver/teleoperators/robodriver-teleoperator-pico-ultra4-dora
source .venv/bin/activate
```

### 3.设置机械臂的Home位置

使用 `scripts/read_joints.py` 脚本管理双臂的 Home 位置（右臂 `can0`，左臂 `can1`）。

```bash
# 持续刷新显示双臂关节状态（每 0.5s 一次，Ctrl+C 停止）
python scripts/read_joints.py

# 只读一次就退出
python scripts/read_joints.py --once

# 将当前位置保存为 home（写入 scripts/home_positions.json）
python scripts/read_joints.py --set-home

# 将双臂发送到已保存的 home 位置
python scripts/read_joints.py --go-home
```

输出格式示例：
```
[RIGHT]  joints: +0.0123  -0.4567  +0.1234  -0.2345  +0.3456  -0.4567  gripper: 0.250
[LEFT ]  joints: +0.0456  -0.3210  +0.1111  -0.2222  +0.3333  -0.4444  gripper: 0.100
```

**典型流程**：手动将双臂摆到合适的初始姿态 → `--set-home` 保存 → 每次启动前 `--go-home` 复位。

### 4. 启动 Dora 服务

```bash
dora up
```

### 5. 启动数据流

```bash
dora start dora/dataflow.yml
```

### 操作说明
- Realsense相机绑定序列号，得根据硬件连接修改序列号。（Realsense的图像画面是通过sdk读取的，奥比中光的相机是通过uvc协议获取的，即/dev/video8.[V4L2 = Linux 下获取视频的编程接口/框架、UVC = USB 摄像头的通信协议标准]
- Pico 头显连接同一局域网，打开 XRoboToolkit App与xrobotoolkits
- 按住 **左右手 grip 键** 激活控制，移动手柄控制 Piper 末端位姿，松开 grip 键暂停控制，保存一组数据；
- **左右手扳机键** 控制夹爪开合

## 注意事项
# 清理残留进程（重要！避免相机设备被占用）
pkill -f "dora run\|dora_node_piper\|dora_camera\|dora_arm_piper" 2>/dev/null; sleep 1

# 重置 Orbbec 相机 USB（重要！dora 被强杀后 Orbbec SDK 的 UVC 锁不会自动释放）
# 不加这步会报 uvc_open res-6 / Device or resource busy
sudo usbreset 2bc5:0800; sleep 1

## 相关文档

- [RoboDriver 主文档](../../README.md)
- [Dora 框架文档](https://dora-rs.ai/)
- [RealSense SDK 文档](https://github.com/IntelRealSense/librealsense)
- [Orbbec SDK 文档](https://github.com/orbbec/pyorbbecsdk)
- [Piper SDK 文档](https://github.com/agilexrobotics/piper_sdk)
