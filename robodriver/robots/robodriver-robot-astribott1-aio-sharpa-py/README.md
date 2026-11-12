# AstriBot T1 + Sharpa Wave

独立的 RoboDriver 机器人适配包；复用基础 AstriBot T1 的 ROS2 关节、SDK 相机和控制层，增加 Sharpa Wave 灵巧手。**不采集 AstriBot 原装 1-DoF 夹爪**，仅采集身体关节、Sharpa 手关节和相机（可选触觉）。

## 安装

先按照 [基础包说明](../robodriver-robot-astribott1-aio-py/README.md) 安装 AstriBot SDK、ROS2 消息和基础包，再在 RoboDriver 的 Python 环境中执行：

```bash
uv pip install --python /home/baai/RoboDriver/.venv/bin/python --no-build-isolation --no-deps -e /home/baai/RoboDriver/robodriver/robots/robodriver-robot-astribott1-aio-sharpa-py
```

也可以使用当前 Python 环境的 `pip install --no-deps -e .`。

Sharpa 厂商 SDK 可装在 `/opt/sharpa-wave-sdk` 或 `~/.local/opt/sharpa-wave-sdk`，或通过 `SHARPA_SDK_ROOT` 指定；连接手需同网段且设备在线。运行前 source ROS2 和 AstriBot 消息环境。

```bash
source /opt/ros/humble/setup.bash
source ~/astribot_sdk/ros-msgs-workspace/install/setup.bash
source /home/baai/RoboDriver/.venv/bin/activate
robodriver-run --robot.type=astribott1_aio_sharpa_py
```

单手：`--robot.sharpa_hands.side=left`（或 `right`）；如只调试手关节，使用 `--robot.sharpa_hands.tactile_enabled=false`。默认双手各 22 DoF，触觉默认开启。组合模式禁止在 `record_parts` 或 `control_parts` 配置原装 `effector_left/right`。

纯 AstriBot（含原装夹爪）请改用 `robodriver-run --robot.type=astribott1_aio_py`。