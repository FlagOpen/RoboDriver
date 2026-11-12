# AstriBot T1 AIO（原装夹爪）

本目录只提供 `astribott1_aio_py`：通过 ROS2 读取 AstriBot T1 关节状态/遥操指令，通过厂商 `astribot_sdk` 读取相机及执行回放。默认包含双臂、左右原装 1-DoF 夹爪、头部、躯干与底盘。

**Sharpa Wave 灵巧手版本位于独立目录：**[robodriver-robot-astribott1-aio-sharpa-py](../robodriver-robot-astribott1-aio-sharpa-py/README.md)。该类型不记录原夹爪；两个版本可以同时安装，但启动时只选择一个 robot type。

## 环境与安装

- 使用厂商提供的 Python 3.10 `astribot_sdk` wheel 和配套的 `astribot_msgs` ROS2 消息；厂商 SDK 不在 PyPI 上。
- 机器人与主机需同网段，机器人 supervisor 和相机服务在线。
- 启动前 source ROS2 和 AstriBot 的 ROS 消息工作空间，确保当前 Python 环境能导入 `astribot_sdk`、`rclpy`、`astribot_msgs`。

厂商 SDK 不在 PyPI：获取适用于 Python 3.10 的 `astribot_sdk`、`astribot_link`、`astribot_deps` wheel 后安装进 **RoboDriver 的环境**。安装厂商 wheel 时使用 `--no-deps`，避免将 RoboDriver 的 numpy 降级；`astribot_link` 的运行依赖 `eclipse-zenoh==1.10.1` 也需安装。例如将 wheel 解压在 `~/astribot_examples/astribot_wheel` 后：

```bash
source /home/baai/RoboDriver/.venv/bin/activate
uv pip install --python /home/baai/RoboDriver/.venv/bin/python --no-deps --find-links /home/baai/astribot_examples/astribot_wheel astribot_sdk astribot_link astribot_deps
uv pip install --python /home/baai/RoboDriver/.venv/bin/python 'eclipse-zenoh==1.10.1'
```

wheel 的实际解压路径请按你的厂商安装包调整；如果已安装可跳过此步骤。

```bash
source /opt/ros/humble/setup.bash
source ~/astribot_sdk/ros-msgs-workspace/install/setup.bash
source /home/baai/RoboDriver/.venv/bin/activate
uv pip install --python /home/baai/RoboDriver/.venv/bin/python --no-build-isolation --no-deps -e /home/baai/RoboDriver/robodriver/robots/robodriver-robot-astribott1-aio-py
robodriver-run --robot.type=astribott1_aio_py
```

如果环境里有 pip，也可在本目录运行 `pip install --no-deps -e .`。仅安装本包不会提供 Sharpa 机器人类型。

## 采集内容

- ROS2 观测：`/astribot_<part>/joint_space_states`（BEST_EFFORT）；原夹爪 `effector_left/right` 对应 `/astribot_gripper_left/right/joint_space_states`。
- ROS2 动作：`/astribot_<part>/joint_space_command_forward`（RELIABLE），静止时若没有指令则使用当前观测值回退。
- 输出关节：`follower_<part>_joint<N>.pos` / `leader_<part>_joint<N>.pos`。默认原夹爪各一个关节。
- 相机：头部双目左右拆分、双腕 D405 和 Anjia；SDK BGR 帧转换为 RGB，视频流 id 由 `config.py` 的相机配置声明。
- `node.py` 固定 ROS2 话题映射；`config.py` 可选择 `record_parts`、`ros_domain_id`、相机及控制权设置。AIO 采集只读；仅在回放/显式扰动时下发动作，强制接管控制权前需确认现场安全。

只读验证 ROS2 话题可运行：

```bash
ros2 topic hz /astribot_arm_left/joint_space_states
```

当前包没有 `check_env*.py` 自检脚本。手部灵巧手安装、触觉和启动选项见独立包 README。
