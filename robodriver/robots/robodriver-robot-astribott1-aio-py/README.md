# RoboDriver-Robot-AstribotT1-AIO-Py

AstriBot T1（星动纪元）通过厂商 Python SDK `astribot_sdk` 接入 RoboDriver，操作类型 `aio`（机器人自带遥操作，采集 desired 关节角作为 action）。

- `--robot.type=astribott1_aio_py`
- 接入类型：`py`，**不需要 ROS**（SDK 走 zenoh，仅激光雷达才依赖 `astribot_msgs`）

## 环境准备

### 1. 安装厂商 SDK

SDK 不在 PyPI 上，用厂商 wheel（本机已存在 `~/astribot_examples/astribot_sdk-v3.1.2-cp310-linux_x86_64.zip`）。
装进 **RoboDriver 的环境**，并用 `--no-deps` 保住 RoboDriver 的 numpy：

```bash
cd ~/astribot_examples && unzip -o astribot_sdk-v3.1.2-cp310-linux_x86_64.zip -d astribot_wheel
source ~/RoboDriver/.venv/bin/activate
pip install --no-deps --find-links=astribot_wheel astribot_sdk astribot_link astribot_deps
pip install "eclipse-zenoh==1.10.1"          # astribot_link 的运行依赖，厂商 wheel 未打包
```

> **必须用 `--no-deps`**：厂商 `INSTALL.txt` 里的 `pip install ./*.whl` 会把 numpy 降到 1.22.4，
> 与 RoboDriver 的 numpy 2.x / opencv 冲突。
>
> 已实测 numpy 2.2.6 + opencv 4.12 下 `astribot_sdk` / `astribot_link` / `astribot_deps` /
> `camera_client` 均可正常导入，但这属于厂商未验证组合；若相机取帧时出现 ABI 报错，见文末「退路」。

### 2. 安装本适配包

```bash
cd ~/RoboDriver/robodriver/robots/robodriver-robot-astribott1-aio-py
pip install --no-deps -e .
```

### 3. 启动前准备

机器人需与本机同网段、supervisor 在线。若还要用激光雷达，先 source ROS：

```bash
source /opt/ros/humble/setup.bash
source ~/astribot_sdk/ros-msgs-workspace/install/setup.bash
source ~/RoboDriver/.venv/bin/activate
```

## 先做方案 A 判定（只读，不下发任何动作）

```bash
source ~/RoboDriver/.venv/bin/activate
python -m robodriver_robot_astribott1_aio_py.check_env
```

它会打印 numpy/cv2/SDK 版本、各部件真实自由度、当前与 desired 关节角、可用相机名与每帧的
dtype/shape/像素范围，并给出「方案 A 可用 / 建议改用方案 B」的结论。
出现 `_ARRAY_API not found`、关节返回 NaN、或某路相机整帧同值，就说明单环境不合适，走文末「退路」。

也可以再用厂商例子交叉确认：

```bash
source ~/astribot_examples/astribot_env.sh
python ~/astribot_examples/100-get_robot_properties.py   # 部件名与自由度
python ~/astribot_examples/301-get_images.py             # 相机名与分辨率
```

> SDK 已装进 RoboDriver 环境后，上面两个脚本也可以直接用 `~/RoboDriver/.venv/bin/python` 跑，
> 不必切到厂商 venv（只有激光雷达相关例子需要额外 source ROS）。

把这两个脚本输出的**相机名**填进 `camera_name_map`，**分辨率**填进 `cameras`。

## 启动

```bash
source ~/RoboDriver/.venv/bin/activate
robodriver-run --robot.type=astribott1_aio_py
```

验证：`curl http://localhost:8088/api/stream_info` 应列出三路相机。

## 配置项

| 字段 | 默认值 | 说明 |
|---|---|---|
| `ip` | `None` | 机器人地址，留空由 SDK 自动发现 |
| `freq` | `125.0` | SDK 控制频率（Hz） |
| `remote_wbc` | `True` | 是否启用远端 WBC |
| `record_parts` | 双臂 + 双夹爪 | 记录哪些部件，可选 `head`/`torso`/`chassis`；自由度在 connect 时由机器人上报 |
| `move_to_home_on_connect` | `False` | 连接时是否先回 home（真机上会动，谨慎开） |
| `camera_name_map` | `image_head` 等三路 | 数据集图像 key → AstriBot 相机名；key 必须含 `image`，否则平台不识别为视频流 |
| `cameras` | 1280x720 / 640x480 | 只用于声明特征与提供视频流 id（`index_or_path` 即流号），不会打开 `/dev/video` |

数据字段命名：`follower_<部件>_joint<N>.pos`（观测）、`leader_<部件>_joint<N>.pos`（动作），单位弧度。

## 真机实测结果（Astribot T1，SDK 3.1.2，RoboDriver 单环境）

| 项目 | 实测值 |
|---|---|
| 部件自由度 | `arm_left` 7、`arm_right` 7、`effector_left` 1、`effector_right` 1、`head` 2、`torso` 4、`chassis` 3 |
| 相机 | `UDP_Camera` 头部**左右目并排立体图**（2560x720 = 2×1280x720）、`left_D405`/`right_D405` 双腕 RealSense D405 640x360、`Anjia` 640x360 |
| 相机数据 | `uint8`，SDK 返回 **BGR**（已在 `get_observation()` 内转成 RGB） |
| `Anjia` | 正常工作。**必须先 `register_image_callback` 订阅**，否则 `get_all_color()` 只返回占位零帧（D405 与 UDP_Camera 由 SDK 默认订阅，Anjia 不是）；本包在 `connect()` 里已为每路相机注册订阅 |
| 夹爪读数 | 实测约 10~12，限位 `0 ~ 100`；量纲存在分歧（见下） |
| 连接耗时 | 相机服务冷启动约 15 秒，热启动约 1 秒 |

### 各部件的物理含义（据实测限位判断）

| 部件 | 值 | 说明 |
|---|---|---|
| `chassis` | 3，限位 ±99999999 | **不是关节角**，是全向底盘的平面位姿/速度 `x, y, yaw`（速度上限实测 1.0/1.0/2.0，即 m/s、m/s、rad/s） |
| `torso` | 4，限位 `[-0.002,0.805] [-2.316,0.001] [-0.049,2.604] [-1.627,1.645]` rad | 腰部 4 个**转动**关节：1 号俯仰(弯腰 0~46°)、2/3 号两级回转(串联合计约 -136°~149°)、4 号侧倾(±93°)；每关节 200 N·m，速度上限 1.5/2/2/3 rad/s |
| `head` | 2，限位 `[-1.61,1.6]` `[-0.58,0.53]` rad | 头部回转（yaw）+ 俯仰（pitch） |
| `effector_*` | 1，限位 `0 ~ 100` | 夹爪开合行程 |

### ⚠️ 夹爪量纲必须先确认

厂商规格写「行程 0~100**mm**」，但你反馈读数 10~12 的单位是 **cm**。两者矛盾：
若单位是 cm，`0~100` 就是 0~100cm（1 米），远超 100mm 行程；若单位是 mm，12 表示张开 12mm。
**在确认之前不要做归一化**，否则数据集里夹爪的物理含义是错的。适配包目前原样记录 SDK 读数，
不做任何缩放，确认后可在 `get_observation()` 里统一换算。

## 已知问题

**进程退出时返回 134（SIGABRT）**。已定位在厂商 SDK 的收尾阶段：本包 `disconnect()` 会把
相机与 Astribot 都显式关掉（日志能看到 `Meta 'camera' deactivated` → `Astribot Offline`），
之后厂商 SDK 自身的 native 析构仍会 abort；厂商自带例子 `301-get_images.py` 同样如此。
发生在所有数据落盘、与平台正常断开之后，不影响已采集数据，但退出码非 0，
建议向 Astribot 反馈。因此本包必须保留 `atexit` 收尾，否则机器人端的相机服务不会被释放。

## 联调时需要确认的事项

1. **夹爪量纲**（mm 还是 cm，见上）。
2. **`Anjia` 是哪一路设备**：画面正常（实测均值 107~129），但 SDK 元数据只有 `activate: true`，查不到厂商/型号；名字来自机器人端相机服务的自定义配置。
3. **首帧可能为全零**：刚订阅完的头 1~2 帧是占位零帧，之后才有内容；对时间敏感的统计要丢掉第一帧。
3. **`chassis` 记录方式**：它是位姿/速度而非关节角，若只要末端与双臂数据，把它从 `record_parts` 去掉更干净。

## 已知限制

- AIO 模式下 RoboDriver 默认**不下发 action**（仅在 disturbance 激活时下发），`send_action()` 已实现但正常采集不会调用。
- 未实现 `status.py`，平台侧设备状态使用默认值；需要上报在线/故障信息时再补。
- 底盘 `chassis` 与全身 `torso/head` 默认不记录，全记录会让数据集维度显著变大。

## 退路：双环境方案

若 numpy 2 与厂商 wheel 冲突，改为 RoboDriver 侧写 `astribott1-aio-dora`、硬件节点单独跑在
`~/.venvs/astribot-3.1.2` 里，两者用 dora 话题交换关节与图像——结构与仓库内的
`robodriver-robot-so101-aio-dora` 及 Genesis 仿真完全一致。

## Cite

```bibtex
@misc{RoboDriver,
  author = {RoboDriver Authors},
  title = {RoboDriver: A robot control and data acquisition framework},
  year = {2025},
  url = {https://github.com/FlagOpen/RoboDriver}
}
```
