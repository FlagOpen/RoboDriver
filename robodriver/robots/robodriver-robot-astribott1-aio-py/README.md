# RoboDriver-Robot-AstribotT1-AIO-Py

AstriBot T1（星动纪元）通过厂商 Python SDK `astribot_sdk` 接入 RoboDriver，操作类型 `aio`（机器人自带遥操作，采集 desired 关节角作为 action）。

- `--robot.type=astribott1_aio_py`：原版，记录 AstriBot 自带的 1-DoF 夹爪
- `--robot.type=astribott1_aio_sharpa_py`：叠加 Sharpa Wave 灵巧手（22 DoF/手），默认不再记录原夹爪
- 接入类型：`py`。关节数据走 ROS2 话题，**需要 ROS**；相机、控制权、`send_action` 走 SDK（zenoh）

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

关节走 ROS2 话题，启动前必须 source ROS 环境，否则 connect 时会因 `rclpy` /
`astribot_msgs` 缺失而失败：

```bash
source /opt/ros/humble/setup.bash
source ~/astribot_sdk/ros-msgs-workspace/install/setup.bash
source ~/RoboDriver/.venv/bin/activate
robodriver-run --robot.type=astribott1_aio_py
```

验证：`curl http://localhost:8088/api/stream_info` 应列出三路相机。

## Sharpa Wave 灵巧手（astribott1_aio_sharpa_py）

灵巧手走**千兆以太网**，与 AstriBot 的 zenoh 通信相互独立；SDK 不通过 pip 安装。

### 1. 安装 Sharpa SDK

```bash
# 推荐：系统级安装（需要 sudo）
sudo dpkg -i sharpa-wave-sdk_5.0.11_amd64.deb
sudo apt-get install -f        # 仅缺依赖时
```

没有 sudo 时可以用户级解包，代码会按 `$SHARPA_SDK_ROOT` → `/opt/sharpa-wave-sdk` → `~/.local/opt/sharpa-wave-sdk` 的顺序查找：

```bash
mkdir -p /tmp/sharpa-sdk ~/.local/opt
dpkg-deb -x sharpa-wave-sdk_5.0.11_amd64.deb /tmp/sharpa-sdk
cp -a /tmp/sharpa-sdk/opt/sharpa-wave-sdk ~/.local/opt/
```

本机已验证 Python 3.10.12 + numpy 2.2.6 下可以导入 `sharpa`，并可在无 sudo 的用户级安装下运行。ARM64 用 release 里的 `.zip`，解包到同一目录结构。安装包声明支持 Python 3.10–3.13。

### 2. 只读自检（先跑这个）

```bash
source ~/RoboDriver/.venv/bin/activate
python -m robodriver_robot_astribott1_aio_py.check_env_sharpa
# 需要读关节/触觉时（仍不下发动作）：
python -m robodriver_robot_astribott1_aio_py.check_env_sharpa --start --tactile
```

它会打印发现到的 SN/左右手/IP/固件、关节数量与角度、手型、触觉支持情况；`--tactile` 还会打印每指一帧的 RAW/DEFORM 等 shape。默认不会设置控制模式，也不会调用 `set_joint_position`。单手时加 `--side left` 或 `--side right`。

### 3. 启动

```bash
# 双手，自动发现
robodriver-run --robot.type=astribott1_aio_sharpa_py

# 单手 / 固定 SN
robodriver-run --robot.type=astribott1_aio_sharpa_py \
  --robot.sharpa_hands.side=left \
  --robot.sharpa_hands.left_sn=HAND-L-0001
```

### 4. 配置

| 字段 | 默认值 | 说明 |
|---|---|---|
| `sharpa_hands.enabled` | `True` | 关闭后不连接手，但保留新类型方便调试 |
| `sharpa_hands.side` | `both` | `left` / `right` / `both` |
| `sharpa_hands.left_sn` / `right_sn` | `None` | 可选，固定设备 SN；不填按心跳里的左右手自动分配 |
| `sharpa_hands.dof` | `22` | 每只手关节数；W02/Mozi 等型号按实际上报修改 |
| `sharpa_hands.speed_coeff` | `0.3` | 速度系数，首次联调建议保持低值 |
| `sharpa_hands.current_coeff` | `0.6` | 电流系数 |
| `sharpa_hands.interpolate` | `True` | 下发关节目标时是否插值 |
| `sharpa_hands.discovery_timeout_s` | `10.0` | 等待 UDP 心跳的秒数 |
| `sharpa_hands.tactile_enabled` | `True` | 双手触觉；每个指尖一路图像 |
| `sharpa_hands.tactile_content` | `auto` | `auto` 优先 DEFORM，其次 DEFORM3_SE/DEFORM3/RAW/DIST_FORCE；`deform`/`raw` 可指定优先源 |
| `sharpa_hands.tactile_fps` | `30.0` | 触觉轮询频率（Hz） |
| `sharpa_hands.tactile_timeout_s` | `3.0` | 连接时等待首帧触觉的超时（秒/手） |
| `sharpa_hands.move_to_zero_on_connect` | `False` | 连接时回零位，真机会动 |
| `sharpa_hands.park_on_disconnect` | `False` | 断开时是否先回零位；默认直接 `stop()`，不额外动作 |

新类型默认把 `record_parts` 设为 `arm_left/arm_right/head/torso/chassis`，不再记录 AstriBot 的 1-DoF `effector_*`；如果现场仍保留夹爪，把它加回 `--robot.record_parts`。

### 5. 数据字段

- 观测：`follower_hand_<left|right>_joint1..N.pos`，单位弧度
- 动作：`leader_hand_<left|right>_joint1..N.pos`，单位弧度
- 触觉：`tactile_<left|right>_<thumb|index|middle|ring|pinky>`，RGB 图像。数据集里会变成 `observation.images.tactile_*`；key 本身不含 `image`，所以不会占用前端视频流。
- 关节顺序与 Sharpa SDK 一致：拇指 5、食指/中指/无名指/小指各 4（具体限位见 URDF）
- 触觉默认源是 `DEFORM`（Wave01 为 240×240 uint8）；`RAW` 为 240×320，SE/Mozi 可能是 60×60 的 `DEFORM3_SE`。实际 shape 以 `check_env_sharpa --tactile` 输出为准。

没有手部 teleop 时，`leader_hand_*` 记录的是**实测位置**（AIO 镜像），不是独立 leader 信号。要做真正的模仿学习 action，应接入 Sharpa 的视觉/Quest/Manus 遥操作，或实现一个 teleoperator 把目标位置交给 `send_action()`。

### 6. 当前限制

- 双手触觉共 10 路图像，30 Hz 下数据量较大；建议 `--robot.use_videos=true`，或降低 `sharpa_hands.tactile_fps`。
- 平台保存时的 `validate_session` 会扫描 `images/tactile_*` 并做帧变化检查；触觉图在无接触时可能静态，可能把 `camera_frame_rate` 标成 no pass。数据会正常保存，需要时再调整校验规则。
- 触觉只写入数据集，不推送到前端视频流；需要实时预览时再扩展 stream_info。
- 双手同时开触觉时，第二只手可能遇到 UDP 端口冲突，本包会自动调用 `retry_tactile_alternate_port()` 切换到备用端口；若仍失败，检查防火墙或 Sharpa Pilot 是否占用。
- 组合机器人连接手时使用 `skip_tactile=False`（默认开触觉）；只调试关节时可设 `sharpa_hands.tactile_enabled=false`。若启动失败，先确认设备在线且没有被其他进程占用。
- 断开顺序是 `wave.stop()` → `wave.destroy()` → `manager.disconnect_all()`；本包已在 `disconnect()` 里处理。`park_on_disconnect` 可让手先回零，默认不回零；进程退出时请走正常退出。
- 平台侧 `reset_pose` 会把非 `gripper` 关节插值到 0；手的 0 位是否安全需要实机确认，首次联调建议不要用 `reset_pose`。

## 配置项

| 字段 | 默认值 | 说明 |
|---|---|---|
| `ip` | `None` | 机器人地址，留空由 SDK 自动发现 |
| `freq` | `125.0` | SDK 控制频率（Hz） |
| `remote_wbc` | `True` | 是否启用远端 WBC |
| `record_parts` | 双臂 + 双夹爪 | 记录哪些部件，可选 `head`/`torso`/`chassis`；自由度在 connect 时由机器人上报 |
| `move_to_home_on_connect` | `False` | 连接时是否先回 home（真机上会动，谨慎开） |
| `camera_name_map` | `image_head_left/right` 等五路 | 数据集图像 key → AstriBot 相机名；key 必须含 `image`，否则平台不识别为视频流 |
| `camera_crops` | 头部左右目各 1280×720 | 从原始帧裁子图 `(x, y, w, h)`；头部双目原图 2560×720，按 1280 宽切成左右两目 |
| `cameras` | 1280x720 / 640x480 | 只用于声明特征与提供视频流 id（`index_or_path` 即流号），不会打开 `/dev/video` |
| `ros_domain_id` | `25` | 实测 T1 关节话题在此 domain；模块会在 `rclpy.init()` 前设置 `ROS_DOMAIN_ID` |
| `ros_obs_suffix` | `joint_space_states` | 观测话题后缀，`/astribot_<部件>/<后缀>`，`RobotJointState.position`，BEST_EFFORT |
| `ros_action_suffix` | `joint_space_command_forward` | 动作话题后缀；静止时该话题无数据，此时 `get_action` 回退到观测值 |
| `ros_wait_first_frame_s` | `10.0` | 连接时等待各部件首帧观测的超时；动作话题静止时无数据，不纳入硬性等待 |

数据字段命名：`follower_<部件>_joint<N>.pos`（观测）、`leader_<部件>_joint<N>.pos`（动作），单位弧度。

## 关节走 ROS2 话题

关节观测与动作一律从 ROS 话题读取，启动前必须 source ROS 环境：

```bash
source /opt/ros/humble/setup.bash
source ~/astribot_sdk/ros-msgs-workspace/install/setup.bash
source ~/RoboDriver/.venv/bin/activate
# 只读自检：验证 domain 25 下各部件 obs/action 话题与自由度
python -m robodriver_robot_astribott1_aio_py.check_env_ros
# 采集
robodriver-run --robot.type=astribott1_aio_py
```

- 观测 ← `/astribot_<部件>/joint_space_states`（`mode=0`），动作 ← `/astribot_<部件>/joint_space_command_forward`，消息类型为 `astribot_msgs/msg/RobotJointState` / `RobotJointController`。
- 实测该消息 `name` 字段恒为空，关节名由本包按「部件 + 序号」生成，命名与 SDK 版一致，数据集 schema 不变。
- `record_parts` 里的 `effector_left/right` 会自动映射到话题名 `astribot_gripper_left/right`。
- **QoS**：`joint_space_states` 是 BEST_EFFORT，本包用 `qos_profile_sensor_data` 订阅；`command_recv` 是 RELIABLE。若换成默认 reliable 订阅 states 会静默收不到。
- 只有关节数据走 ROS；相机、控制权抢占、`send_action` 仍走 SDK。
- `joint_space_command_forward` 只在有外部指令时发布，机器人静止时零帧；此时 `get_action` 回退为观测值（AIO 下 desired≈current）。如需改用 `command_recv`，设 `--robot.ros_action_suffix=joint_space_command_recv`。

## 真机实测结果（Astribot T1，SDK 3.1.2，RoboDriver 单环境）

| 项目 | 实测值 |
|---|---|
| 部件自由度 | `arm_left` 7、`arm_right` 7、`effector_left` 1、`effector_right` 1、`head` 2、`torso` 4、`chassis` 3 |
| 相机 | `UDP_Camera` 头部双目，拆成 `image_head_left` / `image_head_right` 两路 1280×720（原始图为 2560×720 并排）；`left_D405`/`right_D405` 双腕 RealSense D405 640×360；`Anjia` 640×360。拆两路是为了匹配网页 16:9 小窗，避免超宽图被裁 |
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

## 回放 / 外部控制（控制权）

实测机器人所有部件（chassis/torso/双臂/夹爪/head）的控制权由 `vr_teleoperation` 持有，
**priority_level = 0（最高）**。这种状态下外部 `set_joints_position` 会被静默忽略——表现为
"回放启动了但机器人不动"。

默认 `control_mode=takeover`：**只有在真正下发动作（回放 / 扰动）时**才会 `acquire_control`，
正常采集只读不写，不会干扰遥操作。若遥操正在写（租约非空闲）且未开 `force_takeover`，会拒绝抢占并报错。

需要回放/外部控制时：

```bash
robodriver-run --robot.type=astribott1_aio_py \
  --robot.control_mode=takeover \
  --robot.force_takeover=true
```

- `takeover` 会在首次 `send_action` 时调用 `acquire_control(priority=0)` 抢占租约，
  `disconnect` 时自动 `release_control`
- 目标部件仍有**非空闲**租约时默认拒绝抢占，必须显式 `force_takeover=true`
- **会顶掉 VR 遥操作的控制权**，必须现场无人操作、急停在手
- `control_parts` 默认为双臂+双夹爪（不含 chassis），避免回放时底盘开走

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
