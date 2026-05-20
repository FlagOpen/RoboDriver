# robodriver-robot-galaxealite-aio-ros2
## 快速开始
### 接入要求
#### 1. 硬件要求
参考文档：[https://jwolpxeehx.feishu.cn/wiki/LYcNwC2rBirg4Dk9CoScemx3n8f?from=from_copylink](https://jwolpxeehx.feishu.cn/wiki/LYcNwC2rBirg4Dk9CoScemx3n8f?from=from_copylink)

#### 2. 环境与网络要求
- galaxea 遥操作功能正常；
- 主机已安装 ROS2（Humble/Iron 版本），可接收 galaxea 话题数据；
- 主机与 galaxea 主控接入同一局域网（推荐网线直连）；
- 主机连接网络，可以正常上网；
- 设置 ROS 域 ID（示例）：
  ```bash
  export ROS_DOMAIN_ID=1
  ```

### 前置准备（未完成需先执行）
1. 具身平台申请：[https://ei2data.baai.ac.cn/home](https://ei2data.baai.ac.cn/home)
2. 端侧服务安装：[https://github.com/FlagOpen/RoboDriver-Server.git](https://github.com/FlagOpen/RoboDriver-Server.git)


### 克隆代码仓库
#### 1. 克隆 RoboDriver 核心仓库（已克隆可跳过）
```bash
git clone https://github.com/FlagOpen/RoboDriver.git
```

#### 2. 进入 galaxea ros2 文件夹
```bash
git /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-galaxealite-aio-ros2
```

### 创建 Miniconda 虚拟环境
#### 1. 安装 Miniconda（未安装时执行，适配 Linux/macOS）
```bash
# 下载安装包
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# 执行安装（按提示操作，建议同意 conda init）
bash Miniconda3-latest-Linux-x86_64.sh
# 刷新终端使 conda 生效
source ~/.bashrc
```

#### 2. 安装依赖
```bash
# 创建并激活 Python 3.10 环境 （未创建环境时运行）
conda create -n robodriver python=3.10 -y
conda activate robodriver

# 安装 RoboDriver 核心依赖 （未安装依赖时运行）
cd /path/to/your/RoboDriver  
pip install -e .

# 安装 galaxea 机器人硬件依赖
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-galaxealite-aio-ros2
pip install -e .
```

### 配置 node.py（适配机器人实际话题）
该脚本实现 GALAXEALITE 机器人多话题同步订阅、数据转换、指令发布等核心功能，需根据机器人实际话题名称修改以下配置（核心位置：节点初始化函数 `__init__`）。

#### 1. 发布器话题（运动指令输出）
| 发布器变量               | 默认话题                                  | 功能                     |
|--------------------------|-------------------------------------------|--------------------------|
| `publisher_left_arm`     | `/motion_target/target_joint_state_arm_left` | 左臂关节目标值发布       |
| `publisher_right_arm`    | `/motion_target/target_joint_state_arm_right` | 右臂关节目标值发布      |
| `publisher_left_gripper` | `/motion_target/target_position_gripper_left` | 左夹爪位置目标值发布    |
| `publisher_right_gripper`| `/motion_target/target_position_gripper_right` | 右夹爪位置目标值发布   |
| `publisher_state_torso`  | `/motion_target/target_joint_state_torso` | 躯干关节目标值发布       |

#### 2. 跟随反馈订阅话题（机器人状态输入）
| 订阅器变量       | 默认话题               | 功能                 |
|------------------|------------------------|----------------------|
| `sub_arm_left`   | `/hdas/feedback_arm_left` | 左臂关节反馈订阅     |
| `sub_arm_right`  | `/hdas/feedback_arm_right` | 右臂关节反馈订阅    |
| `sub_gripper_left` | `/hdas/feedback_gripper_left` | 左夹爪反馈订阅   |
| `sub_gripper_right` | `/hdas/feedback_gripper_right` | 右夹爪反馈订阅 |
| `sub_torso`      | `/hdas/feedback_torso` | 躯干关节反馈订阅     |

#### 3. 主运动指令订阅话题（目标指令输入）
| 订阅器变量         | 默认话题                                  | 功能                     |
|--------------------|-------------------------------------------|--------------------------|
| `sub_joint_left`   | `/motion_target/target_joint_state_arm_left` | 左臂关节目标值订阅       |
| `sub_joint_right`  | `/motion_target/target_joint_state_arm_right` | 右臂关节目标值订阅      |
| `sub_joint_torso`  | `/motion_target/target_joint_state_torso` | 躯干关节目标值订阅       |
| `sub_pose_left`    | `/motion_target/target_pose_arm_left` | 左臂位姿目标值订阅        |
| `sub_pose_right`   | `/motion_target/target_pose_arm_right` | 右臂位姿目标值订阅       |
| `sub_torso`        | `/motion_target/target_pose_torso` | 躯干位姿目标值订阅        |
| `sub_gripper_left` | `/motion_target/target_position_gripper_left` | 左夹爪位置目标值订阅    |
| `sub_gripper_right`| `/motion_target/target_position_gripper_right` | 右夹爪位置目标值订阅   |

#### 4. 图像话题订阅（相机数据输入）
| 订阅器变量           | 默认话题                                      | 功能                 |
|----------------------|-----------------------------------------------|----------------------|
| `sub_camera_top_left` | `/hdas/camera_head/left_raw/image_raw_color/compressed` | 顶部左相机图像订阅   |
| `sub_camera_top_right` | `/hdas/camera_head/right_raw/image_raw_color/compressed` | 顶部右相机图像订阅 |
| `sub_camera_wrist_left` | `/hdas/camera_wrist_left/color/image_raw/compressed` | 左手腕相机图像订阅  |
| `sub_camera_wrist_right` | `/hdas/camera_wrist_right/color/image_raw/compressed` | 右手腕相机图像订阅 |

**修改示例**：将顶部左相机订阅话题改为自定义路径
```python
sub_camera_top_left = Subscriber(self, CompressedImage, '/my_robot/camera/top_left/compressed')
```

#### 5. 关键参数调整
##### （1）QoS 配置（网络传输策略）
```python
# 可靠传输（默认用于发布器/关键反馈）
self.qos = QoSProfile(
    durability=DurabilityPolicy.VOLATILE,  # 不持久化
    reliability=ReliabilityPolicy.RELIABLE, # 确保消息到达
    history=HistoryPolicy.KEEP_LAST,        # 保留最后N条
    depth=10                                # 队列深度
)

# 尽力传输（非关键指令，优先速度）
self.qos_best_effort = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=10
)
```

##### （2）多话题同步参数
```python
self.sync = ApproximateTimeSynchronizer(
    [sub_arm_left, sub_arm_right, sub_gripper_left, sub_gripper_right, sub_torso],
    queue_size=10,  # 队列越大容错越高，内存占用越多
    slop=0.1        # 时间容差（秒）：允许话题时间戳最大差值
)
```

##### （3）发布频率限制
```python
# 默认 30Hz，修改为 10Hz 示例
self.min_interval_ns = 1e9 / 10
```

### 配置 config.py（硬件采集模板）
该脚本基于 `lerobot` 框架定义机器人硬件配置，需根据实际采集需求调整（如关节、相机、录制模式）。

#### 1. 电机配置修改示例（新增躯干第4关节）
```python
follower_motors: Dict[str, Motor] = field(
    default_factory=lambda norm_mode_body=norm_mode_body: {
        "follower_arms":{
            # 原有配置
            "torso_joint_3":Motor(17, "sts3215", norm_mode_body),
            # 新增关节
            "torso_joint_4":Motor(18, "sts3215", norm_mode_body),
        }
    }
)
```

#### 2. 视频录制开关（use_videos）
| 配置值 | 行为说明                                                                 | 额外操作                                                                 |
|--------|--------------------------------------------------------------------------|--------------------------------------------------------------------------|
| `False`| 图像采集后滞后编码为视频                                                 | 需修改 `lerobot` 源码：<br>编辑 `/path/to/miniconda3/envs/robodriver/lib/python3.10/site-packages/lerobot/datasets/pipeline_features.py`<br>注释：<br>`# if is_image and not use_videos:`<br>`#     continue` |
| `True` | 图像采集时实时编码为视频（编码时长依赖主机性能）| 无需修改源码，直接使用                                                   |

## 开始采集数据
### 1. 激活环境并启动基础服务
```bash
# 启动 nginx 服务
sudo systemctl start nginx
# 进入 RoboDriver 目录并激活环境
cd /path/to/your/RoboDriver
conda activate robodriver
```

### 2. 启动 Galaxealite 话题
执行机器人自带的启动脚本，确保 ROS2 话题正常发布。

### 3. 启动 RoboDriver
```bash
python -m robodriver.scripts.run  --robot.type=galaxealite-aio-ros2 
```

### 4. 任务发布与采集
1. 进入具身一体化平台发布任务：[https://ei2rmd.baai.ac.cn/userlogin](https://ei2rmd.baai.ac.cn/userlogin)
2. 进入采集平台开始采集：[http://localhost:5805/hmi/](http://localhost:5805/hmi/)

### 5. 回放采集数据
采集完成后断开遥操作，在采集平台点击「回放」按钮即可。

## 常见问题修复
1. **视频回放重新运行失败**  
   编辑 `RoboDriver/robodriver/core/coordinator.py`，将：
   ```python
   visual_worker(mode="distant")
   ```
   修改为：
   ```python
   visual_worker(mode="local")
   ```

2. **启动时 OpenCV cvShowImage 错误 或者 需要删除启动时的图像弹窗**  
   注释 `robodriver/scripts/run.py` 中以下两行：
   ```python
   # cv2.imshow(key, img)
   # cv2.waitKey(1)
   ```

3. **无法访问 http://localhost:5805/hmi/**  
   重新启动 nginx 服务：
   ```bash
   sudo systemctl restart nginx
   ```

4. **机器人响应超时**  
   检查主机与 galaxea 主控的网络连通性、RoboDriver 启动脚本参数是否正确。

## 数据说明
### 存储路径
采集数据默认存储于 `/home/yourname/DoRobot` 目录，按任务命名划分文件夹。

### 目录结构
```
TaskName_TaskId/
├── audio/          # 音频数据
│   └── chunk-000/
│       ├── observation.audio.audio_left/  # 左声道音频（WAV格式）
│       └── observation.audio.audio_right/ # 右声道音频（WAV格式）
├── data/           # 运动指令/反馈数据（Parquet格式）
│   └── chunk-000/
├── depth/          # 深度图像（AVI格式）
│   └── chunk-000/
│       ├── observation.images.image_depth_right/
│       └── observation.images.image_depth_top/
├── device/         # 设备信息
│   └── device_info.json
├── label/          # 标注数据（标注后生成）
│   └── data_annotation.json
├── meta/           # 元数据
│   ├── common_record.json    # 采集任务信息
│   ├── episodes.jsonl        # 任务描述与帧长
│   ├── episodes_stats.jsonl  # 归一化统计信息
│   ├── info.json             # 特征schema、帧率、版本
│   ├── op_dataid.jsonl       # 设备编号
│   └── tasks.jsonl
└── videos/         # 可见光图像（MP4格式）
    └── chunk-000/
        ├── observation.images.image_left/        # 左侧相机
        ├── observation.images.image_left_tac_l/  # 左侧触觉左相机
        ├── observation.images.image_left_tac_r/  # 左侧触觉右相机
        ├── observation.images.image_right/       # 右侧相机
        ├── observation.images.image_right_tac_l/ # 右侧触觉左相机
        ├── observation.images.image_right_tac_r/ # 右侧触觉右相机
        └── observation.images.image_top/         # 顶部相机
```
