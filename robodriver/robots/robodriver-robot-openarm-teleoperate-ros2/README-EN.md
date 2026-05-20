# robodriver-robot-galaxealite-aio-ros2
## Quick Start
### Access Requirements
#### 1. Hardware Requirements
Refer to the document: [https://jwolpxeehx.feishu.cn/wiki/LYcNwC2rBirg4Dk9CoScemx3n8f?from=from_copylink](https://jwolpxeehx.feishu.cn/wiki/LYcNwC2rBirg4Dk9CoScemx3n8f?from=from_copylink)

#### 2. Environment & Network Requirements
- Galaxea teleoperation function is normal;
- ROS2 (Humble/Iron version) is installed on the host, which can receive galaxea topic data;
- The host and galaxea main controller are connected to the same local area network (Ethernet connection is recommended);
- Host is connected to the internet and can access the web normally;
- Set ROS Domain ID (example):
  ```bash
  export ROS_DOMAIN_ID=1
  ```

### Prerequisites (Execute first if not completed)
1. Embodied Platform Application: [https://ei2data.baai.ac.cn/home](https://ei2data.baai.ac.cn/home)
2. Edge Service Installation: [https://github.com/FlagOpen/RoboDriver-Server.git](https://github.com/FlagOpen/RoboDriver-Server.git)

### Clone Code Repositories
#### 1. Clone RoboDriver Core Repository (Skip if already cloned)
```bash
git clone https://github.com/FlagOpen/RoboDriver.git
```

#### 2. Enter Galaxea ROS2 folder
```bash
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-galaxealite-aio-ros2
```

### Create Miniconda Virtual Environment
#### 1. Install Miniconda (Execute if not installed, compatible with Linux/macOS)
```bash
# Download installation package
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# Execute installation (follow prompts, recommended to agree to conda init)
bash Miniconda3-latest-Linux-x86_64.sh
# Refresh terminal to activate conda
source ~/.bashrc
```

#### 2. Install Dependencies
```bash
# Create and activate Python 3.10 environment (run if environment not created)
conda create -n robodriver python=3.10 -y
conda activate robodriver

# Install RoboDriver core dependencies (run if dependencies not installed)
cd /path/to/your/RoboDriver  
pip install -e .

# Install galaxea robot hardware dependencies
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-galaxealite-aio-ros2
pip install -e .
```

### Configure node.py (Adapt to Robot's Actual Topics)
This script implements core functions such as multi-topic synchronous subscription, data conversion, and command publishing for the GALAXEALITE robot. You need to modify the following configurations according to the actual topic names of the robot (core location: node initialization function `__init__`).

#### 1. Publisher Topics (Motion Command Output)
| Publisher Variable       | Default Topic                              | Function                  |
|--------------------------|--------------------------------------------|---------------------------|
| `publisher_left_arm`     | `/motion_target/target_joint_state_arm_left` | Publish left arm joint target values |
| `publisher_right_arm`    | `/motion_target/target_joint_state_arm_right` | Publish right arm joint target values |
| `publisher_left_gripper` | `/motion_target/target_position_gripper_left` | Publish left gripper position target values |
| `publisher_right_gripper`| `/motion_target/target_position_gripper_right` | Publish right gripper position target values |
| `publisher_state_torso`  | `/motion_target/target_joint_state_torso` | Publish torso joint target values |

#### 2. Follow Feedback Subscription Topics (Robot State Input)
| Subscriber Variable     | Default Topic                | Function                  |
|-------------------------|------------------------------|---------------------------|
| `sub_arm_left`          | `/hdas/feedback_arm_left`    | Subscribe to left arm joint feedback |
| `sub_arm_right`         | `/hdas/feedback_arm_right`   | Subscribe to right arm joint feedback |
| `sub_gripper_left`      | `/hdas/feedback_gripper_left` | Subscribe to left gripper feedback |
| `sub_gripper_right`     | `/hdas/feedback_gripper_right` | Subscribe to right gripper feedback |
| `sub_torso`             | `/hdas/feedback_torso`       | Subscribe to torso joint feedback |

#### 3. Main Motion Command Subscription Topics (Target Command Input)
| Subscriber Variable     | Default Topic                              | Function                  |
|-------------------------|--------------------------------------------|---------------------------|
| `sub_joint_left`        | `/motion_target/target_joint_state_arm_left` | Subscribe to left arm joint target values |
| `sub_joint_right`       | `/motion_target/target_joint_state_arm_right` | Subscribe to right arm joint target values |
| `sub_joint_torso`       | `/motion_target/target_joint_state_torso` | Subscribe to torso joint target values |
| `sub_pose_left`         | `/motion_target/target_pose_arm_left` | Subscribe to left arm pose target values |
| `sub_pose_right`        | `/motion_target/target_pose_arm_right` | Subscribe to right arm pose target values |
| `sub_torso`             | `/motion_target/target_pose_torso` | Subscribe to torso pose target values |
| `sub_gripper_left`      | `/motion_target/target_position_gripper_left` | Subscribe to left gripper position target values |
| `sub_gripper_right`     | `/motion_target/target_position_gripper_right` | Subscribe to right gripper position target values |

#### 4. Image Topic Subscription (Camera Data Input)
| Subscriber Variable       | Default Topic                                      | Function                  |
|---------------------------|----------------------------------------------------|---------------------------|
| `sub_camera_top_left`     | `/hdas/camera_head/left_raw/image_raw_color/compressed` | Subscribe to top-left camera image |
| `sub_camera_top_right`    | `/hdas/camera_head/right_raw/image_raw_color/compressed` | Subscribe to top-right camera image |
| `sub_camera_wrist_left`   | `/hdas/camera_wrist_left/color/image_raw/compressed` | Subscribe to left wrist camera image |
| `sub_camera_wrist_right`  | `/hdas/camera_wrist_right/color/image_raw/compressed` | Subscribe to right wrist camera image |

**Modification Example**: Change the top-left camera subscription topic to a custom path
```python
sub_camera_top_left = Subscriber(self, CompressedImage, '/my_robot/camera/top_left/compressed')
```

#### 5. Key Parameter Adjustments
##### (1) QoS Configuration (Network Transmission Strategy)
```python
# Reliable transmission (default for publishers/critical feedback)
self.qos = QoSProfile(
    durability=DurabilityPolicy.VOLATILE,  # Non-persistent
    reliability=ReliabilityPolicy.RELIABLE, # Ensure message delivery
    history=HistoryPolicy.KEEP_LAST,        # Keep last N messages
    depth=10                                # Queue depth
)

# Best-effort transmission (non-critical commands, prioritize speed)
self.qos_best_effort = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    depth=10
)
```

##### (2) Multi-Topic Synchronization Parameters
```python
self.sync = ApproximateTimeSynchronizer(
    [sub_arm_left, sub_arm_right, sub_gripper_left, sub_gripper_right, sub_torso],
    queue_size=10,  # Larger queue = higher fault tolerance, more memory usage
    slop=0.1        # Time tolerance (seconds): maximum allowed timestamp difference between topics
)
```

##### (3) Publish Frequency Limit
```python
# Default 30Hz, example to modify to 10Hz
self.min_interval_ns = 1e9 / 10
```

### Configure config.py (Hardware Collection Template)
This script defines the robot hardware configuration based on the `lerobot` framework. Adjust according to actual collection requirements (e.g., joints, cameras, recording mode).

#### 1. Motor Configuration Modification Example (Add 4th Torso Joint)
```python
follower_motors: Dict[str, Motor] = field(
    default_factory=lambda norm_mode_body=norm_mode_body: {
        "follower_arms":{
            # Original configuration
            "torso_joint_3":Motor(17, "sts3215", norm_mode_body),
            # Add new joint
            "torso_joint_4":Motor(18, "sts3215", norm_mode_body),
        }
    }
)
```

#### 2. Video Recording Switch (use_videos)
| Configuration Value | Behavior Description                                                                 | Additional Operations                                                                 |
|---------------------|--------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------|
| `False`             | Images are encoded into videos after collection (delayed encoding)                    | Need to modify `lerobot` source code:<br>Edit `/path/to/miniconda3/envs/robodriver/lib/python3.10/site-packages/lerobot/datasets/pipeline_features.py`<br>Comment out:<br>`# if is_image and not use_videos:`<br>`#     continue` |
| `True`              | Images are encoded into videos in real-time during collection (encoding time depends on host performance) | No need to modify source code, use directly |

## Start Data Collection
### 1. Activate Environment and Start Basic Services
```bash
# Start nginx service
sudo systemctl start nginx
# Enter RoboDriver directory and activate environment
cd /path/to/your/RoboDriver
conda activate robodriver
```

### 2. Start Galaxealite Topics
Execute the robot's built-in startup script to ensure ROS2 topics are published normally.

### 3. Start RoboDriver
```bash
python -m robodriver.scripts.run  --robot.type=galaxealite-aio-ros2 
```

### 4. Publish Tasks and Start Collection
1. Enter the Embodied Integrated Platform to publish tasks: [https://ei2rmd.baai.ac.cn/userlogin](https://ei2rmd.baai.ac.cn/userlogin)
2. Enter the collection platform to start collection: [http://localhost:5805/hmi/](http://localhost:5805/hmi/)

### 5. Playback Collected Data
After collection is completed, disconnect teleoperation and click the "Playback" button on the collection platform.

## Common Issue Fixes
1. **Video playback fails to re-run**  
   Edit `RoboDriver/robodriver/core/coordinator.py`, change:
   ```python
   visual_worker(mode="distant")
   ```
   to:
   ```python
   visual_worker(mode="local")
   ```

2. **OpenCV cvShowImage error on startup or need to remove image popups on startup**  
   Comment out the following two lines in `robodriver/scripts/run.py`:
   ```python
   # cv2.imshow(key, img)
   # cv2.waitKey(1)
   ```

3. **Cannot access http://localhost:5805/hmi/**  
   Restart the nginx service:
   ```bash
   sudo systemctl restart nginx
   ```

4. **Robot response timeout**  
   Check the network connectivity between the host and galaxea main controller, and verify that the RoboDriver startup script parameters are correct.

## Data Description
### Storage Path
Collected data is stored in the `/home/yourname/DoRobot` directory by default, organized into folders named by task.

### Directory Structure
```
TaskName_TaskId/
├── audio/          # Audio data
│   └── chunk-000/
│       ├── observation.audio.audio_left/  # Left channel audio (WAV format)
│       └── observation.audio.audio_right/ # Right channel audio (WAV format)
├── data/           # Motion command/feedback data (Parquet format)
│   └── chunk-000/
├── depth/          # Depth images (AVI format)
│   └── chunk-000/
│       ├── observation.images.image_depth_right/
│       └── observation.images.image_depth_top/
├── device/         # Device information
│   └── device_info.json
├── label/          # Annotation data (generated after annotation)
│   └── data_annotation.json
├── meta/           # Metadata
│   ├── common_record.json    # Collection task information
│   ├── episodes.jsonl        # Task description and frame length
│   ├── episodes_stats.jsonl  # Normalized statistical information
│   ├── info.json             # Feature schema, frame rate, version
│   ├── op_dataid.jsonl       # Device number
│   └── tasks.jsonl
└── videos/         # Visible light images (MP4 format)
    └── chunk-000/
        ├── observation.images.image_left/        # Left camera
        ├── observation.images.image_left_tac_l/  # Left tactile left camera
        ├── observation.images.image_left_tac_r/  # Left tactile right camera
        ├── observation.images.image_right/       # Right camera
        ├── observation.images.image_right_tac_l/ # Right tactile left camera
        ├── observation.images.image_right_tac_r/ # Right tactile right camera
        └── observation.images.image_top/         # Top camera
```

## Acknowledgment

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).

## Cite
