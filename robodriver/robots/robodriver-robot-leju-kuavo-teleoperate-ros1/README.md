# robodriver-robot-leju-kuavo-teleoperate-ros1
## 快速开始
### 接入要求
#### 1. 硬件要求
参考文档：[https://jwolpxeehx.feishu.cn/wiki/LYcNwC2rBirg4Dk9CoScemx3n8f?from=from_copylink](https://jwolpxeehx.feishu.cn/wiki/LYcNwC2rBirg4Dk9CoScemx3n8f?from=from_copylink)

乐聚机器人官网：[https://kuavo.lejurobot.com/beta_manual/basic_usage/kuavo-ros-control/docs/1%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/ from=from_copylink](https://kuavo.lejurobot.com/beta_manual/basic_usage/kuavo-ros-control/docs/1%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/)

#### 2. 环境与网络要求
- leju机器人 VR功能正常；
- 主机已安装 ROS1（Humble/Iron 版本），可接收 leju 话题数据；
- 主机通过路由与leju机器人上位机相连；
- 主机连接网络，可以正常上网；
- 在主机终端中输入：
  ```bash
  sudo nano /etc/hosts
  ```
- 在打开的hosts文件中加入:
  ```bash
  127.0.0.1       localhost
  127.0.1.1       robot-System-Product-Name
  192.168.26.1    kuavo_master（加上此行）
  #注意最后一行的ip和名称不能修改，必须保持和leju机器人下位机的网络配置一致
  ```
- 在主机上的bashrc文件加入ROS网络配置：
  ```bash
  export ROS_MASTER_URI=http://kauvo_master:11311
  export ROS_HOSTNAME=192.168.1.110
  退出文件后保存，执行：
  source ~/.bashrc
  ```

### 前置准备（未完成需先执行）
1. 具身平台申请：[https://ei2data.baai.ac.cn/home](https://ei2data.baai.ac.cn/home)
2. 端侧服务安装：[https://github.com/FlagOpen/RoboDriver-Server.git](https://github.com/FlagOpen/RoboDriver-Server.git)
3. 对主机和leju机器人的上下位机进行配网，配网步骤如下：
    - (1) 将主机和leju机器人的上位机连接到同一路由器下，使两者在一个子网内
    - (2) 手动修改主机和上位机的网段及IP地址，如果192.168.1.x网段没有占用则优先使用该网段 
    - (3) 进入主机的网络设置界面，在IPV4设置中选择手动，修改IP地址为192.168.1.110，子网掩码为255.255.255.0，网关为192.168.1.100。设置完成后修改IPV4的路由规则，目的地址为192.168.26.0，子网掩码为255.255.255.0，网关为192.168.1.100
    - (4) 进入leju机器人的上位机，在有线连接界面找到Ethernet(eth2)进行修改，在IPV4界面选择手动，修改IP地址为192.168.1.100，子网掩码为255.255.255.0，网关为192.168.1.100，设置完成后退出保存
    - (5) 在主机的终端中输入ping 192.168.1.100此时应该可以ping通
    - (6) 进入leju机器人的下位机，找到有线连接界面，找到IPV4界面选择手动，在最后的路由规则中新加一条，其中目标地址为192.168.1.0，子网掩码为255.255.255.0，网关为192.168.26.12，保存退出
    - (7) 在主机的终端中检查ip route：
        ```bash
        ip route show
        #应该存在这条路由规则：
        192.168.26.0/24 via 192.168.1.100 dev enp131s0 
        ```
    - (8) 在下位机的终端中检查ip route：
        ```bash
        ip route show
        #应该存在这条路由规则：
        192.168.1.0/24 via 192.168.26.12 dev 端口号
        ```
    - (9) 在主机的终端中ping下位机的IP地址：
        ```bash
        ping 192.168.26.1
        #如果可以ping通则说明配网成功
        ```
4. 同步ROS消息类型:
由于leju的反馈和发布节点使用了自定义消息类型，需要先在ROS中编译生成消息类型后才能正常使用，确保上面网络配置正确，在主机终端中输入：
```bash
conda deactivate 
```
检查python版本：
```bash
which python3
python3 --version #应该输出3.8.x
```
确认输出正确后，切换到系统环境，输入：
```bash
git clone https://gitee.com/leju-robot/kuavo-ros-opensource.git
```
将kuavo的相关自定义内容安装到默认目录
之后在终端中输入：
```bash
rsync -avz lab@192.168.26.1:/home/lab/kuavo-ros-opensource/src/kuavo_msgs/ ~/kuavo-ros-opensource/src/kuavo_msgs/
```
之后输入：
```bash
cd ~/kuavo-ros-opensource
catkin build kuavo_msgs
catkin build kuavo_sdk
```


    

### 克隆代码仓库
#### 1. 克隆 RoboDriver 核心仓库（已克隆可跳过）
```bash
git clone https://github.com/FlagOpen/RoboDriver.git
```

#### 2. 进入 leju kuavo Ros1 文件夹
```bash
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-leju-kuavo-teleoperate-ros1
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

# 安装 leju kuavo 机器人硬件依赖
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-leju-kuavo-teleoperate-ros1
pip install -e .
出现lerobot和torch版本冲突时可以忽略
```

### 配置 node.py（适配机器人实际话题）
该脚本实现 leju 机器人多话题同步订阅、数据转换、指令发布等核心功能，需根据机器人实际话题名称修改以下配置（核心位置：节点初始化函数 `__init__`）。

#### 1. 发布器话题（运动指令输出）
| 发布器变量               | 默认话题                                  | 功能                     |
|--------------------------|-------------------------------------------|--------------------------|
| `arm_pub`     | `/kuavo_arm_traj` | 双臂关节目标值发布       |
| `hand_pub`    | `/control_robot_hand_position` | 双手关节目标值发布      |
| `head_pub` | `/robot_head_motion_data` | 头部关节目标值发布    |


#### 2. 跟随反馈订阅话题（机器人状态输入）
| 订阅器变量       | 默认话题               | 功能                 |
|------------------|------------------------|----------------------|
| `sub_body`   | `/sensors_data_raw` | 躯干关节反馈订阅     |
| `sub_hand`  | `/dexhand/state` | 双手关节反馈订阅    |

#### 3. 图像话题订阅（相机数据输入）
| 订阅器变量           | 默认话题                                      | 功能                 |
|----------------------|-----------------------------------------------|----------------------|
| `sub_camera_top` | `/camera/color/image_raw` | 顶部相机图像订阅   |
| `sub_camera_wrist_left` | `/left_wrist_camera/color/image_raw` | 左手腕相机图像订阅  |
| `sub_camera_wrist_right` | `/right_wrist_camera/color/image_raw` | 右手腕相机图像订阅 |

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
cd ~/kuavo-ros-opensource
source devel/setup.bash
# 进入 RoboDriver 目录并激活环境
cd /path/to/your/RoboDriver
conda activate robodriver

```

### 2. 启动 leju Ros1 话题
乐聚机器人官网：https://kuavo.lejurobot.com/beta_manual/basic_usage/kuavo-ros-control/docs/1%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/%E4%BA%A7%E5%93%81%E4%BB%8B%E7%BB%8D/
按照乐聚官网快速开始页面启动机器人并使能所有关节
之后按照官网说明配置机器人进入VR控制模式，确保机器人的上下位机和VR接入同一个wifi网络
如果上位机网络配置正确，此时机器人上位机IP应该为192.168.1.100，确保已经安装ssh插件，进入主机终端：
```bash
ssh leju_kuavo@192.168.1.100
```
进入上位机终端后执行：
```bash
cd /kuavo_ros_application
source devel/setup.bash
roslaunch dynamic_biped load_robot_head.launch use_obrecc:=true enable_wrist_camera:=true
#在输入完.launch后可以直接按tab补全，其中use_obrecc和enable_wrist_camera参数全部为:=true，启用后会增加相应的图像话题输出，供RoboDriver采集使用
```

### 3. 启动 RoboDriver
```bash
python -m robodriver.scripts.run  --robot.type=leju-kuavo-teleop-ros1 
```
### 4.启动VR头显
按照乐聚官网快速开始页面说明启动VR头显，进入VR控制软件按照操作控制机器人

### 5. 任务发布与采集
1. 进入具身一体化平台发布任务：[https://ei2rmd.baai.ac.cn/userlogin](https://ei2rmd.baai.ac.cn/userlogin)
2. 进入采集平台开始采集：[http://localhost:5805/hmi/](http://localhost:5805/hmi/)

### 6. 回放采集数据
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

   检查主机与 leju 主控的网络连通性、RoboDriver 启动脚本参数是否正确。

5. **VR无法控制机器人**

   检查 VR，上下位机是否在同一wifi网络下

6. **无法查看ros话题或无法连接到ros主机**

   检查 ROS Master URI 和 Hostname 配置 

7. **关节订阅连接超时**

   确认此终端中已经 source 了自定义消息类型的 ROS 工作空间，若没有执行则执行：
    ```bash
    source ~/kuavo-ros-opensource/devel/setup.bash
    ```
8. **rostopic数据异常**

   在rostopic存在一条消息为sensors_data_raw,如果使用`rostopic echo /sensors_data_raw`命令查看数据时发现数据异常或者没有数据显示，则是kuavo_msgs版本的问题，分别在下位机和主机上执行以下命令检查md5版本
   ```bash
   rostopic type /sensors_data_raw | rosmsg md5
   ```
   两者的输出应该一样，如果md5编码不同则重新执行上面的步骤将机器人内的版本同步到主机上

9. **catkin build编译报错**

确保在编译时关闭conda环境，在本机环境中，使用命令：
```bash
conda deactivate
```
确保没有在base或者其他环境中，检查当前的python环境:
```bash
which python3
python3 --version #应该输出3.8.x
```
如果输出的路径中包含conda或者anaconda，则说明当前环境没有正确切换，需要先切换到正确的环境后再进行编译
如果提示缺少依赖，则使用pip安装相关的依赖,例如：
```bash
pip install python3-empy
```

   
   
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
