# robodriver-robot-dobot-nova2-ros2

## 1. Get Start / 快速开始

### 1.1 Install the project to RoboDriver / 安装到 RoboDriver 环境

Make sure you are in the `robodriver` conda env.  
确保你已经进入 `robodriver` 的 conda 环境：

```bash
conda activate robodriver
pip install -e .
```

## 2. Start Collecting / 开始采集数据

### 2.1 Activate environment / 激活环境

```bash
conda activate robodriver
```
### 2.2 启动realsense相机ros-sdk
```bash
https://github.com/IntelRealSense/realsense-ros
```

### 2.3 启动越疆nova2遥操作程序
#### 项目名称：dobot_xtrainer（官方申请dobot_xtrainer开源版本）

```bash
https://www.dobot.cn/service/download-center?keyword=xtrainer
```

### 2.4 Launch RoboXStudio / 启动 RoboXStudio

In your RoboDriver project directory:  
在你的 RoboDriver 工程目录中：

```bash
cd /path/to/your/RoboDriver
python -m robodriver.scripts.run --robot.type=dobot-nova2-ros2
```
## 3. Notes / 注意事项

- dobot_xtrainer不支持ros2，需将采集数据自行发布成ros2 topic，代码位置：experiments/run_control.py。

