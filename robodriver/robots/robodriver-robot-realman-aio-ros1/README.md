# robodriver-robot-realman-aio-ros1

## 1. Get Start / 快速开始

### 1.1 Clone the repository / 克隆仓库

```bash
git clone https://github.com/FlagOpen/RoboDriver.git
```

### 1.2 Install the project to RoboDriver / 安装到 RoboDriver 环境

Make sure you are in the `robodriver` conda env.  
确保你已经进入 `robodriver` 的 conda 环境：

```bash
conda create -n robodriver python=3.10
conda activate robodriver
pip install -e .
```

## 2. Start Collecting / 开始采集数据

### 2.1 Activate environment / 激活环境

```bash
conda activate robodriver
```

### 2.2 Prepare ROS1 environment / 准备 ROS1 环境

Make sure ROS1 (e.g. `noetic`) is installed and sourced.  
确保已经安装并正确配置 ROS1（例如 `noetic`）：

```bash
source /opt/ros/noetic/setup.bash
```

> 如果你的 ROS 发行版不是 `noetic`，请根据实际安装路径进行修改。

### 2.3 Start dataflow (ROS1 → ZMQ bridge) / 启动 ROS1 → ZMQ 数据流

In this repository directory:  
在本仓库目录中运行：

```bash
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman-aio-ros1
python -m robodriver_robot_realman_aio_ros1.ros1_zmq_bridge
```

### 2.4 Launch RoboXStudio / 启动 RoboXStudio

In your RoboDriver project directory:  
在你的 RoboDriver 工程目录中：

```bash
cd /path/to/your/RoboDriver
conda activate robodriver
python robodriver/scripts/run.py   --robot.type=realman_aio_ros1
```


## 3. Notes / 注意事项

- Make sure the ROS1 bridge (`ros1_zmq_bridge.py`) is running **before** starting RoboDriver.  
  启动 RoboDriver 前，请确保 ROS1 → ZMQ 桥接脚本已经在运行。  

- If you encounter OpenCV `cvShowImage` related errors when launching RoboXStudio,  
  you can comment out `cv2.imshow` / `cv2.waitKey` in `robodriver/scripts/run.py`.  
  如果在启动 RoboXStudio 时遇到 OpenCV `cvShowImage` 的报错，可以在  
  `robodriver/scripts/run.py` 中注释掉 `cv2.imshow` 和 `cv2.waitKey` 调用。