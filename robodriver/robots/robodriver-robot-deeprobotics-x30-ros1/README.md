# robodriver-robot-deeprobotics-x30-ros1

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

### 2.2 在云深处X30机器狗开发板上运行ros1_to_zmq.py

```bash
python sdk_to_zmq.py
```

### 2.4 Launch RoboXStudio / 启动 RoboXStudio

In your RoboDriver project directory:  
在你的 RoboDriver 工程目录中：

```bash
cd /path/to/your/RoboDriver
python -m robodriver.scripts.run --robot.type=deeprobotics_x30_ros1
```
