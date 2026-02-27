# robodriver-robot-unitree-g1-sdk-py

## 1. Get Start / 快速开始

### 1.1 Clone the repository / 克隆仓库

```bash
git clone --recurse-submodules https://github.com/wangqi951002/robodriver-plugin.git
cd robodriver-plugin/robodriver-robot-unitree-g1-sdk-py
```

### 1.2 Install the project to RoboDriver / 安装到 RoboDriver 环境

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

### 2.2 Prepare unitree-sdk2py environment / 准备 unitree-sdk2py 环境

```bash
https://github.com/unitreerobotics/unitree_sdk2_python 
```


在unitree-sdk2py 环境运行sdk_to_zmq.py

```bash
python sdk_to_zmq.py
```

### 2.4 Launch RoboXStudio / 启动 RoboXStudio

In your RoboDriver project directory:  
在你的 RoboDriver 工程目录中：

```bash
cd /path/to/your/RoboDriver
python -m robodriver.scripts.run --robot.type=unitree_g1_sdk_py
```


## 3. Notes / 注意事项

- 运行sdk_to_zmq.py前，需要将机器人搭载的 realsense D435i相机视频流自行转成ws流，也可以修改sdk_to_zmq.py，选择其他视频流接入方式。
