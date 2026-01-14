# RoboDriver-Robot-A2D-AIO-ROS2
[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README_en.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README.md)
## 环境部署

### 系统要求

OS：Ubuntu 22.04

硬件： 
- CPU：x86 Arch
- GPU：根据客户需求自行选择

软件:
- miniconda
- Python 3.10
- ROS2 Humble

### PC端首次预配置
1. 使用前将机器人背面的 Debug 网口与外部 PC 组成局域网。\
   PC 通过网线直接连接 G1接口面板的 Debug 网口。

   ```
   G1 控制器网络配置信息：
   • IP：10.42.0.101
   • 子网掩码：255.255.255.0

   PC 网络配置：
   • 固定 IP 方式
   • IP：10.42.0.xxx（10.42.0.10~10.42.0.99）
   • 子网掩码：255.255.255.0
   • 网关：（可选）
   • 网络带宽：1000Mbps
   ```

   配置完成后，验证 PC 和 G1 可以使用网络互相访问。
   ```bash
   ping 10.42.0.101
   ```
2. 在合适的位置打开一个新的终端

   用`miniconda`创建一个机器人依赖环境（这里以agibot为例）

   [Installing Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

   ```bash
   conda create agibot
   ```
   然后在环境内安装依赖

3. 创建一个`requirements`文档并填入以下内容，然后让pip定位到此文件进行安装
   ``` 
   # requirements.txt
   numpy
   protobuf==3.12.4
   ruckig==0.14.0
   opencv-python==4.10.0.84
   scipy
   zmq==0.0.0
   pyzmq==26.2.0
   matplotlib
   ```
   开始安装：
   ```bash
   sudo apt install iproute2
   pip install -r requirements.txt
   ```
4. 执行以下命令，x86 PC 内部署 GDK 环境。
   ```bash
   curl -sSL http://10.42.0.101:8849/install.sh | bash
   ```
   这会在目录下创建一个`a2d_sdk`的文件夹

   安装完 sdk 后再执行
   ```bash
   cd a2d_sdk
   source env.sh
   ```

现在，我们已经安装好了依赖，并临时激活了环境变量，我们的robodriver就可以导入官方的接口了。

### 部署Roboriver
参考[RoboDriver文档/概览/安装与部署](https://flagopen.github.io/RoboDriver-Doc/docs/overview/installation/) 中的步骤。


## 开始数据采集
1. 进入先前创建的`agibot`环境，激活G1机器人的Copilot模式
   ```bash
   conda activate agibot
   robot-service -s -c ./conf/copilot.pbtxt
   ```
2. 在`Roboriver`目录下打开终端，激活环境
   ```bash
   conda activate robodriver
   ```
3. 确保已经[部署并启动 `RoboDriver-Server`](https://github.com/FlagOpen/RoboDriver-Server)，登陆账号
4. 每次使用时在`robodriver`环境中source一下机器人的`env.sh`，确保官方的库能被导入
   ```bash
   source $Your_Customizable_Directory/a2d_sdk/env.sh
   ```
5. 进入`RoboDriver`的a2d ros2 文件夹
   ```bash
   cd robodriver/robots/robodriver-robot-a2d-aio-ros2
   ```
6. 启动 `Roboriver`
   ```bash
   robodriver-run --robot.type=a2d_aio_ros2
   ```
7. 在`Roboriver-Server`中开始使用。



## 致谢

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to TheRobotStudio 🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100).

## 引用
