# RoboDriver-Robot-A2D-AIO-ROS2
[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README_en.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README.md)
## Environment Deployment

### Requirements

OS：Ubuntu 22.04

Hardware： 
- CPU：x86 Arch
- GPU：Customizable

Software :
- miniconda
- Python 3.10
- ROS2 Humble

### First Pre-Configuration for the PC 
1. Before use, establish a local area network between the robot's rear Debug port and an external PC.\
   Connect the PC directly to the Debug port on the G1 interface panel via an Ethernet cable.

   ```
   G1 Robot Network Configuration:
   • IP: 10.42.0.101
   • Subnet Mask: 255.255.255.0

   PC Network Configuration:
   • Static IP method
   • IP: 10.42.0.xxx (10.42.0.10~10.42.0.99)
   • Subnet mask: 255.255.255.0
   • Gateway: (optional)
   • Network Bandwidth: 1000Mbps
   ```
   After configuration, verify that the PC and G1 can access each other over the network.
   ```bash
   ping 10.42.0.101
   ```
2. Open a new terminal in the appropriate location.

   Create a dependency environment for a robot using `miniconda` (using agibot as an example)

   [Installing Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

   ```bash
   conda create agibot
   ```
   Then install dependencies within the environment

3. Create a `requirements` file and populate it with the following content, then instruct pip to locate this file for installation:
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
   Then begin installation:
   ```bash
   sudo apt install iproute2
   pip install -r requirements.txt
   ```
4. Run the following command to deploy the GDK environment on an x86 PC.
   ```bash
   curl -sSL http://10.42.0.101:8849/install.sh | bash
   ```
   This will create an `a2d_sdk` folder in the directory.

   After installing the SDK, run: 
   ```bash
   cd a2d_sdk
   source env.sh
   ```
Now that we've installed the dependencies and temporarily activated the environment variables, our RoboDriver can import the official interfaces.

### Deploy Roboriver
Refer to the steps in [RoboDriver Documentation/Overview/Installation & Deployment](https://flagopen.github.io/RoboDriver-Doc/docs/overview/installation/).

## Begin Data Collection
1. Enter the previously created `agibot` environment and activate Copilot mode for the G1 robot.
   ```bash
   conda activate agibot
   robot-service -s -c ./conf/copilot.pbtxt
   ```
2. Open a terminal in the `Roboriver` directory and activate the environment.
   ```bash
   conda activate robodriver
   ```
3. [Deploy and start `RoboDriver-Server`](https://github.com/FlagOpen/RoboDriver-Server), log in to your account
4. Each time you use it, source the bot's `env.sh` in the `robodriver` environment to ensure the official libraries can be imported.
   ```bash
   source $Your_Customizable_Directory/a2d_sdk/env.sh
   ```
5. Enter the a2d ros2 folder in `RoboDriver`
   ```bash
   cd robodriver/robots/robodriver-robot-a2d-aio-ros2
   ```
6. Launch `Roboriver`
   ```bash
   robodriver-run --robot.type=a2d_aio_ros2
   ```
7. Getting Started with `Roboriver-Server`.



## Acknowledgment

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to TheRobotStudio 🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100).

## Cite
