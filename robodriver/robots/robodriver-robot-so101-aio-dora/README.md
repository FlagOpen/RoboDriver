# RoboDriver-Robot-SO101-AIO-Dora

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README_en.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README.md)

## 快速开始

在开始前，请确保您已经完成 [RoboDriver文档/概览/安装与部署](https://flagopen.github.io/RoboDriver-Doc/docs/overview/installation/) 中的步骤。

要启动使用 `Dora` 驱动的机器人，需要分别启动两套程序，分别是 `dora数据流` 和 `RoboDriver`。这两套程序默认运行在不同的环境中，为了使 `dora` 节点和其对应硬件的复杂依赖问题和 `RoboDriver` 本身解耦。当然，如果dora部分依赖足够简单，也可统一放到`RoboDriver`环境中。

### 配置环境并启动 dora 数据流

新建一个终端，且暂时不激活任何环境。

检查您的系统中是否已经安装好 `dora-rs-cli`:

```
dora -V
```

如果正常安装，您应该可以看到输出： 

```
dora-cli <版本号>
```

如果没有，请参考 [RoboDriver文档/概览/安装与部署/推荐可选安装/dora](https://flagopen.github.io/RoboDriver-Doc/docs/overview/installation/#dora)

确保进入RoboDriver目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

进入到 `robodriver-robot-so101-aio-dora/dora` 目录。

```bash
cd robodriver/robots/robodriver-robot-so101-aio-dora/dora
```

创建多个 `uv` 环境:

```bash
uv venv camera.venv
uv venv arm.venv
```

通过 `dora` 自动安装依赖：

```bash
dora build dataflow.yml --uv
```

环境安装正确执行完成后，执行下一步 `硬件连接`。

硬件连接需要先将所有硬件断开连接，再重新按顺序连接，从而获得正确的编号。

1. 断开所有硬件USB连接。

2. 插入头部摄像头，这里默认插入的是 `realsense 435` 相机，如果您用的是别的相机或电脑自带有相机，编号及其数量可能会有所不同，请根据情况修改dora/dataflow.yml：

    ```bash
    ls /dev/video*
    # 可以看到： /dev/video0 /dev/video1 /dev/video2 /dev/video3 /dev/video4 /dev/video5
    # 可以查看(请先安装sudo apt install ffmpeg)： ffplay /dev/video2
    # 如果编号不同，请查看确认后，调整dora/dataflow.yml
    ```

3. 插入腕部摄像头
    ```bash
    ls /dev/video*
    # 可以看到： /dev/video0 /dev/video1 /dev/video2 /dev/video3 /dev/video4 /dev/video5 /dev/video6 /dev/video7
    ```

4. 插入 SO101 主臂 USB（如何区分主从臂? 主臂末端是一个扳机，主臂使用5V电源）：
    ```bash
    ls /dev/ttyACM*
    # 可以看到: /dev/ttyACM0
    ```

5. 插入 SO101 从臂 USB（如何区分主从臂? 主臂末端是一个扳机，主臂使用5V电源）：
    ```bash
    ls /dev/ttyACM*
    # 可以看到: /dev/ttyACM0 /dev/ttyACM1
    ```

6. 为机械臂 USB 接口赋予权限：
    ```
    sudo chmod 666 /dev/ttyACM0
    sudo chmod 666 /dev/ttyACM1
    ```

启动 `dora` ：

```
dora up
```

启动 `dora` 数据流

```bash
dora start dataflow.yml --uv
```

### 配置环境并启动 RoboDriver

新建一个终端，且暂时不激活任何环境。

确保进入RoboDriver目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

激活 `RoboDriver` 环境：

```bash
source .venv/bin/activate
```

进入到 `robodriver-robot-so101-aio-dora` 目录。

```bash
cd robodriver/robots/robodriver-robot-so101-aio-dora
```

安装依赖

```bash
uv pip install -e .
```

`RoboDriver` 部分启动命令如下:

```bash
robodriver-run --robot.type=so101_aio_dora
```

## LeRobot 推理

AIO 插件也可以用于 LeRobot 推理。如果不需要主臂参与，可跳过 RoboDriver 对主臂关节数据的等待：

```bash
robodriver-run --robot.type=so101_aio_dora --robot.require_leader=false
```

然后在平台发送 `start_inference`，设置 `policy_type=lerobot`、`data_channel.url=ws://10.1.2.74:8003` 和任务 `prompt`。当前 ACT 模型直接使用 AIO 的 `observation.images.image_top` 和 `observation.images.image_wrist`，不需要额外相机映射。

`require_leader=false` 只跳过 RoboDriver 的主臂数据检查，`dora/dataflow.yml` 中仍保留主臂节点。推理时可在确保机械臂安全的前提下断开主臂，避免 `arm_so101_leader/all_joint` 继续向从臂发送遥操动作；如果需要从启动时起就完全移除主臂节点，还应同步移除 dataflow 中该节点及其连线。

## TODO

- 完善校准程序

## 致谢

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to TheRobotStudio 🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).

## 引用

```bibtex
@misc{RoboDriver,
  author = {RoboDriver Authors},
  title = {RoboDriver: A robot control and data acquisition framework},
  month = {November},
  year = {2025},
  url = {https://github.com/FlagOpen/RoboDriver}
}
```