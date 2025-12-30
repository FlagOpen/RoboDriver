# RoboDriver-Robot-Agilex-Aloha-AIO-Dora

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

进入到 `robodriver-robot-agilex-aloha-aio-dora/dora` 目录。

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

2. 连接三个 Orbbec 摄像头（顶部、右侧、左侧）：
    - 确保摄像头已连接并通电
    - 检查设备序列号与 dataflow.yml 中的配置匹配

3. 连接 Piper 右臂 CAN 总线：
    ```bash
    sudo ip link set can_right up type can bitrate 1000000
    ```

4. 连接 Piper 左臂 CAN 总线：
    ```bash
    sudo ip link set can_left up type can bitrate 1000000
    ```

5. 检查 CAN 总线状态：
    ```bash
    ip -details link show can_right
    ip -details link show can_left
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

进入到 `robodriver-robot-agilex-aloha-aio-dora` 目录。

```bash
cd robodriver/robots/robodriver-robot-agilex-aloha-aio-dora
```

安装依赖

```bash
uv pip install -e .
```

`RoboDriver` 部分启动命令如下:

```bash
robodriver-run --robot.type=agilex_aloha_aio_dora
```

## TODO

- 完善校准程序
- 改进错误处理
- 添加更多文档

## 致谢

- Thanks to LeRobot team 🤗, [LeRobot](https://github.com/huggingface/lerobot).
- Thanks to Agilex Robotics 🤗, [Agilex Robotics](https://www.agilex.ai/).
- Thanks to dora-rs 🤗, [dora](https://github.com/dora-rs/dora).
- Thanks to Piper team 🤗, [Piper](https://github.com/your-piper-repo).

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
