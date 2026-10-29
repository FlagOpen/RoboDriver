# RoboDriver-Teleoperator-Agilex-Aloha-Leader-Dora

[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README_en.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README.md)


在开始前，请确保您已经完成 [RoboDriver文档/概览/安装与部署](https://flagopen.github.io/RoboDriver-Doc/docs/overview/installation/) 中的步骤或已完成RoboDriver仓库README中的 `快速入门` 。

要启动使用 `Dora` 驱动的机器人，需要分别启动两套程序，分别是 `dora数据流` 和 `RoboDriver`。这两套程序默认运行在不同的环境中，为了使 `dora` 节点和其对应硬件的复杂依赖问题和 `RoboDriver` 本身解耦。当然，如果dora部分依赖足够简单，也可统一放到`RoboDriver`环境中。

在首次部署Roboriver时，需要在机器上执行环境安装和硬件配置。

## 环境安装

### dora 数据流

新建一个终端，且暂时不激活任何环境。

检查您的系统中是否已经安装好 `dora-rs-cli-robodriver`:

```bash
dora -V
```

如果正常安装，您应该可以看到输出： 

```bash
dora-cli 0.3.14
```

如果没有，请参考 [RoboDriver文档/概览/安装与部署/推荐可选安装/dora](https://flagopen.github.io/RoboDriver-Doc/docs/overview/installation/#dora) 或直接安装：

```bash
pip install dora-rs-cli-robodriver
```

确保进入RoboDriver目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

进入到 `robodriver-teleoperator-agilex-aloha-leader-dora/` 目录。

```bash
cd robodriver/teleoperators/robodriver-teleoperator-agilex-aloha-leader-dora/
```

进入到 `dora/` 目录。

```bash
cd dora
```

创建多个 `uv` 环境:

```bash
uv venv arm.venv
```

```bash
dora build dataflow.yml --uv
```

### robodriver-teleoperator-agilex-aloha-leader-dora

新建一个终端，且暂时不激活任何环境。

确保进入RoboDriver目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

激活 `RoboDriver` 环境：

```bash
source .venv/bin/activate
```

进入到 `robodriver-teleoperator-agilex-aloha-leader-dora` 目录。

```bash
cd robodriver/teleoperators/robodriver-teleoperator-agilex-aloha-leader-dora
```

安装依赖

```bash
uv pip install -e .
```

## 硬件配置

新建终端并进入 `RoboDriver` 项目目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

进入到 `robodriver-teleoperator-agilex-aloha-leader-dora/` 目录。

```bash
cd robodriver/teleoperators/robodriver-teleoperator-agilex-aloha-leader-dora/
```

### 机械臂CAN激活

首先先将机械臂的CAN转USB接口从电脑拔出。然后先插入右臂的USB，然后运行。

```bash
sudo bash ./scripts/find_can_port.sh
# 记住这里的USB位置
```

然后根据刚才的命令输出，修改 `./scripts/can_muti_activate.sh` 中第4行。

```bash
USB_PORTS["1-1:1.0"]="can_leader_right:1000000"
```

然后插入左臂，再次按照上文操作。

然后运行，`can_muti_activate.sh` 激活：

```bash
sudo bash ./scripts/can_muti_activate.sh
```

## 启动

### dora 数据流

新建终端并进入 `RoboDriver` 项目目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

不激活任何环境，如果激活了就退出：

```bash
deactivate # uv
conda deactivate # conda
```

启动 `dora` ：

```bash
dora up
```

启动 `dora` 数据流

```bash
dora start robodriver/teleoperators/robodriver-teleoperator-agilex-aloha-leader-dora/dora/dataflow.yml --uv
```

如果 `dora` 数据流在运行过程中出现了任何问题，或后续步骤不正常了，请关闭该程序后重新插拔硬件USB或重启后，再次运行上文的 `硬件配置` 后，再次尝试。

### RoboDriver

新建终端并进入 `RoboDriver` 项目目录，如果已经进入就跳过：

```bash
cd RoboDriver/
```

激活 `RoboDriver` 环境：

```bash
source .venv/bin/activate
```

运行：

```bash
robodriver-run \
    --teleop.type=agilex_aloha_leader_dora \
    --robot.type=agilex_aloha_follower_dora \
    --sim.xml_path=descriptions/agilex_aloha/scene.xml \
    --sim.from_unit=rad
```

干扰触发功能默认关闭。如需使用脚踏板，请增加
`--robot.disturbance.enabled=true`。运行期间踩下映射为 `s` 键的脚踏板，
会同时将左右夹爪设置为张开值 `100`，默认保持 0.5 秒。该功能只修改夹爪动作，
不影响双臂关节动作。

按键监听通过 `evdev` 直接读取系统输入设备，程序在后台或其它终端运行时同样生效，
无需聚焦本终端。需要当前用户对 `/dev/input` 有读权限，如没有请执行后重新登录：

```bash
sudo usermod -aG input $USER
```

如果自动识别不到脚踏板，可用 `ls /dev/input/by-id`（或 `sudo evtest`）找到设备，
再用 `--robot.disturbance.manual_device=/dev/input/eventX` 指定。evdev 不可用时
自动回退到终端按键监听（此时仅在终端聚焦时有效）。

如需同时启用随机触发（下例为每隔 10～30 秒随机触发一次）：

```bash
robodriver-run \
    --teleop.type=agilex_aloha_leader_dora \
    --robot.type=agilex_aloha_follower_dora \
    --sim.xml_path=descriptions/agilex_aloha/scene.xml \
    --sim.from_unit=rad \
    --robot.disturbance.enabled=true \
    --robot.disturbance.random_enabled=true \
    --robot.disturbance.random_min_interval_s=10 \
    --robot.disturbance.random_max_interval_s=30
```

可通过 `--robot.disturbance.duration_s=1.0` 修改夹爪张开的保持时间。

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
