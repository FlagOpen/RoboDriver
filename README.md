![RoboDriver](assets/images/robodriver.png)

[![GitHub Issues](https://img.shields.io/github/issues/FlagOpen/RoboDriver)](https://github.com/FlagOpen/RoboDriver/issues)
[![GitHub Discussions](https://img.shields.io/github/discussions/FlagOpen/RoboDriver)](https://github.com/FlagOpen/RoboDriver/discussions)


[![README in English](https://img.shields.io/badge/English-d9d9d9)](./README.md)
[![简体中文版自述文件](https://img.shields.io/badge/简体中文-d9d9d9)](./README_zh.md)


# RoboDriver
RoboDriver是DataCollect的核心驱动层组件，也是[CoRobot](https://github.com/FlagOpen/CoRobot)数据体系中的标准化机器人接入模块。

<p align="center">
  <img src="assets/images/robodriver_struct_1.png" alt="RoboDriver 架构图" width="70%"/>
</p>

如图所示，RoboDriver为“设备端驱动适配层”，[RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server)是“数据/控制的桥接层与通道路由器”，[RoboXStudio](https://ei2data.baai.ac.cn/home)是“云端或平台侧总控制台与数据管理中心”。

RoboDriver使用文档: [RoboDriver-Doc](https://flagopen.github.io/RoboDriver-Doc)
## 最新消息
- [2025-12-01] RoboDriver项目开源

## 目录

1. [概述](#概述)
2. [主要特点](#主要特点)
3. [快速入门](#快速入门)
4. [仿真示例](#仿真示例)
5. [机器人示例](#机器人示例)
6. [参与贡献](#参与贡献)
7. [帮助支持](#帮助支持)
8. [许可证与致谢](#许可证和致谢)
9. [引用](#引用)

## 主要特点

- **多种机器人接入方式**： RoboDriver 考虑了除SDK外，使用ROS、Dora的接入方式。
- **LeRobot兼容**： RoboDriver 的机器人接口直接使用了LeRobot的`Robot`类，这意味着RoboDriver与LeRobot可以互相兼容。
- **改进的LeRobot数据集格式**：在不同数据阶段采取不同数据结构。在采集端单条储存，更易编辑和传输。还扩展了LeRobot格式的内容。


## 快速入门

请参考项目文档：[RoboDriver-Doc](https://flagopen.github.io/RoboDriver-Doc)

## 仿真示例

考虑到机器人在实际环境中的各种不确定性，我们推荐您首先使用我们提供的 `仿真示例` 来尝试使用 `RoboDriver`。

RoboDriver 已完成 `Genesis` 仿真环境的适配，`mujoco` 和 `isaac sim` 等环境的适配正在开发中。使用请参考项目文档和仓库内对应的文件夹内`README`。

### 🪞 Genesis

| 机器人型号 | 简介 | 仓库链接 | 仓库贡献人（联系方式） |
|------------|------|--------------|------------------------|
| Franka 机械臂 | 一台Franka机械臂抓取方块 | [robodriver/simulations/robodriver-sim-genesis-franka-aio-dora](./robodriver/simulations/robodriver-sim-genesis-franka-aio-dora) | yangxiang（yangxiang@baai.ac.cn） |

## 机器人示例
RoboDriver 已完成多款主流机器人的适配，按接入方式示例如下（各仓库包含对应机器人的接入步骤、环境配置、指令适配等完整指南）：

### 🔌 ROS1 接入
| 机器人型号 | 简介 | 代码链接 | 贡献人（联系方式） |
|------------|------|--------------|------------------------|
| Realman 机械臂 | 基于Realman，6DOF+力控模块，3*RealSense相机 | [robodriver/robots/robodriver-robot-realman-aio-ros1](./robodriver/robots/robodriver-robot-realman-aio-ros1) | yangxiang（[https://github.com/Ryu-Yang](https://github.com/Ryu-Yang)） |

### 🔌 ROS2 接入
| 机器人型号 | 简介 | 代码链接 | 贡献人（联系方式） |
|--------------|--------------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------|
| GALAXEALITE | 基于Galaxealite，双臂6DOF+末端夹爪，4*RealSense相机 | [robodriver/robots/robodriver-robot-galaxealite-aio-ros2](./robodriver/robots/robodriver-robot-galaxealite-aio-ros2) | liuyou（[https://github.com/liuyou1103](https://github.com/liuyou1103)） |
| SO101 机械臂 | 开源轻量级机械臂，6DOF+末端夹爪，1*RealSense相机，1*RGB相机模块 | [robodriver/robots/robodriver-robot-so101-aio-ros2](./robodriver/robots/robodriver-robot-so101-aio-ros2) | yangxiang（[https://github.com/Ryu-Yang](https://github.com/Ryu-Yang)） |

### 🔌 Dora（SDK）接入
| 机器人型号 | 简介 | 代码链接 | 贡献人（联系方式） |
|--------------|--------------------------------------------------------------|------------------------------------------------------------------------------------------|------------------------|
| Realman 机械臂 | 基于Realman，6DOF+力控模块，3*RealSense相机 | [robodriver/robots/robodriver-robot-realman1-aio-dora](./robodriver/robots/robodriver-robot-realman1-aio-dora) | xuruntian（[https://github.com/XuRuntian](https://github.com/XuRuntian)） |
| SO101 机械臂 | 开源轻量级机械臂，6DOF+末端夹爪，1*RealSense相机，1*RGB相机模块 | [robodriver/robots/robodriver-robot-so101-aio-dora](./robodriver/robots/robodriver-robot-so101-aio-dora) | yangxiang（[https://github.com/Ryu-Yang](https://github.com/Ryu-Yang)） |
| Franka | 工业级机械臂，6DOF+末端夹爪，1*RealSense相机 | [robodriver/robots/robodriver-robot-franka-aio-dora](./robodriver/robots/robodriver-robot-franka-aio-dora) | xuruntian（[https://github.com/XuRuntian](https://github.com/XuRuntian)） |

> ✨ 说明：
> 1. 接入方式命名规范：`robodriver-robot-[机器人型号]-[遥操方式]-[接入类型]`（如 `aio`/`follwer`/`teleoperate`, `ros2`/`dora`）；
> 2. 每个适配仓库内包含**环境搭建、配置修改、采集/控制验证**等完整接入指南；
> 3. 持续新增适配机器人，可关注本列表或项目更新。

我们非常欢迎社区开发者贡献更多机器人的实现！可按以下方式参与：
1. 参考已适配机器人的代码结构和 README 模板，按接入类型（ROS1/ROS2/Dora）完成适配开发；
2. 将适配代码新增至主仓库的 `robodriver/robots/` 目录下（命名规范与已适配机器人保持一致）；
3. 确保代码规范、文档完整（包含环境准备、配置步骤、功能验证）；
4. 提交代码 PR 至主仓库的 `dev` 分支，我们将及时审核并合并。

期待与您一起丰富 RoboDriver 的机器人生态！ 🤝

## 参与贡献

我们真诚地欢迎来自社区的 *任何形式的贡献*。从**新功能的拉取请求**、**错误报告**，到甚至是使RoboDriver更易用的微小**建议**，我们都全心全意地感谢！

## 帮助支持

- 请使用 Github [Issues](https://github.com/FlagOpen/RoboDriver/issues) 报告错误和提出功能请求。

- 请使用 GitHub [Discussions](https://github.com/FlagOpen/RoboDriver/discussions) 讨论想法和提问。


## 许可证和致谢

RoboDriver 源代码根据 Apache 2.0 许可证授权。 没有这些令人惊叹的开源项目，RoboDriver 的开发是不可能的：

- 感谢LeRobot团队开源LeRobot🤗, [LeRobot](https://github.com/huggingface/lerobot)。本项目由LeRobot改进而来。
- 感谢TheRobotStudio团队开源的SO100和SO101机械臂🤗, [SO101](https://github.com/TheRobotStudio/SO-ARM100)。在本项目中，SO101机械臂作为部署案例。
- 感谢dora-rs团队开源的机器人框架🤗, [dora](https://github.com/dora-rs/dora)。在本项目中，该框架为机器人带来了全新的接入方式。

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
