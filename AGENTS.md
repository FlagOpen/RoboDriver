# AGENTS.md

本文件面向 AI 编码代理，介绍 RoboDriver 项目的架构、开发流程与约定。

## 项目概述

RoboDriver 是一个机器人控制与数据采集框架，是 [CoRobot](https://github.com/FlagOpen/CoRobot) 数据体系中的**设备端驱动适配层**，向上对接 [RoboDriver-Server](https://github.com/FlagOpen/RoboDriver-Server)（数据/控制桥接层）和 RoboXStudio（云端控制台）。项目由 HuggingFace [LeRobot](https://github.com/huggingface/lerobot) 改进而来，机器人接口直接兼容 LeRobot 的 `Robot` 类，并扩展了 LeRobot 数据集格式（`DoRobotDataset`，采集端单条存储）。

- 仓库：https://github.com/FlagOpen/RoboDriver （许可证：Apache 2.0）
- 文档站点：https://flagopen.github.io/RoboDriver-Doc
- 包名：`RoboDriver`，版本 0.0.3，要求 Python >=3.10,<4.0
- 项目同时维护中文（README.md，默认）和英文（README_en.md）文档；代码注释与 docstring 以英文为主

## 技术栈

- **语言/运行时**：Python 3.10，asyncio 驱动主控制循环
- **核心依赖**（见 `pyproject.toml`）：
  - `torch>=2.7.1`
  - `lerobot-robodriver>=0.4.0`（LeRobot 的 RoboDriver 定制 fork，提供 `Robot`/`RobotConfig`、`Teleoperator`/`TeleoperatorConfig` 基类与 draccus ChoiceRegistry 配置注册机制）
  - `mujoco`（仿真）、`rerun-sdk`（数据可视化）、`python-socketio` + `aiohttp`（与 RoboDriver-Server 通信）
  - `logging_mp`（多进程日志）、`sounddevice`/`soundfile`（音频）、`evdev`（手柄/键盘输入）
- **接入方式**：除直接 SDK 外，支持 ROS1、ROS2 与 [dora-rs](https://github.com/dora-rs/dora) 数据流接入
- **配置解析**：`robodriver/utils/parser.py` 基于 draccus 风格，CLI 参数形如 `--robot.type=xxx --teleop.type=yyy`

## 构建与安装

使用 `uv`（不要先激活其他环境）：

```bash
uv venv -p 3.10
uv pip install -e .
source .venv/bin/activate
```

`setup.py` 仅为 setuptools 占位，真正的元数据在 `pyproject.toml`。仓库根目录已有 `.venv/`。机器人/遥操/仿真插件需单独安装（见下文“插件机制”）。

## 命令行入口

`pyproject.toml` 注册了四个入口点（对应 `robodriver/scripts/`）：

| 命令 | 模块 | 用途 |
|------|------|------|
| `robodriver-run` | `robodriver.scripts.run:main` | 主控制管线：启动 Daemon、遥操、仿真、Coordinator |
| `robodriver-sim` | `robodriver.scripts.sim:main` | 仅仿真模式 |
| `robodriver-visual` | `robodriver.scripts.visual:main` | 数据集可视化 |
| `robodriver-inference` | `robodriver.scripts.inference:main` | 策略推理 |

`scripts/` 目录下的 shell 脚本（`run-dora.sh`、`record.sh` 等）是交互式包装器。**注意：部分脚本已过时**，例如 `replay.sh` 仍引用旧的 `operating_platform/` 路径，使用前请核对。

## 代码结构

```
robodriver/                  # 主 Python 包
├── core/                    # 运行时核心
│   ├── coordinator.py       # 与 RoboDriver-Server 的 socketio 桥接、录制/回放/推理调度
│   ├── recorder.py replayer.py inferencer.py monitor.py
│   ├── disturbance.py       # 动作扰动注入（有单测覆盖）
│   ├── simulator.py         # 仿真封装（mujoco）
│   ├── ros2thread.py        # ROS2 节点线程管理（robot/teleop type 含 "ros2"/"mix" 时启用）
│   └── policies/            # 推理策略：bc / flagscale / openpi
├── robots/                  # 机器人插件目录 + Daemon/configs/statuses 基础设施
│   └── robodriver-robot-*   # 每个机器人适配是一个独立的可安装子包
├── teleoperators/           # 遥操插件（leader 臂、键盘、autotask）
├── simulations/             # 仿真环境插件（如 Genesis Franka）
├── _simulators/             # 内置 mujoco 仿真器模块
├── dataset/                 # DoRobotDataset、图像/音频写入、统计计算、visual/ 可视化
├── scripts/                 # CLI 入口脚本（见上表）
├── tools/AutoDriver/        # 自动化采集工具
└── utils/                   # 日志、常量、数据文件、插件注册、parser 等

components/                  # dora-rs 数据流节点包（dora-arm-*、dora-camera-*、dora-sim-*、legacy/）
descriptions/                # 机器人描述文件（urdf/、mjcf/ 等）
packages/openpi-client/      # openpi 策略客户端（独立 hatchling 包）
tests/                       # 正式单元测试
test/                        # 临时实验/调试脚本目录，不是测试套件
scripts/                     # shell 包装脚本
```

## 插件机制（重要）

新增机器人/遥操设备**不改动主包代码**，而是新增独立子包：

1. 命名规范：`robodriver-robot-[机器人型号]-[遥操方式]-[接入类型]`（如 `robodriver-robot-so101-aio-dora`；遥操方式如 `aio`/`follower`/`teleoperate`，接入类型如 `ros1`/`ros2`/`dora`/`sdk-py`）。
2. 每个插件目录自带 `pyproject.toml`、`README.md`，需单独 `uv pip install -e <插件路径>` 安装。
3. 运行时由 `robodriver/utils/import_utils.py::register_third_party_devices()` 扫描 `sys.path` 上 `robodriver_robot_`、`robodriver_camera_`、`robodriver_sim`、`robodriver_policy_`、`robodriver_teleoperator` 前缀的模块并导入；插件在导入时通过 draccus `ChoiceRegistry` 注册自己的 `*Config` 类，随后可用 `--robot.type=<注册名>` 引用。
4. 设备类与配置类的对应约定：配置类 `XxxConfig` → 设备类 `Xxx`（去掉 `Config` 后缀），工厂函数 `make_device_from_device_class` 按此规则定位。

## 运行时架构

`robodriver-run` 的主循环（`robodriver/scripts/run.py`）：

1. `register_third_party_devices()` 导入全部插件；
2. 按需创建 `ROS2_NodeManager`、`Teleoperator`、`Simulator`；
3. `Daemon`（`robodriver/robots/daemon.py`）持有 Robot 实例并周期性 `update()` 采集观测；
4. `Coordinator` 通过 socketio 与 RoboDriver-Server 通信，负责流信息上报、图像流推送、录制/回放/推理指令；
5. 遥操动作经 `ActionDisturbance` 扰动层后下发；AIO 机器人默认镜像 leader 硬件，仅在扰动激活时才发送动作；
6. 默认帧率 `DEFAULT_FPS = 30`（`robodriver/utils/constants.py`），数据集默认存放于 `$ROBODRIVER_HOME/dataset`，rerun 端口 9195/9285。

## 测试

- 正式测试在 `tests/`（基于 `unittest`，如 `test_disturbance.py`）：

  ```bash
  python -m pytest tests/
  # 或
  python -m unittest discover tests
  ```

- `test/` 目录（audio、ik、mujoco、lerobot 等子目录）是历史实验/调试脚本，**不要**把新测试放进去，也不要假设它们能运行。
- 项目测试覆盖很少，修改核心模块时应自行验证端到端流程（参考 `README.md` 中各机器人插件内的 README 指南）。

## 代码风格与提交规范

- 预提交钩子（`.pre-commit-config.yaml`）：**black**（Python 3.10）+ **isort**（`--profile black`）、`detect-private-key`、语法/JSON/YAML 检查、行尾与空白修复；**禁止直接提交到 `main` 和 `release` 分支**。
- 提交信息遵循 **Conventional Commits**（由 `conventional-pre-commit` 钩子强制）。
- PR 目标分支为 **`dev`**（README 贡献指南明确要求）。
- 部分源自 LeRobot 的文件保留 Apache 2.0 版权头，修改时予以保留。

## 安全注意事项

- 不要提交任何密钥/凭据（pre-commit 已配置 `detect-private-key`）。
- 机器人插件可能包含厂商 SDK 与硬件配置，修改相机/机械臂的 dora dataflow（`components/` 或插件内 `dataflow.yml`）前请确认目标硬件环境。
- `out/`、`build/`、`dist/`、各 `*.egg-info/` 为生成产物，不应手工编辑或提交。

## 开发提示

- 主包与插件均为 editable 安装，改动即生效，无需重装。
- `robodriver/scripts/record.py` 中存在旧 import 路径（`robodriver.robot.robots.configs`），独立运行该脚本前需核实；录制功能通常经由 `Coordinator`/`Recorder` 在 `robodriver-run` 流程内触发。
- 与 ROS2 混用时，robot/teleop 的 `type` 字符串包含 `ros2` 或 `mix` 会触发 `ROS2_NodeManager` 的创建与节点注册，新增接入类型时注意该字符串匹配逻辑。
