# Changelog

本项目遵循 [Semantic Versioning](https://semver.org/lang/zh-CN/) 与 [Keep a Changelog](https://keepachangelog.com/zh-CN/) 约定。

## [0.0.4] - 2026-10-29

### 新增

- 推理链路支持 BC 策略（`robodriver/core/policies/bc_policy.py`）：直接从 `robot.get_observation()` 提取图像/状态，异步 WebSocket + msgpack 通信，图像 letterbox resize。
- `InferenceConfig` 新增 `headers`、`robot_type` 字段；`policy_type` 支持 `flagscale` / `openpi` / `bc`。
- 扰动模块手动触发改为基于 evdev 的系统级监听（后台/SSH 无终端焦点可用），新增 `manual_device` 配置项，evdev 不可用时回退 stdin 键盘监听；新增依赖 `evdev`。
- 新插件包：`robodriver-robot-agilex-aloha-eepose-mix`、`robodriver-robot-agilex-aloha-follower-dora`、`robodriver-teleoperator-agilex-aloha-leader-dora`。
- dora-arm-piper 组件（0.0.5 → 0.0.16）：新增 `ARM_TYPE`（aio/leader/follower）模式；EEPOSE 模式下基于 Pinocchio + CasADi 的逆运动学与坐标系变换；平滑关节插值。
- `reset_pose` 消息支持 `gripper_reset_value` 参数覆盖夹爪复位目标值。
- 新增 `pi0.5.md` 数据指南、`AGENTS.md` 开发指南。

### 变更

- **行为变化**：`reset_pose` 复位时夹爪不再保持当前值，默认复位到 100（`RANGE_0_100` 归一化下为完全张开，用于松开工件）；可通过 `gripper_reset_value` 覆盖。
- **行为变化**：`start_inference` 未指定 `policy_type` 时默认使用 `bc` 推理路径。
- robot type 含 `mix` 时启用 `ROS2_NodeManager`。

## [0.0.3] - 2026-01-20

### 变更

- 数据集路由更新与推理动作日志增强；推理支持 rerun 可视化与动作块执行。

## [0.0.2] - 2026-01-09

早期版本，详见 Git 历史。

## [0.0.1] - 2026-01-06

首个发布版本。
