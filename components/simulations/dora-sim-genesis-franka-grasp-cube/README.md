# dora-sim-genesis-franka-grasp-cube

This Dora node launches a simulated environment featuring a Franka robot and a graspable cube. It publishes real-time data streams—including the robot’s joint states and end-effector pose—and accepts incoming control commands to drive the robot’s motion.

## 项目概述

这是一个基于Dora框架和Genesis仿真引擎的Franka机械臂抓取立方体仿真系统。通过键盘控制机械臂在仿真环境中移动和抓取立方体，并实时发布图像数据流。

## 安装

您可以使用 **uv** 或 **conda** 以开发模式安装此包：

```bash
pip install -e .
```

要启用可视化支持（例如使用Rerun），请安装可选的 `view` 扩展：

```bash
pip install -e ".[view]"
```

## 使用方法

安装后，该节点可以作为Dora数据流的一部分启动。当安装了 `view` 扩展时，可以查看仿真场景的可视化。

### 运行

```bash
dora up examples/view.yml
```

## 键盘控制说明

### 位置控制
- 方向键↑/↓ - 前/后移动 (X轴)
- 方向键←/→ - 左/右移动 (Y轴)
- +/= 键 - 向上移动 (Z轴)
- - 键 - 向下移动 (Z轴)

### 姿态控制 (欧拉角)
- 6/4 键 - 增加/减少绕X轴旋转 (Roll)
- 8/2 键 - 增加/减少绕Y轴旋转 (Pitch)
- 7/9 键 - 增加/减少绕Z轴旋转 (Yaw)
- 空格键 - 重置姿态为初始值

### 夹爪控制
- B 键 - 夹紧 (切换)
- N 键 - 松开 (切换)

### 其他控制
- 退格键 - 重置位置和姿态
- P 键 - 打印当前位置和姿态
- V 键 - 切换速度模式 (标准/高速)
- ESC键 - 退出程序

## 代码改进

### 1. 配置外部化
所有硬编码参数已移到 `config/simulation_config.yaml` 配置文件中，包括：
- 仿真环境参数（平面、立方体、相机）
- 机械臂控制参数（KP/KV增益、力范围）
- 键盘控制参数（步长、初始位置）
- 性能监控设置
- 线程配置

### 2. 模块化设计
- `src/config_loader.py` - 配置加载器，支持YAML配置文件和参数验证
- `src/keyboard_controller.py` - 改进的键盘控制器，支持速度模式切换和状态反馈
- `src/main_improved.py` - 改进的主程序，增强错误处理和性能监控

### 3. 错误处理增强
- 逆运动学求解失败统计和恢复机制
- 详细的异常处理和堆栈跟踪
- 线程安全的资源清理

### 4. 性能监控改进
- scene.step 执行时间和间隔统计
- cam.render 时间统计
- tick 调用间隔统计
- 逆运动学成功率统计

### 5. 用户体验提升
- 速度模式切换（标准/高速）
- 实时状态反馈
- 更详细的控制说明
- 定期性能统计报告

## 测试

运行测试脚本验证改进功能：

```bash
python test_improvements.py
```

## 项目结构

```
dora-sim-genesis-franka-grasp-cube/
├── config/
│   └── simulation_config.yaml    # 配置文件
├── src/
│   ├── main.py                   # 原始主程序
│   ├── main_improved.py          # 改进版主程序
│   ├── keyboard_controller.py    # 键盘控制器
│   └── config_loader.py          # 配置加载器
├── examples/
│   ├── view.yml                  # Dora数据流配置
│   └── test_rerun.yml
├── test_improvements.py          # 测试脚本
└── README.md                     # 本文档
```

## 依赖

- Python >= 3.10, < 3.14
- dora-rs (>=0.3.11, <0.4.0)
- genesis-world >= 0.3.7
- torch
- pynput
- opencv-python
- pyarrow
- PyYAML

## 许可证

详见 LICENSE 文件。
