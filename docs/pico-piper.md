# Pico Ultra4 遥操 Piper 实现需求文档

## 概述
本文档描述了在 RoboDriver 框架中添加 Pico Ultra4 遥操控制 Piper 机械臂所需实现的组件和功能。

## 1. 核心组件结构

### 1.1 遥操器包 (Teleoperator Package)
需要创建新的遥操器包：`robodriver-teleoperator-pico-ultra4-dora`

**目录结构：**
```
robodriver/teleoperators/robodriver-teleoperator-pico-ultra4-dora/
├── pyproject.toml                          # 包配置文件
├── README.md                               # 说明文档
├── dora/
│   └── dataflow.yml                        # Dora 数据流配置
├── robodriver_teleoperator_pico_ultra4_dora/
│   ├── __init__.py                         # 导出主要类
│   ├── config.py                           # 配置类
│   ├── status.py                           # 状态类
│   ├── node.py                             # Dora 节点类
│   ├── teleoperator.py                     # 遥操器主类
│   └── calibrate.py                        # 校准功能（可选）
└── lerobot_teleoperator_pico_ultra4_dora/
    └── __init__.py                         # LeRobot 兼容层
```

### 1.2 Pico Ultra4 硬件组件
需要在 `components/arms/` 下创建或使用现有的 Pico Ultra4 驱动组件。

**可能需要创建：**
```
components/arms/dora-arm-pico-ultra4/
├── pyproject.toml
├── dora_arm_pico_ultra4/
│   ├── __init__.py
│   ├── main.py                             # Dora 节点主程序
│   └── read_data.py                        # 数据读取工具
└── examples/
    └── read_pico_ultra4.yml                # 示例配置
```

## 2. 需要实现的核心类

### 2.1 配置类 (config.py)
```python
@TeleoperatorConfig.register_subclass("pico_ultra4_dora")
@dataclass
class PicoUltra4DoraTeleoperatorConfig(TeleoperatorConfig):
    # 定义 Pico Ultra4 的电机配置
    motors: Dict[str, Motor]
    # 其他配置参数
```

**关键配置项：**
- 电机数量和类型
- 归一化模式（角度/范围）
- 通信参数（端口、波特率等）

### 2.2 状态类 (status.py)
```python
@TeleoperatorStatus.register_subclass("pico_ultra4_dora")
@dataclass
class PicoUltra4DoraTeleoperatorStatus(TeleoperatorStatus):
    device_name: str = "Pico_Ultra4"
    device_body: str = "Pico"
    # 定义手臂规格、自由度、关节限位等
```

**关键状态信息：**
- 设备名称和型号
- 手臂数量和类型
- 关节限位
- 连接状态
- FPS 等性能参数

### 2.3 Dora 节点类 (node.py)
```python
class PicoUltra4DoraTeleoperatorNode(DoraTeleoperatorNode):
    def __init__(self):
        # 初始化 Dora 节点
        # 创建数据接收缓冲区
        # 启动接收线程

    def dora_recv(self, timeout: float):
        # 接收来自 Pico Ultra4 的关节数据

    def dora_send(self, event_id, data):
        # 发送控制命令（如果需要反馈）
```

**关键功能：**
- 异步接收 Pico Ultra4 的关节角度数据
- 管理数据缓冲和超时检测
- 线程安全的数据访问

### 2.4 遥操器主类 (teleoperator.py)
```python
class PicoUltra4DoraTeleoperator(Teleoperator):
    config_class = PicoUltra4DoraTeleoperatorConfig
    name = "pico_ultra4_dora"

    def connect(self):
        # 连接 Pico Ultra4 设备
        # 等待数据流就绪

    def get_action(self) -> dict[str, float]:
        # 读取当前关节位置作为动作

    def send_feedback(self, feedback: dict[str, Any]):
        # 发送力反馈（如果支持）

    def update_status(self) -> str:
        # 更新设备状态

    def disconnect(self):
        # 断开连接
```

**关键方法：**
- `connect()`: 建立与设备的连接，等待数据流
- `get_action()`: 获取遥操动作（关节位置）
- `send_feedback()`: 发送反馈信号（可选）
- `update_status()`: 更新连接状态
- `disconnect()`: 清理资源

## 3. Dora 数据流配置

### 3.1 dataflow.yml 结构
```yaml
nodes:
  - id: pico_ultra4_leader
    path: nodes/dora_arm_pico_ultra4/dora_arm_pico_ultra4/main.py
    build: pip install -e nodes/dora_arm_pico_ultra4
    inputs:
      get_joint: dora/timer/millis/33  # 30Hz 采样
    outputs:
      - joint_1
      - joint_2
      - joint_3
      - joint_4
      - joint_5
      - joint_6
      - joint_gripper
    env:
      DEVICE_PORT: /dev/ttyUSB0        # 根据实际设备调整
      ARM_NAME: Pico-Ultra4-leader
      ARM_ROLE: leader

  - id: pico_ultra4_dora
    path: dynamic
    inputs:
      leader_joint_1: pico_ultra4_leader/joint_1
      leader_joint_2: pico_ultra4_leader/joint_2
      # ... 其他关节
    outputs:
      - action_joint
```

**关键配置：**
- 定义 Pico Ultra4 硬件节点
- 配置数据采样频率（通常 30Hz）
- 映射关节输入/输出
- 设置环境变量（设备端口、名称等）

## 4. 硬件驱动实现

### 4.1 Pico Ultra4 SDK 集成
需要集成 Pico Ultra4 的 Python SDK 或通信协议。

**参考 Piper 实现：**
- 使用 `piper_sdk.C_PiperInterface` 类似的接口
- 实现关节角度读取
- 实现夹爪控制（如果有）

### 4.2 数据转换
- 原始数据 → 标准化关节角度（弧度）
- 处理数据格式转换和单位换算
- 实现数据滤波（如果需要）

## 5. 与 Piper 机械臂的集成

### 5.1 使用现有 Piper 组件
可以复用现有的 `components/arms/dora-arm-piper` 组件作为从臂（follower）。

### 5.2 数据映射
需要建立 Pico Ultra4 关节到 Piper 关节的映射关系：
- 关节数量匹配（Pico Ultra4 vs Piper）
- 关节角度范围映射
- 夹爪控制映射

## 6. 依赖项

### 6.1 Python 包依赖
在 `pyproject.toml` 中添加：
```toml
dependencies = [
    "dora-rs-cli (>=0.3.11,<0.4.0)",
    "logging_mp",
    "numpy",
    "pyarrow",
    # Pico Ultra4 SDK（根据实际情况）
]
```

### 6.2 系统依赖
- Dora-rs 运行时
- 硬件驱动（USB/串口驱动）
- 可能需要的权限配置（udev rules）

## 7. 测试和验证

### 7.1 单元测试
- 测试配置加载
- 测试数据接收
- 测试动作生成

### 7.2 集成测试
- 测试 Pico Ultra4 连接
- 测试数据流完整性
- 测试与 Piper 的协同工作

### 7.3 性能测试
- 延迟测试（< 50ms）
- 频率测试（30Hz 稳定）
- 长时间运行稳定性

## 8. 实现步骤建议

1. **准备阶段**
   - 研究 Pico Ultra4 硬件规格和 SDK
   - 确认关节数量、自由度、通信协议

2. **硬件驱动层**
   - 创建 `dora-arm-pico-ultra4` 组件
   - 实现基本的数据读取功能
   - 测试硬件通信

3. **遥操器层**
   - 创建遥操器包结构
   - 实现配置、状态、节点类
   - 实现主遥操器类

4. **Dora 集成**
   - 编写 dataflow.yml
   - 测试数据流
   - 调试节点通信

5. **与 Piper 集成**
   - 配置 Piper 作为从臂
   - 建立数据映射
   - 测试遥操控制

6. **优化和调试**
   - 性能优化
   - 延迟优化
   - 错误处理完善

## 9. 参考实现

可以参考以下现有实现：
- `robodriver-teleoperator-so101-leader-dora`: 完整的遥操器实现示例
- `components/arms/dora-arm-piper`: Piper 硬件驱动
- `components/arms/dora-arm-so101`: SO101 硬件驱动

## 10. 注意事项

1. **安全性**
   - 实现急停功能
   - 关节限位保护
   - 通信超时处理

2. **性能**
   - 保持 30Hz 控制频率
   - 最小化延迟
   - 避免数据丢失

3. **可维护性**
   - 遵循现有代码风格
   - 添加充分的日志
   - 编写清晰的文档

4. **兼容性**
   - 与 LeRobot 框架兼容
   - 支持标准的 Teleoperator 接口
   - 可与其他组件组合使用
