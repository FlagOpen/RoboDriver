# Pico Ultra4 遥操 Piper 实现总结

## 实现完成情况

### ✅ 已完成
1. **遥操器包结构** - `robodriver-teleoperator-pico-ultra4-dora`
2. **核心类实现**：
   - `config.py`: 配置类，定义 VR 控制器和 IK 参数
   - `status.py`: 状态类，管理设备连接状态
   - `node.py`: Dora 节点类，集成 VR 数据获取和 IK 求解
   - `teleoperator.py`: 遥操器主类，实现 Teleoperator 接口

3. **关键功能**：
   - XRoboToolkit SDK 集成：获取 Pico VR 控制器位姿
   - Placo IK 求解器集成：将末端位姿转换为关节角度
   - 多线程架构：VR 数据读取在独立线程中运行
   - 控制激活机制：通过握持键激活/停止控制
   - 夹爪控制：通过扳机键控制夹爪开合

## 架构说明

### 数据流
```
Pico VR 控制器
  ↓ (XRoboToolkit SDK)
末端位姿 [x, y, z, qx, qy, qz, qw]
  ↓ (缩放因子)
缩放后的位姿
  ↓ (Placo IK 求解器)
关节角度 [joint1...joint6]
  ↓ (Teleoperator 接口)
Piper 机械臂控制
```

### 核心组件

#### 1. PicoUltra4DoraTeleoperatorNode
- 管理 VR 数据获取（100Hz 更新频率）
- 执行 IK 求解
- 提供线程安全的数据访问

#### 2. PicoUltra4DoraTeleoperator
- 实现 LeRobot Teleoperator 接口
- 管理连接/断开流程
- 提供动作获取接口

## 待完成任务

### 🔄 需要进一步工作

1. **Piper 硬件接口适配**
   - 将 `/home/dora/teleop_pico` 中的 `PiperInterface` 类集成到 robodriver
   - 创建 Piper 的 Dora 节点（如果需要）
   - 或直接在遥操器中使用 PiperInterface

2. **Dora 数据流配置**
   - 创建 `dataflow.yml`
   - 定义节点间的数据流
   - 配置 Piper 机械臂节点

3. **依赖项安装**
   - xrobotoolkit_sdk
   - placo
   - piper_sdk
   - scipy (用于四元数转换)

4. **测试和调试**
   - 单元测试
   - 集成测试
   - 性能优化

## 使用方法（计划）

```bash
# 安装依赖
pip install -e robodriver/teleoperators/robodriver-teleoperator-pico-ultra4-dora

# 启动遥操（需要先完成 Piper 集成）
python -m robodriver.scripts.run \
    --teleoperator pico_ultra4_dora \
    --robot piper_aio_dora
```

## 配置示例

```python
from robodriver_teleoperator_pico_ultra4_dora import PicoUltra4DoraTeleoperatorConfig

config = PicoUltra4DoraTeleoperatorConfig(
    robot_urdf_path="/path/to/piper.urdf",
    vr_controller="right_controller",
    control_trigger="right_grip",
    gripper_trigger="right_trigger",
    scale_factor=1.5,
    control_rate_hz=50,
)
```

## 技术细节

### IK 求解
- 使用 Placo 库进行实时 IK 求解
- 支持软约束和硬约束
- 考虑当前关节状态作为初始值

### VR 数据处理
- 位姿格式：[x, y, z, qx, qy, qz, qw]
- 坐标系转换：支持自定义头显到世界坐标系的转换
- 缩放因子：可调整 VR 移动到机械臂移动的比例

### 安全特性
- 控制激活机制：必须按住握持键才能控制
- 关节限位检查（在 Piper 接口层）
- 超时检测

## 下一步工作

1. 复制 `PiperInterface` 到 robodriver
2. 创建 Piper 的 Dora 集成
3. 编写 dataflow.yml
4. 进行端到端测试
5. 性能优化和调试

## 参考文件

- 原始实现：`/home/dora/teleop_pico/XRoboToolkit-Teleop-Sample-Python/`
- Piper 接口：`/home/dora/teleop_pico/.../hardware/interface/piper.py`
- 遥操控制器：`/home/dora/teleop_pico/.../hardware/piper_teleop_controller.py`
