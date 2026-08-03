好的，我已经将添加 SO102 机器人的完整步骤总结并整理到了 `/home/dora/RoboDriver/docs/add_so102.md` 文档中。

以下是文档的具体内容：

```markdown
# 添加新机器人 SO102-AIO-Dora 到 RoboDriver

本文档说明如何将一个新的机器人实现（以SO102为例）添加到RoboDriver项目中。

## 📚 目录

- [核心原理](#核心原理)
- [前置条件](#前置条件)
- [步骤1：修改SO102代码](#步骤1修改so102代码)
- [步骤2：安装SO102包](#步骤2安装so102包)
- [步骤3：验证注册](#步骤3验证注册)
- [步骤4：配置Dora数据流](#步骤4配置dora数据流)
- [步骤5：启动测试](#步骤5启动测试)
- [故障排查](#故障排查)
- [检查清单](#检查清单)

---

## 🎯 核心原理

RoboDriver使用**自动发现机制**来加载机器人插件：

1. **包命名规则**：任何以 `robodriver_robot_` 开头的Python包都会被自动导入
2. **注册机制**：通过 `@RobotConfig.register_subclass("robot_name")` 注册机器人类型
3. **启动方式**：使用 `--robot.type=robot_name` 启动指定机器人

**关键文件**：
- `robodriver/utils/import_utils.py` - 自动发现和导入插件
- `config.py` - 注册机器人类型
- `robot.py` - 机器人主类实现
- `node.py` - Dora节点通信
- `dora/dataflow.yml` - Dora数据流配置

### 🔍 自动检测机制详解

**为什么新安装的机器人会被自动检测到？**

RoboDriver 采用标准的 **Python 插件系统**设计，每个机器人都是一个独立的 Python 包：

#### 1. 机器人作为 Python 包

```
robodriver-robot-so102-aio-dora/          # 项目目录（连字符）
├── pyproject.toml                         # 包定义文件
└── robodriver_robot_so102_aio_dora/      # Python 包（下划线）
    ├── __init__.py                        # 包入口
    ├── config.py                          # 配置类（含注册装饰器）
    ├── robot.py                           # 机器人实现
    └── node.py                            # Dora 节点
```

#### 2. 安装注册到 Python 系统

```bash
pip install -e /path/to/robodriver-robot-so102-aio-dora
```

执行后：
- Python 将 `robodriver_robot_so102_aio_dora` 注册到模块系统
- 包名被添加到可导入模块列表
- 可以通过 `import robodriver_robot_so102_aio_dora` 导入

#### 3. daemon 启动时自动扫描

当 RoboDriver 启动时，会调用 `register_third_party_devices()` 函数：

```python
# robodriver/utils/import_utils.py
def register_third_party_devices():
    # 扫描所有已安装的 Python 包
    for module_info in pkgutil.iter_modules():
        name = module_info.name
        # 匹配以 robodriver_robot_ 开头的包
        if name.startswith("robodriver_robot_"):
            # 自动导入包
            importlib.import_module(name)
```

**关键点**：
- `pkgutil.iter_modules()` 返回所有已安装的包（包括新安装的）
- 只要包名符合 `robodriver_robot_*` 模式，就会被自动导入
- 支持可编辑安装（`-e` 模式）

#### 4. 导入触发注册

当包被导入时，会执行 `config.py` 中的装饰器：

```python
# config.py
@RobotConfig.register_subclass("so102_aio_dora")  # 装饰器自动执行
@dataclass
class SO102AIODoraRobotConfig(RobotConfig):
    pass
```

装饰器将机器人配置注册到全局注册表：
- `"so102_aio_dora"` → `SO102AIODoraRobotConfig` 映射被添加
- 后续可以通过 `--robot.type=so102_aio_dora` 使用

#### 5. 工作流程总结

```
安装包 (pip install -e .)
    ↓
Python 注册包名到模块系统
    ↓
daemon 启动时调用 register_third_party_devices()
    ↓
pkgutil.iter_modules() 扫描所有包
    ↓
匹配 robodriver_robot_* 前缀
    ↓
importlib.import_module() 自动导入
    ↓
执行 __init__.py 和 config.py
    ↓
@RobotConfig.register_subclass() 装饰器执行
    ↓
机器人注册到全局注册表
    ↓
可以通过 --robot.type 使用
```

**重要提示**：
- ✅ 无需修改核心代码，只需安装包
- ✅ 无需重启系统，只需重启 daemon
- ✅ 支持热插拔，随时安装/卸载机器人包
- ⚠️ 包名必须以 `robodriver_robot_` 开头
- ⚠️ 必须有 `@RobotConfig.register_subclass()` 装饰器

---

## 📋 前置条件

假设你已经：
- ✅ 复制了SO101的代码到 `robodriver-robot-so102-aio-dora` 目录
- ✅ 了解SO102的硬件配置（电机、相机、USB端口等）
- ✅ 准备好SO102的硬件设备

---

## ✅ 步骤1：修改SO102代码

### 1.1 重命名Python包目录

```bash
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora

# 重命名主包
mv robodriver_robot_so101_aio_dora robodriver_robot_so102_aio_dora

# 重命名LeRobot兼容包
mv lerobot_robot_so101_aio_dora lerobot_robot_so102_aio_dora
```

### 1.2 修改 `pyproject.toml`

```toml
[project]
name = "robodriver_robot_so102_aio_dora"  # ← 改为so102
version = "0.1.0"
readme = "README.md"
requires-python = ">=3.8"
license = "Apache-2.0"
authors = [
    {name = "Your Name", email = "your.email@example.com"},
]
keywords = ["robotics", "lerobot", "so102"]  # ← 改为so102
dependencies = [
    "dora-rs",
    "logging_mp",
    "opencv-python",
]

[tool.setuptools.packages.find]
include = [
    "robodriver_robot_so102_aio_dora",  # ← 改为so102
    "lerobot_robot_so102_aio_dora"      # ← 改为so102
]
```

### 1.3 修改 `config.py`

**关键修改**：注册名称必须改为 `so102_aio_dora`

```python
from typing import Dict
from dataclasses import dataclass, field

from lerobot.robots.config import RobotConfig
from lerobot.cameras import CameraConfig
from lerobot.cameras.opencv import OpenCVCameraConfig
from lerobot.motors import Motor, MotorNormMode


@RobotConfig.register_subclass("so102_aio_dora")  # ← 改为so102
@dataclass
class SO102AIODoraRobotConfig(RobotConfig):  # ← 改为SO102
    use_degrees = True
    norm_mode_body = (
        MotorNormMode.DEGREES if use_degrees else MotorNormMode.RANGE_M100_100
    )

    # 根据你的SO102硬件配置修改电机和相机
    leader_motors: Dict[str, Motor] = field(...)
    follower_motors: Dict[str, Motor] = field(...)
    cameras: Dict[str, CameraConfig] = field(...)

    use_videos: bool = False
    microphones: Dict[str, int] = field(default_factory=lambda: {})
```

### 1.4 修改 `robot.py`

```python
from .config import SO102AIODoraRobotConfig  # ← 改为SO102
from .status import SO102AIODoraRobotStatus  # ← 改为SO102
from .node import SO102AIODoraRobotNode      # ← 改为SO102

class SO102AIODoraRobot(Robot):  # ← 改为SO102
    config_class = SO102AIODoraRobotConfig  # ← 改为SO102
    name = "so102_aio_dora"  # ← 改为so102

    def __init__(self, config: SO102AIODoraRobotConfig):  # ← 改为SO102
        super().__init__(config)
        # ... 其他代码 ...
        self.status = SO102AIODoraRobotStatus()  # ← 改为SO102
        self.robot_dora_node = SO102AIODoraRobotNode()  # ← 改为SO102
```

### 1.5 修改 `node.py`

```python
class SO102AIODoraRobotNode(DoraRobotNode):  # ← 改为SO102
    def __init__(self):
        self.node = Node("so102_aio_dora")  # ← 改为so102，必须与dataflow.yml中的id一致
        # ... 其他代码 ...
```

### 1.6 修改 `status.py`

```python
class SO102AIODoraRobotStatus:  # ← 改为SO102
    # ... 代码 ...
```

### 1.7 修改 `dora/dataflow.yml`

**关键修改**：节点ID必须与`node.py`中的Node名称一致

```yaml
nodes:
  - id: camera_top
    # ... 相机配置 ...

  - id: arm_so102_leader  # ← 改为so102
    build: pip install dora-arm-so101  # 如果SO102使用相同的驱动，保持不变
    path: dora-arm-so101
    inputs:
      get_joint: dora/timer/millis/33
    outputs:
      - joint_shoulder_pan
      # ... 其他输出 ...
    env:
      UV_PROJECT_ENVIRONMENT: /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/arm.venv
      PORT: /dev/ttyACM0
      ARM_NAME: SO102-leader  # ← 改为SO102
      ARM_ROLE: leader

  - id: arm_so102_follower  # ← 改为so102
    # ... 类似配置 ...
    env:
      ARM_NAME: SO102-follower  # ← 改为SO102

  - id: so102_aio_dora  # ← 改为so102，必须与node.py中的Node名称一致
    path: dynamic
    inputs:
      image_top: camera_top/image
      # ... 其他输入 ...
      leader_joint_shoulder_pan: arm_so102_leader/joint_shoulder_pan  # ← 改为so102
      follower_joint_shoulder_pan: arm_so102_follower/joint_shoulder_pan  # ← 改为so102
    outputs:
      - action_joint
```

---

## 📦 步骤2：安装SO102包

在RoboDriver主环境下安装新的插件包：

```bash
cd /home/dora/RoboDriver
source .venv/bin/activate

# 进入SO102目录
cd robodriver/robots/robodriver-robot-so102-aio-dora

# 以可编辑模式安装
uv pip install -e .
```

验证安装：

```bash
python -c "import robodriver_robot_so102_aio_dora; print('✅ SO102 package imported successfully')"
```

---

## 🔍 步骤3：验证注册

检查SO102是否已成功注册到RoboDriver：

```bash
cd /home/dora/RoboDriver
source .venv/bin/activate

# 查看所有已注册的机器人类型
python -c "
from lerobot.robots import RobotConfig
from robodriver.utils.import_utils import register_third_party_devices

register_third_party_devices()
print('Registered robot types:')
for robot_type in sorted(RobotConfig._choice_registry.keys()):
    print(f'  - {robot_type}')
"
```

**期望输出**：
```
✅ Successfully imported plugin: robodriver_robot_so102_aio_dora
Registered robot types:
  - so101_aio_dora
  - so102_aio_dora  ← 你的新机器人
  - ...
```

---

## ⚙️ 步骤4：配置Dora数据流

在运行之前，需要配置Dora环境并连接硬件。

```bash
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora

# 创建虚拟环境
uv venv camera.venv
uv venv arm.venv

# 安装依赖
dora build dataflow.yml --uv

# 连接硬件（按顺序）
# 1. 断开所有USB
# 2. 插入相机
# 3. 插入SO102主臂 → /dev/ttyACM0
# 4. 插入SO102从臂 → /dev/ttyACM1
# 5. 赋予权限
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# 启动Dora
dora up
dora start dataflow.yml --uv
```

---

## 🚀 步骤5：启动测试

一切准备就绪后，启动RoboDriver进行测试。

**新开一个终端**：

```bash
cd /home/dora/RoboDriver
source .venv/bin/activate

# 使用SO102启动
robodriver-run --robot.type=so102_aio_dora
```

**成功标志**：
```
[连接成功] 所有设备已就绪:
  - 摄像头: image_top, image_wrist, image_top_dep
  - 主臂关节角度: shoulder_pan, shoulder_lift, ...
  - 从臂关节角度: shoulder_pan, shoulder_lift, ...
总耗时: 2.34 秒
```

---

## 🔧 故障排查

### 问题1：找不到机器人类型

**错误**：
`ValueError: Invalid choice 'so102_aio_dora' for field 'type'`

**解决**：
1. 检查 `config.py` 中的注册名称：
   `@RobotConfig.register_subclass("so102_aio_dora")  # 必须正确`
2. 确认包已安装：
   `pip list | grep so102`
3. 检查包名是否以 `robodriver_robot_` 开头。

### 问题2：Dora节点连接失败

**错误**：
`等待主臂关节角度超时`

**解决**：
1. 检查 `node.py` 中的Node名称与 `dataflow.yml` 中的节点ID是否一致：
   ```python
   # node.py
   self.node = Node("so102_aio_dora")
   ```
   ```yaml
   # dataflow.yml
   - id: so102_aio_dora  # 必须一致
   ```
2. 检查USB设备是否正确连接：
   `ls /dev/ttyACM*`
3. 查看Dora日志：
   `cat dora/out/dora-coordinator.txt`

### 问题3：导入错误

**错误**：
`ModuleNotFoundError: No module named 'robodriver_robot_so102_aio_dora'`

**解决**：
1. 确认目录名和包名一致。
2. 重新安装：
   ```bash
   cd robodriver/robots/robodriver-robot-so102-aio-dora
   uv pip install -e .
   ```

---

## 📝 检查清单

在启动前，确认以下所有项：

- [ ] **目录名**：`robodriver-robot-so102-aio-dora`
- [ ] **Python包名**：`robodriver_robot_so102_aio_dora`
- [ ] **注册名称**：`@RobotConfig.register_subclass("so102_aio_dora")`
- [ ] **类名**：`SO102AIODoraRobot`, `SO102AIODoraRobotConfig`, `SO102AIODoraRobotNode`
- [ ] **Dora节点ID**：`so102_aio_dora`（与`node.py`中的Node名称一致）
- [ ] **包已安装**：`uv pip install -e .`
- [ ] **Dora环境已创建**：`camera.venv`, `arm.venv`
- [ ] **硬件已连接并赋权**
```

---

## 🔧 步骤6：创建 dora-arm-so102 包

如果 SO102 使用不同的硬件驱动（不能直接使用 dora-arm-so101），需要创建专用的 dora-arm-so102 包。

### 6.1 复制并重命名包

```bash
# 1. 复制 so101 包作为模板
cd /home/dora/RoboDriver/components/arms
cp -r dora-arm-so101 dora-arm-so102

# 2. 进入新包目录
cd dora-arm-so102
```

### 6.2 重命名 Python 包目录

```bash
# 重命名包目录
mv dora_arm_so101 dora_arm_so102
```

### 6.3 修改 pyproject.toml

编辑 `/home/dora/RoboDriver/components/arms/dora-arm-so102/pyproject.toml`：

```toml
[build-system]
requires = ["uv>=0.1.0", "setuptools>=61", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "dora-arm-so102"  # ← 改为 so102
version = "0.0.2"
description = ""
authors = [
    {name = "Xiang Yang",email = "ryu-yang@qq.com"}
]
readme = "README.md"
requires-python = ">=3.8,<3.14"
dependencies = [
    "dora-rs (>=0.3.11,<0.4.0)",
    "feetech-servo-sdk>=1.0.0",
    "numpy",
    "pyarrow",
    "deepdiff",
    "tqdm",
    "pygame",
    "draccus",
    "pynput"
]

[project.scripts]
dora-arm-so102 = "dora_arm_so102.main:main"  # ← 改为 so102

[tool.setuptools.packages.find]
where = ["."]
include = ["dora_arm_so102"]  # ← 改为 so102
exclude = []
```

### 6.4 安装新包到虚拟环境

```bash
# 进入 SO102 的 dora 目录
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora

# 安装 dora-arm-so102 包到 arm.venv
uv pip install --python arm.venv/bin/python /home/dora/RoboDriver/components/arms/dora-arm-so102
```

### 6.5 验证安装

```bash
# 检查可执行文件是否创建
ls -la arm.venv/bin/ | grep dora-arm
```

**期望输出**：
```
-rwxrwxr-x 1 dora dora  381  dora-arm-so102
```

### 6.6 更新 dataflow.yml

确保 `dataflow.yml` 中使用正确的包名：

```yaml
  - id: arm_so102_leader
    build: pip install dora-arm-so102  # ← 使用 so102
    path: dora-arm-so102               # ← 使用 so102
    # ... 其他配置 ...

  - id: arm_so102_follower
    build: pip install dora-arm-so102  # ← 使用 so102
    path: dora-arm-so102               # ← 使用 so102
    # ... 其他配置 ...
```

### 6.7 测试启动

```bash
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora
dora start dataflow.yml --uv
```

**成功标志**：
- 不再出现 `Failed to spawn: dora-arm-so102` 错误
- arm_so102_leader 和 arm_so102_follower 节点正常启动

---

## 📝 完整检查清单（更新版）

在启动前，确认以下所有项：

- [ ] **目录名**：`robodriver-robot-so102-aio-dora`
- [ ] **Python包名**：`robodriver_robot_so102_aio_dora`
- [ ] **注册名称**：`@RobotConfig.register_subclass("so102_aio_dora")`
- [ ] **类名**：`SO102AIODoraRobot`, `SO102AIODoraRobotConfig`, `SO102AIODoraRobotNode`
- [ ] **Dora节点ID**：`so102_aio_dora`（与`node.py`中的Node名称一致）
- [ ] **包已安装**：`uv pip install -e .`
- [ ] **Dora环境已创建**：`camera.venv`, `arm.venv`
- [ ] **dora-arm-so102 包已创建并安装**
- [ ] **硬件已连接并赋权**
```

---

## 🐛 常见错误及修复步骤（实战记录）

### 错误1：ModuleNotFoundError: No module named 'dora_arm_so102.motors'

**错误信息**：
```
ModuleNotFoundError: No module named 'dora_arm_so102.motors'
```

**原因**：`pyproject.toml` 中的包配置不完整，子包没有被包含。

**修复步骤**：

1. 修改 `/home/dora/RoboDriver/components/arms/dora-arm-so102/pyproject.toml`：

```toml
[tool.setuptools.packages.find]
where = ["."]
include = ["dora_arm_so102*"]  # ← 添加通配符 * 以包含所有子包
exclude = []
```

2. 强制重新安装包：

```bash
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora
uv pip install --python arm.venv/bin/python --force-reinstall /home/dora/RoboDriver/components/arms/dora-arm-so102
```

3. 验证安装：

```bash
arm.venv/bin/python -c "from dora_arm_so102.motors.feetech import FeetechMotorsBus; print('✅ motors 模块导入成功')"
```

---

### 错误2：FileNotFoundError: 校准文件路径不存在

**错误信息**：
```
FileNotFoundError: 校准文件路径不存在: /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/.calibration/SO102-leader.json
```

**原因**：缺少 SO102 的校准文件。

**修复步骤**：

```bash
# 从 SO101 复制校准文件作为初始配置
cp /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so101-aio-dora/dora/.calibration/SO101-leader.json \
   /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/.calibration/SO102-leader.json

cp /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so101-aio-dora/dora/.calibration/SO101-follower.json \
   /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/.calibration/SO102-follower.json

# 验证文件已创建
ls -la /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/.calibration/
```

**注意**：如果 SO102 的电机配置与 SO101 不同，需要修改校准文件中的电机 ID 和参数。

---

### 错误3：PermissionError: Permission denied: '/dev/ttyACM1'

**错误信息**：
```
PermissionError: [Errno 13] Permission denied: '/dev/ttyACM1'
```

**原因**：USB 端口没有读写权限。

**修复步骤**：

```bash
# 赋予 USB 端口权限
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1

# 验证权限
ls -la /dev/ttyACM*
```

**永久解决方案**（可选）：

```bash
# 将用户添加到 dialout 组
sudo usermod -a -G dialout $USER

# 注销并重新登录后生效
```

---

### 错误4：ConnectionError: Failed to write 'Torque_Enable' on id_=3

**错误信息**：
```
ConnectionError: Failed to write 'Torque_Enable' on id_=3 with '0' after 1 tries. [TxRxResult] Incorrect status packet!
```

**原因**：硬件通信失败，可能是：
- SO102 的电机 ID 配置与 SO101 不同
- 硬件没有正确连接或供电
- 校准文件中的电机配置不匹配

**诊断步骤**：

```bash
# 1. 检查 USB 设备是否正确连接
ls -la /dev/ttyACM*

# 2. 查看校准文件中的电机 ID 配置
cat /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/.calibration/SO102-leader.json
cat /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora/.calibration/SO102-follower.json

# 3. 检查硬件供电和连接
# - 确保机械臂已通电
# - 确保 USB 线缆连接稳定
# - 尝试重新插拔 USB
```

**修复步骤**：

如果 SO102 的电机 ID 与 SO101 不同，需要修改校准文件。例如：

```json
{
  "motors": {
    "shoulder_pan": {
      "id": 1,  // ← 根据实际硬件修改
      "offset": 0,
      "orientation": "direct"
    },
    "shoulder_lift": {
      "id": 2,
      "offset": 0,
      "orientation": "direct"
    }
    // ... 其他电机配置
  }
}
```

---

## 🔍 调试技巧

### 1. 查看 Dora 日志

```bash
# 查看最近一次运行的日志
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora
ls -lt out/ | head -5

# 查看特定节点的日志
cat out/<dataflow-id>/dora-coordinator.txt
cat out/<dataflow-id>/arm_so102_leader.txt
```

### 2. 测试单个组件

```bash
# 测试 arm 包是否能正常导入
cd /home/dora/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora
arm.venv/bin/python -c "from dora_arm_so102.main import main; print('✅ 导入成功')"

# 测试相机
camera.venv/bin/python -c "import cv2; print('✅ OpenCV 可用')"
```

### 3. 检查包安装

```bash
# 查看已安装的包
arm.venv/bin/pip list | grep dora
camera.venv/bin/pip list | grep dora

# 查看包的安装位置
arm.venv/bin/pip show dora-arm-so102
```

---

## 📋 完整工作流程总结

1. **创建 dora-arm-so102 包**
   - 复制 dora-arm-so101 → dora-arm-so102
   - 重命名 Python 包目录
   - 修改 pyproject.toml（包名、脚本入口点、packages.find）
   - 安装到虚拟环境

2. **准备校准文件**
   - 从 SO101 复制校准文件
   - 根据 SO102 硬件修改电机 ID 和参数

3. **配置权限**
   - 赋予 USB 端口读写权限
   - 或将用户添加到 dialout 组

4. **测试启动**
   - 启动 dora dataflow
   - 查看日志排查问题
   - 根据错误信息逐步修复

5. **验证功能**
   - 确认所有节点正常启动
   - 测试机械臂控制
   - 测试相机采集
```
