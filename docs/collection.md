# RoboDriver 数据采集指南

本文档说明如何使用 RoboDriver 进行多轮数据采集。

## 📚 目录

- [系统架构](#系统架构)
- [前置条件](#前置条件)
- [启动流程](#启动流程)
- [数据采集操作](#数据采集操作)
- [数据管理](#数据管理)
- [故障排查](#故障排查)

---

## 🏗️ 系统架构

RoboDriver 数据采集系统由三个主要组件组成：

```
┌─────────────────┐      ┌──────────────────┐      ┌─────────────────────┐
│  Dora Dataflow  │ ───> │  RoboDriver      │ ───> │ RoboDriver-Server   │
│  (硬件驱动层)    │      │  (数据采集客户端) │      │ (Web控制台+存储)     │
└─────────────────┘      └──────────────────┘      └─────────────────────┘
     ↓                           ↓                           ↓
  机械臂+相机              实时数据流处理              Web界面控制+数据保存
```

**组件说明**：
- **Dora Dataflow**: 与硬件直接通信，采集相机图像和机械臂关节数据
- **RoboDriver**: 处理数据流，显示实时画面，与服务器通信
- **RoboDriver-Server**: 提供 Web 控制界面，管理数据采集任务，保存数据

---

## 📋 前置条件

### 1. 硬件连接

按照以下顺序连接硬件（**顺序很重要**）：

```bash
# 1. 断开所有 USB 设备
# 2. 插入顶部相机（RealSense 435 或其他）
# 3. 插入腕部相机
# 4. 插入主臂 USB（带扳机，使用 5V 电源）
# 5. 插入从臂 USB
```

验证设备连接：

```bash
# 检查相机设备
ls /dev/video*
# 应该看到: /dev/video0 ... /dev/video7

# 检查机械臂设备
ls /dev/ttyACM*
# 应该看到: /dev/ttyACM0 /dev/ttyACM1

# 赋予机械臂权限
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

### 2. 软件环境

确保已安装：
- RoboDriver 主程序
- RoboDriver-Server
- dora-rs-cli
- 对应机器人的驱动包（如 robodriver-robot-so102-aio-dora）

---

## 🚀 启动流程

数据采集需要启动三个服务，建议使用三个独立的终端窗口。

### 终端 1: 启动 Dora Dataflow

```bash
# 进入机器人的 dora 目录
cd ~/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora

# 清理之前的 dataflow（如果有）
dora destroy

# 启动 dora 守护进程
dora up

# 启动 dataflow
dora start dataflow.yml --uv
```

**成功标志**：
```
INFO    arm_so102_leader: daemon  node is ready
INFO    arm_so102_follower: daemon  node is ready
INFO    camera_top: daemon  node is ready
INFO    camera_wrist: daemon  node is ready
INFO    camera_top_dep: daemon  node is ready
INFO    daemon  all nodes are ready, starting dataflow
```

**保持此终端运行，不要关闭！**

### 终端 2: 启动 RoboDriver

```bash
# 进入 RoboDriver 主目录
cd ~/RoboDriver

# 激活虚拟环境
source .venv/bin/activate

# 启动 RoboDriver（替换为你的机器人类型）
robodriver-run --robot.type=so102_aio_dora
```

**成功标志**：
```
INFO     ✅ Successfully imported plugin: robodriver_robot_so102_aio_dora
INFO     成功连接到服务器
INFO     [连接成功] 所有设备已就绪:
           - 摄像头: image_top, image_wrist, image_top_dep
           - 主臂关节角度: shoulder_pan, shoulder_lift, ...
           - 从臂关节角度: shoulder_pan, shoulder_lift, ...
```

**应该会弹出 OpenCV 窗口显示实时相机画面**

**保持此终端运行，不要关闭！**

### 终端 3: 启动 RoboDriver-Server

```bash
# 进入 RoboDriver-Server 目录
cd ~/RoboDriver-Server  # 或你的服务器安装路径

# 启动服务器
python operating_platform_server_test.py
```

**成功标志**：
```
Server running on http://localhost:8088
```

**保持此终端运行，不要关闭！**

---

## 📊 数据采集操作

### 1. 打开 Web 控制界面

在浏览器中访问：
```
http://localhost:8088
```

### 2. 查看实时视频流

Web 界面应该显示：
- **image_top**: 顶部相机视图
- **image_wrist**: 腕部相机视图
- **image_top_dep**: 顶部深度相机视图

### 3. 开始第一轮采集

**步骤**：

1. **点击"开始采集"按钮**

2. **填写任务信息**：
   - 任务名称（task_name）：例如 `pick_cube`
   - 任务 ID（task_id）：自动生成或手动输入
   - 数据 ID（task_data_id）：自动递增

3. **等待倒计时**：
   - 系统会倒计时 3 秒
   - 准备好演示动作

4. **执行演示**：
   - 移动**主臂**（leader arm，带扳机的那个）
   - **从臂**（follower arm）会自动跟随
   - 系统自动记录：
     - 所有相机图像（30 FPS）
     - 主臂和从臂的关节角度
     - 时间戳

5. **完成采集**：
   - 演示完成后，点击"完成采集"按钮
   - 数据自动保存到本地

### 4. 采集多轮数据

重复步骤 3，每次采集会创建一个新的数据集：

```
第 1 轮: pick_cube_task123_data001
第 2 轮: pick_cube_task123_data002
第 3 轮: pick_cube_task123_data003
...
```

**建议**：
- 每个任务采集 10-50 轮数据
- 变化起始位置和物体摆放
- 包含成功和失败的案例

---

## 💾 数据管理

### 数据保存位置

数据默认保存在：
```
~/DoRobot/data/YYYYMMDD/user/任务名_任务ID/数据集ID/
```

**目录结构示例**：
```
~/DoRobot/data/
└── 20260303/                    # 日期
    └── user/                    # 用户数据
        └── pick_cube_task123/   # 任务目录
            ├── pick_cube_task123_data001/  # 第1轮数据
            │   ├── meta.json
            │   ├── episode_0000.parquet
            │   └── videos/
            ├── pick_cube_task123_data002/  # 第2轮数据
            └── pick_cube_task123_data003/  # 第3轮数据
```

### 查看采集的数据

```bash
# 查看所有数据
ls -lh ~/DoRobot/data/

# 查看今天的数据
ls -lh ~/DoRobot/data/$(date +%Y%m%d)/

# 查看特定任务的数据
ls -lh ~/DoRobot/data/20260303/user/pick_cube_task123/

# 查看数据集详情
cat ~/DoRobot/data/20260303/user/pick_cube_task123/pick_cube_task123_data001/meta.json
```

### 数据格式

每个数据集包含：
- **meta.json**: 元数据（机器人类型、FPS、相机配置等）
- **episode_XXXX.parquet**: 单条数据（图像、关节角度、时间戳）
- **videos/**: 视频文件（如果启用视频模式）

---

## 🔧 故障排查

### 问题 1: 无法连接到服务器

**错误信息**：
```
socketio.exceptions.ConnectionError: Cannot connect to host localhost:8088
```

**解决方案**：
1. 确认 RoboDriver-Server 正在运行
2. 检查端口 8088 是否被占用：
   ```bash
   lsof -i :8088
   ```
3. 重启 RoboDriver-Server

### 问题 2: OpenCV 窗口不显示

**原因**: 代码中 `cv2.imshow` 被注释

**解决方案**: 已在 `/home/dora/RoboDriver/robodriver/scripts/run.py` 中启用

### 问题 3: 相机画面黑屏或错误

**检查相机索引**：
```bash
# 测试相机
ffplay /dev/video4
ffplay /dev/video6
```

**修改配置**：
编辑 `dataflow.yml` 中的 `CAPTURE_PATH` 参数

### 问题 4: 机械臂无响应

**检查权限**：
```bash
sudo chmod 666 /dev/ttyACM0
sudo chmod 666 /dev/ttyACM1
```

**检查连接**：
```bash
# 运行诊断脚本
cd ~/RoboDriver/robodriver/robots/robodriver-robot-so102-aio-dora/dora
arm.venv/bin/python diagnose_motors.py
```

### 问题 5: 磁盘空间不足

**错误信息**：
```
存储空间不足,小于2GB,取消采集！
```

**解决方案**：
```bash
# 检查磁盘空间
df -h ~/DoRobot

# 清理旧数据
rm -rf ~/DoRobot/data/旧日期目录/
```

---

## 📝 最佳实践

### 数据采集建议

1. **采集前准备**：
   - 确保所有硬件连接稳定
   - 检查相机画面清晰
   - 测试机械臂响应

2. **采集过程**：
   - 动作要流畅自然
   - 避免突然停顿或抖动
   - 保持合理的速度

3. **数据质量**：
   - 每个任务至少 20 轮
   - 包含不同起始位置
   - 记录失败案例

4. **数据管理**：
   - 定期备份数据
   - 使用有意义的任务名称
   - 记录采集日志

### 停止服务

采集完成后，按以下顺序停止服务：

```bash
# 1. 在终端 2 按 Ctrl+C 停止 RoboDriver
# 2. 在终端 3 按 Ctrl+C 停止 RoboDriver-Server
# 3. 在终端 1 按 Ctrl+C 停止 Dora Dataflow
```

---

## 🔗 相关文档

- [RoboDriver 主文档](https://flagopen.github.io/RoboDriver-Doc)
- [机器人配置指南](./add_so102.md)
- [数据格式说明](https://flagopen.github.io/RoboDriver-Doc/docs/dataset)

---

## 📞 获取帮助

如遇问题，请：
1. 查看本文档的故障排查部分
2. 检查终端输出的错误信息
3. 访问 [GitHub Issues](https://github.com/FlagOpen/RoboDriver/issues)
