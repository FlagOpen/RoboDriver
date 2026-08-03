# RoboDriver 与 RoboServer 通信机制

## 目录
- [架构概述](#架构概述)
- [通信组件](#通信组件)
- [通信协议](#通信协议)
- [数据流向](#数据流向)
- [API接口](#api接口)
- [消息类型](#消息类型)

---

## 架构概述

RoboDriver 作为**客户端**，通过网络与 RoboServer（服务端）进行通信。通信采用**混合模式**：
- **Socket.IO** (WebSocket) - 用于实时双向通信（心跳、命令下发）
- **HTTP REST API** - 用于数据上传（视频流、状态信息）

### 整体架构图

```
┌─────────────────────────────────────────────────────────────┐
│  RoboDriver (客户端)                                         │
│                                                              │
│  ┌──────────────┐         ┌──────────────┐                 │
│  │   Daemon     │         │   Monitor    │                 │
│  │  (机器人控制) │         │  (状态监控)   │                 │
│  └──────┬───────┘         └──────┬───────┘                 │
│         │                        │                          │
│         └────────┬───────────────┘                          │
│                  │                                          │
│         ┌────────▼────────┐                                 │
│         │  Coordinator    │                                 │
│         │  (协调器)        │                                 │
│         │                 │                                 │
│         │ - Socket.IO     │ ◄─── 实时双向通信               │
│         │ - HTTP Client   │ ◄─── 数据上传                   │
│         └────────┬────────┘                                 │
└──────────────────┼─────────────────────────────────────────┘
                   │
                   │ 网络通信
                   │
┌──────────────────▼─────────────────────────────────────────┐
│  RoboServer (服务端)                                         │
│  默认地址: http://localhost:8088                             │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  Socket.IO Server                                    │  │
│  │  - 接收心跳 (HEARTBEAT)                               │  │
│  │  - 发送命令 (robot_command)                           │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │  HTTP REST API                                       │  │
│  │  - /robot/stream_info (摄像头信息)                    │  │
│  │  - /robot/update_stream/:id (视频流)                  │  │
│  │  - /robot/response (命令响应)                         │  │
│  │  - /robot/update_machine_information (设备状态)       │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

---

## 通信组件

### 1. Daemon (机器人守护进程)

**文件**: `robodriver/robots/daemon.py`

**职责**:
- 管理机器人实例的生命周期
- 采集观测数据 (observation)
- 发送动作指令 (action)
- 维护机器人状态 (status)

**关键方法**:
```python
class Daemon:
    def __init__(self, config: RobotConfig, fps: int = None)
    def start(self)                    # 连接机器人
    def update(self)                   # 主循环：采集数据、发送指令
    def get_observation(self)          # 获取观测数据
    def get_status(self)               # 获取机器人状态
    def set_pre_action(self, action)   # 设置待执行动作
```

### 2. Monitor (状态监控器)

**文件**: `robodriver/core/monitor.py`

**职责**:
- 定期向服务器上报机器人状态
- 使用 HTTP POST 请求

**通信方式**:
```python
# 每60秒发送一次设备信息
URL: http://localhost:8088/robot/update_machine_information
Method: POST
Content-Type: application/json
Body: {机器人状态JSON}
```

### 3. Coordinator (协调器)

**文件**: `robodriver/core/coordinator.py`

**职责**:
- 管理与服务器的所有通信
- 处理服务器下发的命令
- 上传视频流和数据
- 协调录制、回放等任务

**核心属性**:
```python
class Coordinator:
    server_url = "http://localhost:8088"  # 服务器地址
    sio: socketio.AsyncClient             # Socket.IO客户端
    session: aiohttp.ClientSession        # HTTP客户端
    daemon: Daemon                        # 机器人守护进程
    teleop: Teleoperator                  # 遥操作器
```

---

## 通信协议

### 1. Socket.IO 实时通信

#### 客户端 → 服务器

**心跳消息** (每2秒发送一次)
```python
Event: "HEARTBEAT"
Data: None
```

#### 服务器 → 客户端

**命令消息**
```python
Event: "robot_command"
Data: {
    "cmd": "命令类型",
    "msg": {命令参数}
}
```

**心跳响应**
```python
Event: "HEARTBEAT_RESPONSE"
Data: {响应数据}
```

### 2. HTTP REST API

#### 上传摄像头信息
```http
POST /robot/stream_info
Content-Type: application/json

{
    "stream_count": 2,
    "streams": [
        {"id": 1, "name": "image_top"},
        {"id": 2, "name": "image_wrist"}
    ]
}
```

#### 上传视频流
```http
POST /robot/update_stream/:camera_id
Content-Type: image/jpeg

<JPEG图像二进制数据>
```

#### 命令响应
```http
POST /robot/response
Content-Type: application/json

{
    "cmd": "start_collection",
    "msg": "success",
    "data": {额外数据}
}
```

#### 上传设备状态
```http
POST /robot/update_machine_information
Content-Type: application/json

{机器人状态JSON}
```

---

## 数据流向

### 启动流程

```
1. 应用启动 (run.py)
   ↓
2. 创建 Daemon (连接机器人)
   ↓
3. 创建 Monitor (启动状态监控线程)
   ↓
4. 创建 Coordinator (连接服务器)
   ↓
5. Coordinator.start()
   ├─ 连接 Socket.IO
   ├─ 启动心跳循环
   └─ 注册命令回调
   ↓
6. 上传摄像头信息到服务器
   ↓
7. 进入主循环
   ├─ Daemon.update() (采集数据)
   ├─ Monitor 定期上报状态
   └─ Coordinator 处理服务器命令
```

### 主循环数据流

```
┌─────────────────────────────────────────────────────────┐
│  主循环 (run.py)                                         │
│                                                          │
│  while True:                                             │
│      daemon.update()           ◄─── 采集观测数据         │
│      observation = daemon.get_observation()              │
│      action = teleop.get_action()                        │
│      daemon.set_pre_action(action)                       │
│      daemon.set_obs_action(action)                       │
└─────────────────────────────────────────────────────────┘
         │                              │
         │ 观测数据                      │ 动作指令
         ↓                              ↓
┌──────────────────┐          ┌──────────────────┐
│  Coordinator     │          │  Robot Hardware  │
│  (可选上传视频流) │          │  (执行动作)       │
└──────────────────┘          └──────────────────┘
```

---

## API接口

### Coordinator 主要方法

#### 1. 启动与停止
```python
async def start(self)
    # 连接到服务器
    # 启动心跳循环

async def stop(self)
    # 断开连接
    # 清理资源
```

#### 2. 心跳管理
```python
async def send_heartbeat_loop(self)
    # 每2秒发送一次心跳
    while self.running:
        await self.sio.emit("HEARTBEAT")
        await asyncio.sleep(2)
```

#### 3. 命令响应
```python
async def send_response(self, cmd, msg, data=None)
    # 向服务器发送命令执行结果
    POST /robot/response
    {
        "cmd": cmd,
        "msg": msg,
        "data": data
    }
```

#### 4. 视频流上传
```python
async def update_stream_async(self, name, frame)
    # 将图像编码为JPEG
    # 上传到服务器
    POST /robot/update_stream/{camera_id}
```

#### 5. 摄像头信息同步
```python
def stream_info(self, info: Dict[str, int])
    # 更新本地摄像头信息

async def update_stream_info_to_server(self)
    # 同步摄像头信息到服务器
    POST /robot/stream_info
```

---

## 消息类型

### 服务器命令 (robot_command)

#### 1. 查询视频流列表
```json
{
    "cmd": "video_list"
}
```

**响应**:
```json
{
    "stream_count": 2,
    "streams": [
        {"id": 1, "name": "image_top"},
        {"id": 2, "name": "image_wrist"}
    ]
}
```

#### 2. 开始数据采集
```json
{
    "cmd": "start_collection",
    "msg": {
        "task_id": "task_001",
        "task_name": "pick_apple",
        "task_data_id": "data_001",
        "countdown_seconds": 3
    }
}
```

**处理流程**:
1. 检查磁盘空间（至少2GB）
2. 检查是否正在回放（不能同时进行）
3. 停止之前的录制（如果有）
4. 创建数据集目录
5. 初始化 Record 对象
6. 倒计时后开始录制

**响应**:
```json
{
    "cmd": "start_collection",
    "msg": "success"
}
```

#### 3. 完成数据采集
```json
{
    "cmd": "finish_collection"
}
```

**处理流程**:
1. 停止录制
2. 保存数据到磁盘
3. 返回保存结果

**响应**:
```json
{
    "cmd": "finish_collection",
    "msg": "success",
    "data": {
        "episodes": 10,
        "frames": 3000,
        "path": "/path/to/dataset"
    }
}
```

#### 4. 丢弃数据采集
```json
{
    "cmd": "discard_collection"
}
```

**处理流程**:
1. 停止录制
2. 删除临时数据
3. 不保存到磁盘

**响应**:
```json
{
    "cmd": "discard_collection",
    "msg": "success"
}
```

#### 5. 开始数据回放
```json
{
    "cmd": "start_replay",
    "msg": {
        "repo_id": "dataset_001",
        "episode_index": 0
    }
}
```

**处理流程**:
1. 检查是否正在录制（不能同时进行）
2. 加载数据集
3. 开始回放指定episode

**响应**:
```json
{
    "cmd": "start_replay",
    "msg": "success"
}
```

#### 6. 停止数据回放
```json
{
    "cmd": "stop_replay"
}
```

**响应**:
```json
{
    "cmd": "stop_replay",
    "msg": "success"
}
```

---

## 通信时序图

### 启动连接时序

```
RoboDriver                    RoboServer
    |                              |
    |--- Socket.IO Connect ------->|
    |<---- connect event ----------|
    |                              |
    |--- HEARTBEAT --------------->|
    |<--- HEARTBEAT_RESPONSE ------|
    |                              |
    |--- POST /robot/stream_info ->|
    |<--- 200 OK ------------------|
    |                              |
    |--- POST /robot/update_machine_information ->|
    |<--- 200 OK ------------------|
```

### 数据采集时序

```
RoboDriver                    RoboServer
    |                              |
    |<--- robot_command -----------|
    |     {cmd: "start_collection"}|
    |                              |
    |--- POST /robot/response ---->|
    |     {msg: "success"}         |
    |                              |
    |=== 开始录制 ===              |
    |                              |
    |--- POST /robot/update_stream ->| (持续)
    |<--- 200 OK ------------------|
    |                              |
    |<--- robot_command -----------|
    |     {cmd: "finish_collection"}|
    |                              |
    |=== 停止录制并保存 ===        |
    |                              |
    |--- POST /robot/response ---->|
    |     {msg: "success", data: {...}}|
```

---

## 配置说明

### 服务器地址配置

默认服务器地址在 `Coordinator` 中硬编码：
```python
server_url = "http://localhost:8088"
```

### 心跳间隔配置

```python
heartbeat_interval = 2  # 秒
```

### 状态上报间隔配置

```python
interval_seconds = 60  # Monitor每60秒上报一次
```

---

## 错误处理

### 1. 连接失败
```python
# Socket.IO 连接失败会触发 disconnect 事件
async def __on_disconnect_handle(self):
    logger.info("与服务器断开连接")
```

### 2. 请求超时
```python
# HTTP 请求设置超时
timeout=aiohttp.ClientTimeout(total=2)  # 2秒超时
```

### 3. 命令执行失败
```python
# 返回失败响应
await self.send_response("start_collection", "fail")
```

---

## 总结

### 通信特点

| 特性 | 说明 |
|------|------|
| **双向通信** | Socket.IO 支持服务器主动推送命令 |
| **异步处理** | 使用 asyncio 和 aiohttp 实现高性能异步通信 |
| **混合协议** | Socket.IO (实时) + HTTP (数据上传) |
| **容错机制** | 超时重试、错误日志、状态检查 |
| **线程安全** | 使用锁保护共享数据 |

### 关键文件

- `robodriver/core/coordinator.py` - 通信协调器（核心）
- `robodriver/core/monitor.py` - 状态监控器
- `robodriver/robots/daemon.py` - 机器人守护进程
- `robodriver/scripts/run.py` - 主程序入口

### 扩展建议

1. **配置化服务器地址**: 通过环境变量或配置文件设置
2. **重连机制**: Socket.IO 断线自动重连
3. **消息队列**: 处理网络抖动时的消息缓存
4. **压缩传输**: 视频流使用更高效的压缩算法
5. **安全认证**: 添加 Token 或证书认证机制
