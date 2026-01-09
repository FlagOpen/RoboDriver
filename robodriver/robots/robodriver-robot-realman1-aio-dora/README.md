# robodriver-robot-realman1-aio-dora
## 快速开始
克隆代码仓库
```
git clone https://github.com/FlagOpen/RoboDriver.git
```

安装 Robodriver 环境
```
cd /path/to/your/RoboDriver
pip install uv
uv venv .venv -p 3.10
source .venv/bin/activate
uv pip install -e .[hardware]
```

安装 robodriver-robot-realman1-aio-dora 模块 环境
```
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman1-aio-dora
uv venv .venv -p 3.9
source .venv/bin/activate
uv pip install -e .
```

配置 dataflow.yml 文件
```
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman1-aio-dora
cd dora
```

打开 dataflow.yml 文件，修改其中的 VIRTUAL_ENV 路径，以及 DEVICE_SERIAL、ARM_IP 和 ARM_PORT 参数。
### 参数说明：

- VIRTUAL_ENV：realman dora 节点使用的虚拟环境路径
- DEVICE_SERIAL：RealSense 相机的序列号
- ARM_IP：机械臂的 IP 地址
- ARM_PORT：机械臂的端口号

## 开始数据采集
激活环境
```
cd /path/to/your/RoboDriver
source .venv/bin/activate
```

启动 Dora 服务
```
dora up
```

启动数据流
```
cd /path/to/your/RoboDriver/robodriver/robots/robodriver-robot-realman1-aio-dora
dora start dora/dataflow.yml --uv
```

启动 RoboDriver 主程序
```
cd /path/to/your/RoboDriver
source .venv/bin/activate
python robodriver/scripts/run.py \
  --robot.type=realman1_aio_dora 
```

## 问题修复
1. 视频回放功能重新运行失败：  
   编辑 `RoboDriver/robodriver/core/coordinator.py` 文件，将 `visual_worker(mode="distant")` 中的模式改为 `mode="local"`。

2. 启动时出现 OpenCV cvShowImage 错误（执行 `python robodriver/scripts/run.py --robot.type=realman1_aio_dora` 时）：  
   注释掉 `robodriver/scripts/run.py` 文件中的 `cv2.imshow(key, img)` 和 `cv2.waitKey(1)` 两行代码。

## 数据说明
RealMan 机械臂的数据由 Dora 节点进行传输。每个机械臂节点会发送**14 维数据信息**，具体组成如下：

- 关节角度：1-7 号关节的角度值（7 维）
- 夹爪状态：夹爪的开合程度
- 末端执行器位姿：末端欧拉角（3 维，分别表示绕 X/Y/Z 轴的旋转角度）
**如需修改数据信息，可调整 dataflow.yml 和 config.py 文件**

### 总结
1. 部署流程分为克隆代码、安装双环境（RoboDriver 主环境 + realman1-aio-dora 模块环境）、配置核心参数三步；
2. 启动采集需依次激活环境、启动 Dora 服务、运行数据流、启动主程序；
3. 已知两个常见问题的修复方案：修改视频回放模式、注释 OpenCV 显示代码；
4. 机械臂传输 14 维数据，核心维度包含关节角度、夹爪状态和末端位姿，可通过配置文件调整数据结构。