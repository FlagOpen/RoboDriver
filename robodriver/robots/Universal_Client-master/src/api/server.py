from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional

# 引入全新的工厂方法
from src.core.factory import create_connector
from src.core.base_connector import BaseRobotConnector

class ControlCommand(BaseModel):
    key: str                    # yaml 里 devices 下的 key，例如 "master_left"
    position: List[float]       # 目标角度列表
    names: Optional[List[str]] = None # 关节名字 (可选)

app = FastAPI(title="LinkerBot Edge API")

# 全局变量存放硬件抽象层实例
bot: BaseRobotConnector = None

@app.on_event("startup")
def startup_event():
    """
    FastAPI 启动时，根据 yaml 自动初始化 ROS 1 或 ROS 2 节点
    """
    global bot
    try:
        bot = create_connector("config/settings.yaml")
        bot.start() # 启动底层自旋线程和 ZMQ
        print("✅ Robot Connector attached to API Server")
    except Exception as e:
        print(f"❌ Failed to initialize Robot Connector: {e}")

@app.on_event("shutdown")
def shutdown_event():
    """优雅释放资源"""
    global bot
    if bot:
        bot.stop()
        print("🛑 Robot Connector stopped")

@app.get("/")
def health_check():
    return {"status": "online", "robot_connected": bot is not None}

@app.get("/state/{key}")
def get_robot_state(key: str):
    if not bot:
        raise HTTPException(status_code=503, detail="Robot not connected")
    
    data = bot.get_state(key)
    if data is None:
        raise HTTPException(status_code=404, detail=f"Key '{key}' not found or no data yet")
    return data

@app.post("/control")
def send_command(cmd: ControlCommand):
    if not bot:
        raise HTTPException(status_code=503, detail="Robot not connected")

    # 封装为 dict 传递给底层的 send_control
    command_data = {
        "position": cmd.position,
        "names": cmd.names if cmd.names else []
    }
    
    success = bot.send_control(cmd.key, command_data)
    
    if not success:
        raise HTTPException(status_code=400, detail="Command failed (check logs or key name)")
    
    return {"status": "executed", "target": cmd.key}