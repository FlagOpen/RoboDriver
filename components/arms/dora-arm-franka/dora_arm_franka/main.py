import os
import time
import requests
import pyarrow as pa
from dora import Node
from pika.gripper import Gripper  # 导入 Pika Gripper 模块

# ---------------------- Pika Gripper 辅助类（修正参数问题）----------------------
class PikaGripperHelper:
    def __init__(self, port=None):
        self.port = port or os.getenv("PIKA_GRIPPER_PORT", "/dev/ttyUSB0")  # 优先环境变量
        self._gripper = None  # 复用夹爪连接

    def _ensure_connected(self):
        """确保夹爪已连接并启用电机（内部复用连接）"""
        if self._gripper is not None and getattr(self._gripper, "is_connected", False):
            return self._gripper

        print(f"连接 Pika Gripper（串口：{self.port}）...")
        g = Gripper(self.port)
        if not g.connect():
            print("Pika Gripper 连接失败！")
            return None
        if not g.enable():
            print("Pika Gripper 电机启用失败！")
            g.disconnect()
            return None
        print("Pika Gripper 就绪")
        self._gripper = g
        return self._gripper

    def get_gripper_distance(self):
        """获取夹爪当前张开距离（单位：毫米）- 修正：传入 angle=0 参数"""
        gripper = self._ensure_connected()
        if not gripper:
            return 0.0  # 失败返回默认值，不中断流程
        try:
            # 修正：给 get_distance() 传入 angle=0（默认模式，获取整体开合距离）
            # 若 SDK 要求其他值（如 1），可根据实际文档调整
            return gripper.get_distance(angle=0)
        except Exception as e:
            print(f"获取夹爪开合度失败：{e}")
            return 0.0

    def set_gripper_distance(self, target_dist):
        """设置夹爪目标张开距离（单位：毫米）"""
        gripper = self._ensure_connected()
        if not gripper:
            return False
        try:
            # 调用 Pika Gripper SDK 的设置方法（与原代码一致，无需修改）
            gripper.set_gripper_distance(target_dist)
            return True
        except Exception as e:
            print(f"设置夹爪开合度失败：{e}")
            return False

# ---------------------- 原有代码保持不变 ----------------------
def post(url, route, json=None):
    resp = requests.post(f"{url}/{route}", json=json)
    resp.raise_for_status()
    return resp 

def get_arm_data(url, gripper_helper):
    """统一获取机械臂关节、夹爪、位姿数据，返回字典格式"""
    arm_data = {
        "jointstate": None,
        "gripper": None,  # 格式：[当前距离(mm), 是否稳定夹持(0/1)]
        "pose": None,
        "success": False
    }

    try:
        # 获取关节角度
        joint_resp = requests.post(f"{url}getq", timeout=0.1)
        if joint_resp.status_code == 200:
            arm_data["jointstate"] = joint_resp.json()["q"]

        # 获取 Pika Gripper 开合度（已修正参数问题）
        gripper_dist = gripper_helper.get_gripper_distance()
        arm_data["gripper"] = [gripper_dist] 

        # 获取末端位姿
        pose_resp = requests.post(f"{url}getpos_euler", timeout=0.1)
        if pose_resp.status_code == 200:
            pose = pose_resp.json()["pose"]
            arm_data["pose"] = [pose[0], pose[1], pose[2], pose[3], pose[4], pose[5]]

        # 标记数据是否完整（包含夹爪数据）
        if all(v is not None for v in [arm_data["jointstate"], arm_data["gripper"], arm_data["pose"]]):
            arm_data["success"] = True

    except requests.exceptions.RequestException as e:
        print(f"机械臂API请求失败: {e}")

    return arm_data


def main():
    arm_url = os.getenv("url", "http://127.0.0.1:5000/")
    print(f"机械臂API地址: {arm_url}")

    # 初始化 Pika Gripper 辅助类
    gripper_helper = PikaGripperHelper()

    node = Node()
    ctrl_frame = 0

    # 事件循环
    for event in node:
        event_type = event["type"]

        if event_type == "INPUT":
            if "action" in event["id"]:
                pass

            if event["id"] == "action_joint":
                if ctrl_frame > 0:
                    continue
                try:
                    joint = event["value"].to_numpy()
                    goal_eef_quat = joint[:-1].astype(float)
                    gripper_val = float(joint[-1])  # 目标开合度（mm）
                    
                    # 1. 控制机械臂位姿（原有逻辑不变）
                    post(arm_url, "pose", {"arr": goal_eef_quat.tolist()})
                    # 2. 控制 Pika Gripper
                    gripper_helper.set_gripper_distance(gripper_val)
                except Exception as e:
                    print(f"执行 'action_joint' 失败: {e}")
            
            elif event["id"] == "action_joint_ctrl":        
                try:
                    joint = event["value"].to_numpy()
                    goal_eef_quat = joint[:-1]
                    gripper_val = joint[-1]  # 目标开合度（mm）
                    
                    # 1. 控制机械臂位姿（原有逻辑不变）
                    post(arm_url, "pose", {"arr": goal_eef_quat.tolist()})
                    # 2. 控制 Pika Gripper
                    gripper_helper.set_gripper_distance(gripper_val)
                except Exception as e:
                    print(f"执行 'action_joint_ctrl' 失败: {e}")

            elif event["id"] == "get_joint":
                # 传入夹爪辅助类，获取夹爪数据
                arm_data = get_arm_data(arm_url, gripper_helper)
                if arm_data["success"]:
                    combined_list = (
                        arm_data["jointstate"]
                        + arm_data["gripper"]  # 加入夹爪数据（距离+夹持状态）
                        + arm_data["pose"]
                    )
                    node.send_output(
                        "jointstate",
                        pa.array(combined_list, type=pa.float32()),
                        {"timestamp": time.time_ns()}
                    )
        elif event["id"] == "stop":
            print("收到停止指令，停止机械臂...")
            # 停止夹爪电机并断开连接
            if gripper_helper._gripper:
                gripper_helper._gripper.disable()
                gripper_helper._gripper.disconnect()

    print("Dora节点退出，清理资源...")
    # 退出时断开夹爪连接
    if gripper_helper._gripper:
        try:
            gripper_helper._gripper.disable()
            gripper_helper._gripper.disconnect()
        except Exception:
            pass

if __name__ == "__main__":
    main()
