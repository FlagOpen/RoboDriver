import time
from collections import deque 
from src.core.factory import create_connector
from src.inference.factory import create_inference_engine

class CloudInferenceAgent:
    def __init__(self, config):
        self.config = config
        
        # 1. 启动硬件抽象层 (底层数据采集与下发)
        self.robot = create_connector("config/settings.yaml")
        self.robot.start()
        print("✅ Robot Connector attached.")

        # 2. 启动推理引擎 (这里就是调用了刚才的工厂模式，根据 YAML 自动生成 BeingH 插件)
        self.engine = create_inference_engine(config)
        print(f"✅ Inference Engine initialized.")

        self.fps = config.get('app', {}).get('fps', 10)
        self.sleep_time = 1.0 / self.fps
        
        # 你的语言指令
        self.current_instruction = "pick up the red apple on the table"
        self.action_queue = deque()
        self.replan_threshold = 6 # 剩余多少帧就开始重新规划
    def run(self):
        print("🤖 VLA Control Loop is running...")
        try:
            while True:
                step_start = time.time()
                self._step()
                
                # 严格控制控制频率 (例如 10Hz)
                elapsed = time.time() - step_start
                if elapsed < self.sleep_time:
                    time.sleep(self.sleep_time - elapsed)
        except KeyboardInterrupt:
            print("\n🛑 Agent stopped by user.")
        finally:
            self.robot.stop()

    def _step(self):
        # 1. 只有在队列快空了的时候，我们才需要去取图像并调用大模型（节省算力和网络带宽）
        if len(self.action_queue) <= self.replan_threshold:
            raw_states = {
                "follower_left_joints": self.robot.get_state("follower_left_joints"),
                "follower_left_pose": self.robot.get_state("follower_left_pose"),
                "follower_left_hand": self.robot.get_state("follower_left_hand"),
                "image_top": self.robot.get_image("image_top")
            }

            if raw_states["follower_left_joints"]:
                try:
                    # 这时候拿到的不再是一个字典，而是一个包含了 16 个字典的列表
                    actions_chunk = self.engine.predict(raw_states, self.current_instruction)
                    
                    if actions_chunk:
                        # [Receding Horizon 策略]：直接清空旧的剩余动作，用大模型最新的视觉预测覆盖
                        self.action_queue.clear() 
                        self.action_queue.extend(actions_chunk)
                except Exception as e:
                    print(f"⚠️ Inference error: {e}")

        # 2. 如果队列里有动作，就开始出队并发送给硬件
        if len(self.action_queue) > 0:
            # 弹出最早的一个动作
            current_action = self.action_queue.popleft() 
            
            # 分发控制指令 (例如 left_arm 和 left_hand)
            for device_key, command_data in current_action.items():
                self.robot.send_control(device_key, command_data)
                
            print(f"⚡ [Execution] Executed 1 step. Queue remaining: {len(self.action_queue)}")