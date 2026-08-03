import numpy as np
import cv2

# 引入我们刚刚写的注册器
from src.inference.registry import register_engine

from .beingh_service import BeingHInferenceClient

# 👇 神奇的装饰器，直接把这个类注册为 'beingh'
@register_engine("beingh")
class BeingHInferenceAdapter:
    def __init__(self, config):
            
        inf_cfg = config.get('inference', {})
        self.host = inf_cfg.get('host', '127.0.0.1')
        self.port = inf_cfg.get('port', 5555)
        
        # 👇 严格只读取专属配置，避免污染
        kwargs = inf_cfg.get('engine_params', {})
        self.chunk_index = kwargs.get('action_chunk_index', 0)
        
        print(f"🚀 [Plugin] 正在连接 BeingH 模型服务器 tcp://{self.host}:{self.port}...")
        self.client = BeingHInferenceClient(host=self.host, port=self.port)
        if not self.client.ping():
            raise ConnectionError("无法连接到 BeingH 推理服务器！")

    def _convert_image(self, img_data):
        """处理底层传来的图像字典，转为 NumPy"""
        if not img_data:
            return np.zeros((640, 480, 3), dtype=np.uint8)
            
        meta = img_data['metadata']
        raw_bytes = img_data['data']
        h, w = meta.get("height", 480), meta.get("width", 640)
        
        if meta.get("encoding") == "rgb8" or meta.get("format") == "raw":
            return np.frombuffer(raw_bytes, dtype=np.uint8).reshape(h, w, 3)
        else:
            frame = cv2.imdecode(np.frombuffer(raw_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def predict(self, raw_states: dict, instruction: str) -> dict:
        """
        核心接口：接收全量原始状态 -> 转换为模型所需 observations -> 推理 -> 解析并返回通用控制指令
        """
        # 1. 安全提取各部件状态
        l_joints = raw_states.get("follower_left_joints", {}).get("position", [0.0]*7)
        l_hand = raw_states.get("follower_left_hand", {}).get("position", [0.0]*10)
        
        # 解析 Pose (ROS 格式分解为 pos 和 ori)
        l_pose_dict = raw_states.get("follower_left_pose", {})
        l_pos = l_pose_dict.get("position", {"x":0.0, "y":0.0, "z":0.0})
        l_ori = l_pose_dict.get("orientation", {"x":0.0, "y":0.0, "z":0.0, "w":1.0})
        
        l_pos_arr = [l_pos['x'], l_pos['y'], l_pos['z']]
        l_ori_arr = [l_ori['x'], l_ori['y'], l_ori['z'], l_ori['w']] # 四元数
        
        top_img = self._convert_image(raw_states.get("image_top"))

        # 2. 严格对齐 LinkerBotLeftOnlyDataConfig 的 STATE_KEYS
        observations = {
            # 图像：从 (H, W, C) 变为 (1, H, W, C)
            "video.head": top_img[None, ...], 
            
            # 状态向量：从 (D,) 变为 (1, D)
            "state.left_arm_joint_positions": np.array(l_joints, dtype=np.float32)[None, :],
            "state.left_hand_qpos": np.array(l_hand, dtype=np.float32)[None, :],
            "state.left_arm_eef_position": np.array(l_pos_arr, dtype=np.float32)[None, :], 
            "state.left_eef_rotation": np.array(l_ori_arr, dtype=np.float32)[None, :], 
            
            "language.instruction": instruction, # 字符串通常不需要处理，服务端会自行 tokenize
        }
        
        # ========= [DEBUG 打印发送数据] =========
        print("\n\033[96m" + "="*50)
        print("📤 发送给 Being-H 的 Observations:")
        print(f"  - 图像形状: {observations['video.head'].shape}")
        print(f"  - 关节输入: {np.round(observations['state.left_arm_joint_positions'], 3).tolist()}")
        print(f"  - 末端位置: {np.round(observations['state.left_arm_eef_position'], 3).tolist()}")
        print(f"  - 灵巧手输入: {np.round(observations['state.left_hand_qpos'], 3).tolist()}")
        print("="*50 + "\033[0m")

        # 3. 发起推理
        result = self.client.get_action(observations)
        if not result:
            print("⚠️ Server returned empty result!")
            return [] # 注意这里返回空列表

        # 4. 解析结果
        l_arm_chunk = np.array(result.get("action.left_arm_joint_positions", []))
        l_hand_chunk = np.array(result.get("action.left_hand_qpos", []))
        
        # 去掉 Batch 维度 (1, 16, D) -> (16, D)
        if len(l_arm_chunk) > 0: l_arm_chunk = np.squeeze(l_arm_chunk)
        if len(l_hand_chunk) > 0: l_hand_chunk = np.squeeze(l_hand_chunk)

        print(f"🔍 Received raw action chunks: Arm={len(l_arm_chunk)} steps, Hand={len(l_hand_chunk)} steps")

        # 5. 将这 16 帧组装成一个列表
        action_chunk_list = []
        
        # 假设 arm 和 hand 的 chunk 长度一致
        chunk_size = min(len(l_arm_chunk), len(l_hand_chunk)) if len(l_arm_chunk) > 0 else 0
        
        for i in range(chunk_size):
            step_action = {}
            
            # --- 机械臂 ---
            step_action["follower_left_arm_control"] = {
                "position": l_arm_chunk[i].tolist(), 
                "follow": True
            }
            
            # --- 灵巧手 (映射 0-255) ---
            raw_hand_action = l_hand_chunk[i][:10] 
            mapped_hand = np.clip(raw_hand_action * 255.0, 0, 255)
            step_action["follower_left_hand_control"] = {
                "position": [float(int(x)) for x in mapped_hand],
            }
            
            action_chunk_list.append(step_action)

        print(f"✅ Parsed a chunk of {len(action_chunk_list)} actions.")
        return action_chunk_list  # 返回整个列表