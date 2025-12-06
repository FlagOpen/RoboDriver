import os
import json
import numpy as np
import cv2
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_episode_structure(components, base_dir="./episodes", episode_id=0, n_frames=5, create_files=True):
    """
    根据组件配置生成符合 lerobot 格式的 episode 数据结构（data 按 group 分类呈现）。
    
    Args:
        components: dict, 需包含 arms 的 group 信息，e.g.
            {
                "cameras": {
                    "image_top_left": ["image"],
                    "image_top_right": ["image"]
                },
                "arms": {
                    "robot_arm_main_left_joint": {"dof": 7, "outputs": ["joint_positions"], "group": "action"},
                    "robot_arm_main_right_joint": {"dof": 7, "outputs": ["joint_positions"], "group": "action"},
                    "robot_arm_follower_left_joint": {"dof": 7, "outputs": ["joint_positions"], "group": "observation"},
                    "robot_arm_follower_right_joint": {"dof": 7, "outputs": ["joint_positions"], "group": "observation"}
                }
            }
        base_dir: 存放所有 episode 的目录
        episode_id: 当前 episode 编号
        n_frames: 每个 episode 的帧数
        create_files: 是否实际创建文件（False 时仅打印结构）
    """
    ep_dir = os.path.join(base_dir, f"episode_{episode_id:06d}")
    images_dir = os.path.join(ep_dir, "images")
    data_dir = os.path.join(ep_dir, "data")

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(data_dir, exist_ok=True)

    # === 生成相机结构（匹配目标效果中的相机目录名）===
    cameras = components.get("cameras", {})
    for cam_name, outputs in cameras.items():
        # 相机目录名直接用 cam_name（如 "image_top_left"），无需拼接 output
        folder = os.path.join(images_dir, cam_name)
        os.makedirs(folder, exist_ok=True)
        if create_files:
            for i in range(1, n_frames + 1):
                img = np.full((240, 320, 3), 120, np.uint8)
                text = f"{cam_name} frame {i}"
                cv2.putText(img, text, (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
                ext = ".png" if "depth" in cam_name else ".jpg"
                fname = os.path.join(folder, f"frame_{i:06d}{ext}")
                cv2.imwrite(fname, img)

    # === 生成数据结构（按 group 分类：observation/action）===
    arms = components.get("arms", {})
    
    # 1. 按 group 分组（action/observation）
    grouped_arms = {
        "action": [],
        "observation": []
    }
    for arm_name, arm_info in arms.items():
        # 获取 arm 所属的 group，默认归为 observation
        group = arm_info.get("group", "observation").lower()
        # 修正可能的拼写错误
        if group not in grouped_arms:
            group = "observation"
        grouped_arms[group].append((arm_name, arm_info))
    
    # 2. 计算每个 group 的总自由度（匹配目标效果的 14 维）
    group_total_dof = {
        "action": sum(arm_info["dof"] for _, arm_info in grouped_arms["action"]),
        "observation": sum(arm_info["dof"] for _, arm_info in grouped_arms["observation"])
    }
    
    # 3. 生成 observation.npy 和 actions.npy（按 group 聚合数据）
    if create_files:
        # 生成 observation 数据（所有 observation 组 arm 合并为 14 维）
        obs_data = np.zeros((n_frames, group_total_dof["observation"]), dtype=np.float32)
        np.save(os.path.join(data_dir, "observations.npy"), obs_data)
        # 生成 action 数据（所有 action 组 arm 合并为 14 维）
        action_data = np.zeros((n_frames, group_total_dof["action"]), dtype=np.float32)
        np.save(os.path.join(data_dir, "actions.npy"), action_data)

    # 打印最终结构（完全匹配目标效果）
    print_episode_structure(ep_dir, cameras, grouped_arms, group_total_dof)

    return ep_dir

def parse_robot_config_to_episode_components(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    components = {"cameras": {}, "arms": {}}

    for idx, comp in enumerate(config.get("components", [])):
        comp_type = comp["type"]
        params = comp.get("params", {})
        outputs = params.get("output", [])
        comp_id = comp.get("id", f"comp_{idx}")

        # === 相机组件解析（匹配目标效果中的相机名称）===
        if comp_type.startswith("camera/"):
            # 相机名称直接用 comp_id（如 "image_top_left"）， outputs 取第一个即可
            if outputs:
                cam_name = comp_id  # 确保 cam_name 是 "image_top_left" 这类格式
                components["cameras"][cam_name] = outputs

        # === 机械臂组件解析（关键：提取 group 信息，匹配 arm 名称）===
        elif comp_type.startswith("arm/") or comp_type.startswith("robot_arm") or comp_type == "follower_motors":
            # 关键：arm 名称直接用 comp_id（如 "main_left_joint"），前缀统一为 "robot_arm_"
            arm_name = f"robot_arm_{comp_id}"  # 最终名称：robot_arm_main_left_joint
            outputs_info_all = comp.get("outputs_info", {})

            if len(outputs_info_all) > 0:
                first_key = list(outputs_info_all.keys())[0]
                outputs_info = outputs_info_all[first_key]
            else:
                outputs_info = {}

            motors = outputs_info.get("motors", {})
            dof = len(motors) if motors else 6

            # 关键：从配置中提取 group 信息（对应文件一中 DDSMotorsBusConfig 的 group 字段）
            # 假设 group 存储在 outputs_info 或 params 中，根据实际配置调整
            group = outputs_info.get("group") or params.get("group") or "observation"
            # 修正拼写错误
            if group.lower() in ["obeservatin", "observatin"]:
                group = "observation"

            components["arms"][arm_name] = {
                "dof": dof,
                "outputs": outputs,
                "group": group.lower()  # 统一转为小写
            }

    return components

def print_episode_structure(ep_dir, cameras, grouped_arms, group_total_dof):
    print(f"\n{os.path.basename(ep_dir)}/")
    print("├── images/")
    for cam_name in cameras.keys():
        print(f"│   ├── {cam_name}/")
        print(f"│   │   ├── frame_000000.jpg")
        print(f"│   │   ├── frame_000001.jpg")
        print(f"│   │   └── ...")
    print("│")
    print("└── data/")
    # 打印 observation 及其包含的 arm
    print(f"    ├── observations   # shape=(1, {group_total_dof['observation']})")
    for arm_name, arm_info in grouped_arms["observation"]:
        print(f"    │   ├── {arm_name}/")
        for i in range(min(3, arm_info['dof'])):
            print(f"    │   │   ├── joint_{i}")
        if arm_info['dof'] > 3:
            print(f"    │   │   └── ...")
    # 打印 action 及其包含的 arm
    print(f"    └── actions        # shape=(1, {group_total_dof['action']})")
    for arm_name, arm_info in grouped_arms["action"]:
        print(f"        ├── {arm_name}/")
        for i in range(min(3, arm_info['dof'])):
            print(f"        │   ├── joint_{i}")
        if arm_info['dof'] > 3:
            print(f"        │   └── ...")

def main(json_path):
    components = parse_robot_config_to_episode_components(json_path)
    print("解析结果：")
    print(json.dumps(components, indent=4))
    generate_episode_structure(components, create_files=False)

if __name__ == "__main__":
    json_path = "../config/robot_config.json"
    main(json_path)