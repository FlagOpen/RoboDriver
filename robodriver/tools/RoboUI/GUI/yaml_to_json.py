import os
import json
import uuid
import ast
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def generate_config(components, component_manager, develop_mode, use_videos=False, filename="demo_robot"):
    """
    根据画布上的组件生成配置文件
    """
    config = {
        "develop_mode": develop_mode,
        "use_videos": use_videos,
        "components": []
    }

    for item in components:
        full_outputs = {}
        comp_type = item["data"]["type"]
        params = item["data"]["params"]
        comp_id = item["data"].get("id", str(uuid.uuid4()))
        # 获取 outputs_info
        outputs_info_cfg = component_manager.get_flat_components()[comp_type].get("outputs_info", {})
        selected_outputs = params.get("output", []) 
        full_outputs = {}

        if outputs_info_cfg:
            # 先准备一个“基础模板”：对于非 joints，直接用 YAML 里的 outputs_info
            base_outputs = outputs_info_cfg

            # 特殊处理 joints：用 joint_index 生成 motors 映射
            if "joints" in outputs_info_cfg:
                joint_index = params.get("joint_index", [])
                # 清洗 joint_index，确保是 int 列表
                if isinstance(joint_index, list):
                    selected_joints = [j for j in joint_index if isinstance(j, int)]
                elif isinstance(joint_index, str):
                    try:
                        cleaned_str = joint_index.replace("[", "").replace("]", "").replace(" ", "")
                        selected_joints = [int(x) for x in cleaned_str.split(",") if x.strip().isdigit()]
                    except Exception:
                        selected_joints = []
                else:
                    try:
                        selected_joints = [int(joint_index)] if joint_index is not None else []
                    except Exception:
                        selected_joints = []

                joint_map = {}
                for i, joint in enumerate(selected_joints):
                    joint_name = f"joint_{joint}"
                    joint_detail = [i + 1, "robot_motor"]
                    joint_map[joint_name] = joint_detail

                # 用我们动态生成的 joints 覆盖 YAML 里的 joints 定义
                base_outputs = {
                    **outputs_info_cfg,
                    "joints": {"motors": joint_map}
                }

            # 通用：按 output 勾选过滤需要的 output（包含 joints / pose / 其他）
            full_outputs = {
                out: base_outputs[out]
                for out in selected_outputs
                if out in base_outputs
            }


        data_to_save = {
            "type": comp_type,
            "params": params,
            "id": comp_id,
            "outputs_info": full_outputs
        }
        config["components"].append(data_to_save)

    # 保存文件
    save_dir = "../config"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{filename}.json")

    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Existing file {save_path} removed.")

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

    return save_path
