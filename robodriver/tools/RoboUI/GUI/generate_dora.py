import os
import shutil
import json
from dataclasses import dataclass, field
from ruamel.yaml import YAML
from collections import defaultdict

os.chdir(os.path.dirname(os.path.abspath(__file__)))




HEADER = """from dataclasses import dataclass, field
from operating_platform.robot.robots.com_configs.cameras import *
from operating_platform.robot.robots.com_configs.motors import *
from operating_platform.robot.robots.configs import RobotConfig
"""

def json_to_py(json_file: str, py_file: str):
    """
    根据 JSON 文件生成对应的 Python 配置文件
    :param json_file: 输入的 JSON 配置路径
    :param py_file: 输出的 Python 文件路径
    """
    # 读取 JSON
    with open(json_file, "r", encoding="utf-8") as f:
        config = json.load(f)

    # 获取文件名（不带扩展名）
    base_name = "demo_robot"
    class_name = "DemoRobotRobotConfig"

    lines = []
    lines.append(HEADER)
    lines.append(f'@RobotConfig.register_subclass("{base_name}")')
    lines.append("@dataclass")
    lines.append(f"class {class_name}:")

    # -------- 生成 follower_motors --------
    follower_entries = []
    for comp in config["components"]:
        comp_type = comp["type"]
        comp_id = comp["id"]
        outputs = comp["params"].get("output", [])
        outputs_info = comp.get("outputs_info", {})

        for output in outputs:
            if "pose" in output:
                motors = outputs_info.get(output, {}).get("motors", {})
                follower_entries.append(f'            "{comp_id}": DDSMotorsBusConfig(')
                follower_entries.append(f'                topic="",')  # 可根据需要填 topic
                follower_entries.append(f'                group="",')  # 可根据需要填 group
                follower_entries.append("                motors={")
                for k, v in motors.items():
                    follower_entries.append(f'                    "{k}": {v},')
                follower_entries.append("                },")
                follower_entries.append("            ),")

    if follower_entries:
        lines.append("    follower_motors: dict[str, MotorsBusConfig] = field(")
        lines.append("        default_factory=lambda: {")
        lines.extend(follower_entries)
        lines.append("        }")
        lines.append("    )\n")

    # -------- 生成 cameras --------
    camera_entries = []
    camera_index = 0
    for comp in config["components"]:
        comp_type = comp["type"]
        comp_id = comp["id"]
        outputs = comp["params"].get("output", [])
        if "image" in outputs:
            params = comp["params"]
            width = params.get("width", 640)
            height = params.get("height", 480)
            fps = params.get("period", 30)
            camera_entries.append(f'            "image_{comp_id}": OpenCVCameraConfig(')
            camera_entries.append(f"                camera_index={camera_index},")
            camera_entries.append(f"                fps={fps},")
            camera_entries.append(f"                width={width},")
            camera_entries.append(f"                height={height},")
            camera_entries.append("            ),")
            camera_index += 1

    if camera_entries:
        lines.append("    cameras: dict[str, CameraConfig] = field(")
        lines.append("        default_factory=lambda: {")
        lines.extend(camera_entries)
        lines.append("        }")
        lines.append("    )\n")

    # use_videos
    use_videos = config.get("use_videos", False)
    lines.append(f"    use_videos: bool = {str(use_videos)}\n")

    # microphones (固定模板)
    lines.append("    microphones: dict[str, int] = field(")
    lines.append("        default_factory=lambda: {")
    lines.append("            # \"audio_right\": 2,")
    lines.append("            # \"audio_left\": 4,")
    lines.append("        }")
    lines.append("    )\n")

    # -------- 输出到文件 --------
    output_code = "\n".join(lines)
    os.makedirs(os.path.dirname(py_file), exist_ok=True)
    with open(py_file, "w", encoding="utf-8") as f:
        f.write(output_code)


def json_to_yaml(json_file, yaml_file):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    nodes = []
    outputs_map = defaultdict(list)

    for comp in data["components"]:
        comp_id = comp["id"]
        path = comp["params"]["path"]
        period = comp["params"].get("period", 30)
        outputs = comp["params"].get("output", [])

        env = {}
        if "device_serial" in comp["params"]:
            env["DEVICE_SERIAL"] = comp["params"]["device_serial"]
        if "width" in comp["params"]:
            env["IMAGE_WIDTH"] = comp["params"]["width"]
        if "height" in comp["params"]:
            env["IMAGE_HEIGHT"] = comp["params"]["height"]

        # 每个 output 生成单独节点，id = output + "_" + component id
        for out in outputs:
            node_id = f"{out}_{comp_id}" 
            node = {
                "id": node_id,
                "path": path,
                "inputs": {
                    "tick": f"dora/timer/millis/{period}"
                },
                "outputs": [out]
            }
            if env:
                node["env"] = env

            nodes.append(node)
            outputs_map[out].append(node_id)  

    # 生成 zeromq_bridge 输入
    bridge_inputs = {}
    for out, node_ids in outputs_map.items():
        for nid in node_ids:
            bridge_inputs[nid] = f"{nid}/{out}"

    nodes.append({
        "id": "zeromq_bridge",
        "path": "dora_zeromq.py",
        "inputs": bridge_inputs
    })

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.preserve_quotes = True

    with open(yaml_file, "w", encoding="utf-8") as f:
        yaml.dump({"nodes": nodes}, f)




def make_output_dir(base_path: str, folder_name: str) -> str:
    """
    在指定路径下生成文件夹，如果不存在则创建
    :param base_path: 基础路径
    :param folder_name: 文件夹名称
    :return: 文件夹的绝对路径
    """
    output_dir = os.path.join(base_path, folder_name)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.abspath(output_dir)

def copy_file_to_dir(src_file: str, dst_dir: str) -> str:
    """
    把指定文件复制到目标文件夹下
    :param src_file: 源文件路径
    :param dst_dir: 目标文件夹路径
    :return: 复制后的文件路径
    """
    if not os.path.isfile(src_file):
        raise FileNotFoundError(f"源文件不存在: {src_file}")
    os.makedirs(dst_dir, exist_ok=True)

    dst_file = os.path.join(dst_dir, os.path.basename(src_file))
    shutil.copy2(src_file, dst_file)
    return os.path.abspath(dst_file)

def main(robot_name):
    folder = make_output_dir("../../robots", "robot_config")
    json_to_py(f"../config/{robot_name}.json",os.path.join(folder, "robot_config.py"))
    json_to_yaml(f"../config/{robot_name}.json", os.path.join(folder, "robot_config.yml"))
    copy_file_to_dir("../templates/dora_zeromq.py", folder)
    copy_file_to_dir("../templates/manipulator.py", folder)

if __name__ == "__main__":
    folder = make_output_dir("../../robots", "demo_robot")
    json_to_py("../config/robot_config.json",os.path.join(folder, "robot_config.py"))
    json_to_yaml("../config/robot_config.json", os.path.join(folder, "robot_config.yml"))
    copy_file_to_dir("../templates/dora_zeromq.py", folder)
    copy_file_to_dir("../templates/manipulator.py", folder)