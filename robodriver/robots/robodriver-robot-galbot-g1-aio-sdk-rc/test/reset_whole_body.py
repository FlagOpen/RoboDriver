from galbot_sdk.g1 import GalbotRobot, GalbotMotion
import time
import yaml
import argparse
import sys


def load_poses_from_yaml(yaml_file):
    """从YAML文件加载姿势配置"""
    try:
        with open(yaml_file, 'r') as f:
            data = yaml.safe_load(f)
        return data.get('poses', {})
    except Exception as e:
        print(f"加载YAML文件失败: {e}")
        return {}


def get_pose_config(poses, pose_id):
    """根据pose_id获取姿势配置"""
    if pose_id not in poses:
        print(f"错误: pose_id={pose_id} 在配置文件中不存在")
        print(f"可用的pose_id: {list(poses.keys())}")
        return None
    
    pose_config = poses[pose_id]
    
    # 设置默认值
    default_config = {
        'whole_body_joints': [],
        'base_x': 0.0,
        'base_y': 0.0,
        'base_yaw': 0.0,
        'frame_id': 'base_link',
        'reference_frame_id': 'odom',
        'base_time_s': 15.0,
    }
    
    # 合并配置
    for key in default_config:
        if key not in pose_config:
            pose_config[key] = default_config[key]
    
    return pose_config


def execute_pose(robot, pose_config):
    """执行单个姿势"""
    print(f"=== 执行姿势配置 ===")
    print(f"关节数: {len(pose_config['whole_body_joints'])}")
    print(f"底盘位置: x={pose_config['base_x']}, y={pose_config['base_y']}, yaw={pose_config['base_yaw']}")
    print(f"坐标系: frame_id={pose_config['frame_id']}, reference_frame_id={pose_config['reference_frame_id']}")
    
    pose_status = robot.execute_whole_body_target(
        pose_config['whole_body_joints'],
        pose_config['base_x'],
        pose_config['base_y'],
        pose_config['base_yaw'],
        pose_config['frame_id'],
        pose_config['reference_frame_id'],
        True,
        0.1,
        pose_config['base_time_s'],
        pose_config['base_time_s'],
    )
    print(f"execute_whole_body_target 状态: {pose_status}")
    
    return pose_status


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Galbot机器人全身复位脚本')
    parser.add_argument('--pose_id', type=int, default=1, 
                       help='要执行的姿势ID (默认: 1)')
    parser.add_argument('--config', type=str, default='reset_data.yml',
                       help='YAML配置文件路径 (默认: reset_data.yml)')
    parser.add_argument('--list_poses', action='store_true',
                       help='列出所有可用的姿势ID')
    parser.add_argument('--no_zero', action='store_true',
                       help='不执行回零操作')
    
    args = parser.parse_args()
    
    # 加载姿势配置
    poses = load_poses_from_yaml(args.config)
    
    if args.list_poses:
        print("可用的姿势ID:")
        for pose_id in sorted(poses.keys()):
            print(f"  {pose_id}")
        return
    
    if not poses:
        print(f"错误: 配置文件 {args.config} 中没有找到姿势配置")
        print("请确保配置文件包含 'poses' 部分")
        return
    
    # 获取指定姿势配置
    pose_config = get_pose_config(poses, args.pose_id)
    if pose_config is None:
        return
    
    # 初始化机器人
    robot = GalbotRobot.get_instance()

    if not robot.init():
        print("GalbotRobot 初始化失败.")
        return

    time.sleep(2)
    
    # 执行姿势
    _pose_status = execute_pose(robot, pose_config)
    
    # 关闭机器人
    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()


if __name__ == "__main__":
    main()
