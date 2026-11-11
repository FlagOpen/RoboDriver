"""AstriBot T1 关节走 ROS2 话题的只读自检：不下发任何动作。

验证 ROS_DOMAIN_ID 下能否发现并订阅各部件的 obs/action 话题、自由度、首帧数据。
obs   = /astribot_<part>/joint_space_states          (RobotJointState.position, BEST_EFFORT)
action= /astribot_<part>/joint_space_command_forward  (RobotJointController.command, RELIABLE)

注意：command_forward 在机器人静止时不发布，action 首帧收不到属正常，需边遥操作边自检。

用法：
    python -m robodriver_robot_astribott1_aio_py.check_env_ros
    python -m robodriver_robot_astribott1_aio_py.check_env_ros --domain 25 --wait 10
"""

import argparse
import sys

# 与适配器一致的部件列表（effector_* 会在 AstribotJointNode 里映射到 gripper_*）
PARTS = ["arm_left", "arm_right", "effector_left", "effector_right", "head", "torso", "chassis"]


def _show_versions():
    print("=" * 60)
    print(f"python   : {sys.version.split()[0]}")
    try:
        import rclpy
        print(f"rclpy    : OK ({getattr(rclpy, '__file__', '?')})")
    except Exception as e:
        print(f"rclpy    : 导入失败 {type(e).__name__}: {e}")
        return False
    try:
        from astribot_msgs.msg import RobotJointState  # noqa: F401
        print("astribot_msgs: OK")
    except Exception as e:
        print(f"astribot_msgs: 导入失败 {type(e).__name__}: {e}")
        print("       请先 source ~/astribot_sdk/ros-msgs-workspace/install/setup.bash")
        return False
    return True


def _check_ros_joints(domain_id, wait_s):
    print("-" * 60)
    from .ros_joints import AstribotJointNode

    node = AstribotJointNode(parts=PARTS, domain_id=domain_id)
    print(f"ROS_DOMAIN_ID={domain_id}, 等待首帧 obs+action（最长 {wait_s}s）...")
    print("订阅话题：")
    for part in PARTS:
        print(f"  {part:<16} obs={node.obs_topic(part)}  action={node.action_topic(part)}")

    try:
        node.start(wait_first_frame_s=wait_s)
    except Exception as e:
        print(f"[失败] {type(e).__name__}: {e}")
        return False

    dof = node.get_dof()
    obs = node.get_observation()
    action = node.get_action()

    print("各部件自由度与首帧（rad）:")
    all_ok = True
    for part in PARTS:
        o = obs.get(part, [])
        a = action.get(part, [])
        first_o = ", ".join(f"{v:+.3f}" for v in o[:3])
        print(f"  {part:<16} dof={dof.get(part, 0):<3} obs[{first_o}...] action_len={len(a)}")
        if len(o) == 0:
            print(f"    [警告] {part} obs 为空")
            all_ok = False
    # action 走 command_forward，机器人静止时不发布，此时 get_action 回退到 obs，
    # action_len 等于 obs 长度但不代表收到了真实指令。故只提示，不计入失败。
    print("注: command_forward 静止时不发布，action 会回退显示 obs 值；"
          "要验证真实 action，请边遥操作边自检。")

    node.stop()
    return all_ok


def main():
    parser = argparse.ArgumentParser(description="AstriBot T1 ROS 关节只读自检")
    parser.add_argument("--domain", type=int, default=25, help="ROS_DOMAIN_ID（默认 25）")
    parser.add_argument("--wait", type=float, default=10.0, help="等待首帧超时秒数（默认 10）")
    args = parser.parse_args()

    if not _show_versions():
        print("=" * 60)
        print("结论: ROS 环境不完整，无法走 ROS 关节")
        return 1

    ok = _check_ros_joints(args.domain, args.wait)
    print("=" * 60)
    print("结论: " + ("各部件 obs 话题正常，可用 --robot.use_ros_joints=true 采集"
                     if ok else "存在缺失的 obs 话题，请检查部件是否在线或话题名"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
