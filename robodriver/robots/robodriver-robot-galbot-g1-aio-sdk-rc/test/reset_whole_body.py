from galbot_sdk.g1 import GalbotRobot, GalbotMotion
import time


def main():
    robot = GalbotRobot.get_instance()
    motion = GalbotMotion.get_instance()

    if not robot.init():
        print("GalbotRobot init failed.")
        return
    if not motion.init():
        print("GalbotMotion init failed.")
        return

    time.sleep(2)

    # Whole-body joints: leg(5) + head(2) + left_arm(7) + right_arm(7)
    whole_body_joint_1 = [
        0.25, 1.1, 0.85, 0.0, 0.0,       # leg
        0.5, 0.5,                        # head
        2.0, -1.55, -0.55, -1.7, -0.0, -0.8, 0.2,   # left_arm
        -2.0, 1.55, 0.55, 1.7, 0.0, 0.8, 0.2        # right_arm
    ]

    # Base pose command odom(x, y, yaw)
    base_x_1 = 0.2
    base_y_1 = 0.0
    base_yaw_1 = 0.0

    # 可选坐标系（frame_id: base_link/odom/map, reference_frame_id: odom/map）
    frame_id = "base_link"
    reference_frame_id = "odom"

    # 底盘位姿插值时间（秒），用于生成平滑底盘轨迹
    base_time_s = 15.0

    print("=== Whole-body + base pose (odom) ===")
    pose_status = robot.execute_whole_body_target(
        whole_body_joint_1,
        base_x_1,
        base_y_1,
        base_yaw_1,
        frame_id,
        reference_frame_id,
        True,
        0.1,
        base_time_s,
        base_time_s,
    )
    print("execute_whole_body_target (pose) status:", pose_status)

    time.sleep(1)

    # 一键回零
    result = robot.zero_whole_body_and_base(
        frame_id,
        reference_frame_id,
        True,
        0.2,
        15.0,
    )
    print("Zero joint status:", result[0])
    print("Zero base status:", result[1])

    robot.request_shutdown()
    robot.wait_for_shutdown()
    robot.destroy()


if __name__ == "__main__":
    main()