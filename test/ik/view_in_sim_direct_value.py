import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

from kinematic_pin import Arm_IK


def run_mujoco_with_fixed_pose(ik_solver, initial_q, urdf_joint_names, target_pose):
    """加载 MuJoCo 场景，使用固定的目标位姿"""
    scene_path = os.path.join(os.path.dirname(__file__), 
                             "../../descriptions/agilex_piper/scene.xml")
    
    if not os.path.exists(scene_path):
        print(f"❌ Error: Scene file not found at {scene_path}")
        return

    print(f"🚀 Loading MuJoCo Scene: {scene_path}")
    
    model = mujoco.MjModel.from_xml_path(scene_path)
    data = mujoco.MjData(model)

    actuators_idx = [model.actuator(name).id for name in urdf_joint_names]
    
    # 初始化关节角度
    for dof_id, joint_value in zip(actuators_idx, initial_q):
        if dof_id >= 0:
            data.ctrl[dof_id] = joint_value

    mujoco.mj_forward(model, data)

    print("👁️ Opening MuJoCo Viewer...")
    print("📌 Using fixed target pose:")
    print(f"   Position: x={target_pose[0]:.5f}, y={target_pose[1]:.5f}, z={target_pose[2]:.5f}")
    print(f"   Euler(rad): roll={target_pose[3]:.5f}, pitch={target_pose[4]:.5f}, yaw={target_pose[5]:.5f}")
    print("⏱️ Running for 5 seconds...")
    print("-" * 50)
    
    # 获取初始位姿用于对比
    current_pose = ik_solver.get_fk(initial_q)
    print(f"📊 Initial pose:")
    print(f"   Position: x={current_pose[0]:.4f}, y={current_pose[1]:.4f}, z={current_pose[2]:.4f}")
    print(f"   Euler(rad): roll={current_pose[3]:.4f}, pitch={current_pose[4]:.4f}, yaw={current_pose[5]:.4f}")
    
    # 解算固定目标位姿的IK
    print("\n🔧 Solving IK for fixed target pose...")
    q_sol = ik_solver.get_ik_solution(
        target_pose[0], target_pose[1], target_pose[2],
        target_pose[3], target_pose[4], target_pose[5]
    )
    
    if q_sol is None:
        print("❌ No IK solution found for the fixed target pose!")
        print("   Running simulation with initial pose only.")
        viewer = mujoco.viewer.launch_passive(model, data)
    else:
        print(f"✅ IK solution found! joints: {q_sol}")
        
        # 验证误差
        pose_check = ik_solver.get_fk(q_sol)
        pos_err = np.linalg.norm(pose_check[:3] - target_pose[:3])
        euler_err = np.linalg.norm(pose_check[3:] - target_pose[3:])
        print(f"   Position error: {pos_err:.6f} m")
        print(f"   Orientation error: {euler_err:.6f} rad")
        print(f"   Joint angles: {np.round(q_sol, 4)}")
        
        # 应用IK解算的关节角度
        for dof_id, joint_value in zip(actuators_idx, q_sol):
            if dof_id >= 0:
                data.ctrl[dof_id] = joint_value
        
        mujoco.mj_forward(model, data)
        viewer = mujoco.viewer.launch_passive(model, data)
        
        # 显示最终位姿
        final_pose = ik_solver.get_fk(q_sol)
        print(f"\n📊 Final pose achieved:")
        print(f"   Position: x={final_pose[0]:.5f}, y={final_pose[1]:.5f}, z={final_pose[2]:.5f}")
        print(f"   Euler(rad): roll={final_pose[3]:.5f}, pitch={final_pose[4]:.5f}, yaw={final_pose[5]:.5f}")
    
    try:
        # 运行5秒
        import time
        start_time = time.time()
        while viewer.is_running() and (time.time() - start_time) < 5.0:
            mujoco.mj_step(model, data)
            viewer.sync()
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        viewer.close()
        print("✅ Simulation closed after 5 seconds.")


def fixed_pose_ik_control():
    """主控制函数 - 使用固定目标位姿"""
    URDF_PATH = "urdf/piper.urdf"
    JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    
    # 固定目标位姿 [x, y, z, roll, pitch, yaw]
    # 值: [0.26132, -0.12457, 0.20848, -0.31887, 0.30107, -0.99658]

    FIXED_TARGET_POSE = np.array([0.26132, -0.12457, 0.20848, -0.31887, 0.30107, -0.99658])

    # FIXED_TARGET_POSE = np.array([0.5, 0, 0.2, 0, 0.30107, -0.99658])
    
    print("--- 🤖 Fixed Pose IK + MuJoCo Control ---")
    print(f"🎯 Fixed target pose: {FIXED_TARGET_POSE}")
    
    try:
        ik = Arm_IK(URDF_PATH)
        print("✅ IK Solver Initialized")
    except Exception as e:
        print(f"❌ Failed to init IK: {e}")
        sys.exit(1)

    # 计算零位并显示初始位姿
    q_zero = np.zeros(ik.model.nq)
    pose_start = ik.get_fk(q_zero)  # 返回 [x, y, z, roll, pitch, yaw]
    
    print(f"\n[Initial Pose (Zero)]")
    print(f"  Position: x={pose_start[0]:.4f}, y={pose_start[1]:.4f}, z={pose_start[2]:.4f}")
    print(f"  Euler(rad): roll={pose_start[3]:.4f}, pitch={pose_start[4]:.4f}, yaw={pose_start[5]:.4f}")
    print("-" * 50)
    
    # 运行仿真
    run_mujoco_with_fixed_pose(ik, q_zero, JOINT_NAMES, FIXED_TARGET_POSE)


if __name__ == "__main__":
    fixed_pose_ik_control()