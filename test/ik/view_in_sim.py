import os
import sys
import numpy as np
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R

from kinematic_pin import Arm_IK


def run_mujoco_test(ik_solver, initial_q, urdf_joint_names):
    """加载 MuJoCo 场景，支持实时调整目标位姿"""
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
    print("📝 Control Instructions:")
    print("  - x/y/z + value: 设置位置偏移 (例如: x 0.2)")
    print("  - roll/pitch/yaw + value: 设置欧拉角 (例如: roll 0.1)")
    print("  - reset: 重置到初始位姿")
    print("  - status: 显示当前目标")
    print("  - quit: 退出程序")
    print("-" * 50)
    
    # 获取初始位姿 - get_fk返回[x, y, z, roll, pitch, yaw]
    current_pose = ik_solver.get_fk(initial_q)
    target_pos = current_pose[:3].copy()
    target_euler = current_pose[3:].copy()
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        while viewer.is_running():
            # 检查是否有用户输入
            if sys.stdin in select.select([sys.stdin], [], [], 0)[0]:
                command = sys.stdin.readline().strip().lower()
                
                if command == 'quit':
                    break
                elif command == 'reset':
                    current_pose = ik_solver.get_fk(initial_q)
                    target_pos = current_pose[:3].copy()
                    target_euler = current_pose[3:].copy()
                    print("🔄 Reset to initial pose")
                elif command == 'status':
                    print(f"📊 Current target:")
                    print(f"   Position: x={target_pos[0]:.4f}, y={target_pos[1]:.4f}, z={target_pos[2]:.4f}")
                    print(f"   Euler(rad): roll={target_euler[0]:.4f}, pitch={target_euler[1]:.4f}, yaw={target_euler[2]:.4f}")
                else:
                    parts = command.split()
                    if len(parts) == 2:
                        param, value = parts[0], float(parts[1])
                        
                        # 更新位置
                        if param in ['x', 'y', 'z']:
                            idx = {'x': 0, 'y': 1, 'z': 2}[param]
                            target_pos[idx] += value
                            print(f"📍 {param.upper()} offset: {value:+.4f}")
                        
                        # 更新欧拉角
                        elif param in ['roll', 'pitch', 'yaw']:
                            idx = {'roll': 0, 'pitch': 1, 'yaw': 2}[param]
                            target_euler[idx] += value
                            print(f"🔄 {param.upper()} offset: {value:+.4f} rad")
                        else:
                            print(f"❌ Unknown command: {param}")
                            continue
                        
                        # 计算新的IK解 - 直接传入位置和欧拉角
                        q_sol = ik_solver.get_ik_solution(
                            target_pos[0], target_pos[1], target_pos[2],
                            target_euler[0], target_euler[1], target_euler[2]
                        )
                        
                        if q_sol is not None:
                            # 应用新的关节角度
                            for dof_id, joint_value in zip(actuators_idx, q_sol):
                                if dof_id >= 0:
                                    data.ctrl[dof_id] = joint_value
                            
                            # 验证误差
                            pose_check = ik_solver.get_fk(q_sol)
                            pos_err = np.linalg.norm(pose_check[:3] - target_pos)
                            euler_err = np.linalg.norm(pose_check[3:] - target_euler)
                            print(f"✅ IK solved, position error: {pos_err:.6f} m, orientation error: {euler_err:.6f} rad")
                        else:
                            print("❌ No IK solution found, keeping previous position")
            
            mujoco.mj_step(model, data)
            viewer.sync()
            
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")
    finally:
        viewer.close()
        print("✅ Simulation closed.")


def interactive_ik_control():
    """主控制函数"""
    URDF_PATH = "urdf/piper_left.urdf"
    JOINT_NAMES = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    
    print("--- 🤖 Interactive IK + MuJoCo Control ---")
    
    try:
        ik = Arm_IK(URDF_PATH)
        print("✅ IK Solver Initialized")
    except Exception as e:
        print(f"❌ Failed to init IK: {e}")
        sys.exit(1)

    # 计算零位并显示初始位姿
    q_zero = np.zeros(ik.model.nq)
    pose_start = ik.get_fk(q_zero)  # 返回 [x, y, z, roll, pitch, yaw]
    
    print(f"[Initial Pose]")
    print(f"  Position: x={pose_start[0]:.4f}, y={pose_start[1]:.4f}, z={pose_start[2]:.4f}")
    print(f"  Euler(rad): roll={pose_start[3]:.4f}, pitch={pose_start[4]:.4f}, yaw={pose_start[5]:.4f}")
    print("-" * 50)
    
    # 进入仿真控制
    run_mujoco_test(ik, q_zero, JOINT_NAMES)


if __name__ == "__main__":
    import select  # 用于非阻塞输入检查
    
    interactive_ik_control()