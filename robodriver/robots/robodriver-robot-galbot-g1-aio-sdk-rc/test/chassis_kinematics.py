import numpy as np

class FourOmniWheelKinematics:
    """
    四轮万向轮底盘运动学
    用于从四个轮子的速度计算机器人本体的运动速度
    """
    
    def __init__(self, layout_type='x45', L=0.2, W=0.2):
        """
        初始化运动学参数
        
        Args:
            layout_type: 布局类型
                - 'x45': X型45°布局（最常见）
                - 'plus': +型布局（前后左右）
                - 'custom': 自定义布局
            L: 底盘长度的一半（前后方向，单位：米）
            W: 底盘宽度的一半（左右方向，单位：米）
        """
        self.L = L  # 前后方向半长
        self.W = W  # 左右方向半宽
        
        # 轮子位置（机器人坐标系：X向前，Y向左）
        # 轮子编号：1-右前，2-左前，3-左后，4-右后
        self.wheel_positions = np.array([
            [L, -W],   # 轮子1: 右前
            [L, W],    # 轮子2: 左前
            [-L, W],   # 轮子3: 左后
            [-L, -W]   # 轮子4: 右后
        ])
        
        # 根据布局类型设置轮子滚动方向
        if layout_type == 'x45':
            # X型45°布局：轮子滚动方向与底盘轴线成45°
            # 轮子方向单位向量（指向轮子自由滚动方向）
            self.wheel_directions = np.array([
                [np.sqrt(2)/2, -np.sqrt(2)/2],   # 右前: 45°方向
                [np.sqrt(2)/2, np.sqrt(2)/2],    # 左前: 135°方向
                [-np.sqrt(2)/2, np.sqrt(2)/2],   # 左后: 225°方向
                [-np.sqrt(2)/2, -np.sqrt(2)/2]   # 右后: 315°方向
            ])
            
        elif layout_type == 'plus':
            # +型布局：每个轮子正对前后左右方向
            self.wheel_directions = np.array([
                [0, -1],   # 右前: 向左滚动
                [1, 0],    # 左前: 向前滚动
                [0, 1],    # 左后: 向右滚动
                [-1, 0]    # 右后: 向后滚动
            ])
            
        elif layout_type == 'custom':
            # 自定义布局 - 需要后续设置
            self.wheel_directions = None
            
        else:
            raise ValueError(f"未知的布局类型: {layout_type}")
        
        # 计算运动学矩阵J和它的伪逆
        self.J = self._calculate_jacobian()
        self.J_pinv = np.linalg.pinv(self.J)  # 伪逆
        
    def _calculate_jacobian(self):
        """
        计算运动学雅可比矩阵J
        v_wheel = J * [vx, vy, omega]^T
        """
        J = np.zeros((4, 3))
        
        for i in range(4):
            # 轮子位置
            rx, ry = self.wheel_positions[i]
            # 轮子方向
            dx, dy = self.wheel_directions[i]
            
            # 前三列对应vx, vy, omega
            J[i, 0] = dx  # vx系数
            J[i, 1] = dy  # vy系数
            J[i, 2] = dy * rx - dx * ry  # omega系数
            
        return J
    
    def wheels_to_chassis(self, wheel_speeds):
        """
        从四个轮子的速度计算底盘运动速度
        
        Args:
            wheel_speeds: 四个轮子的速度数组 [v1, v2, v3, v4]
                          v1: 右前轮，v2: 左前轮，v3: 左后轮，v4: 右后轮
                          正值表示沿着轮子滚动方向运动
                          
        Returns:
            chassis_velocity: [vx, vy, omega]
                vx: 机器人前进速度 (m/s)
                vy: 机器人横向速度 (m/s)
                omega: 机器人旋转角速度 (rad/s)，逆时针为正
        """
        wheel_speeds = np.array(wheel_speeds).flatten()
        
        if len(wheel_speeds) != 4:
            raise ValueError("轮子速度数组必须是4个元素")
        
        # 使用伪逆计算底盘速度
        chassis_velocity = self.J_pinv @ wheel_speeds
        
        return chassis_velocity
    
    def chassis_to_wheels(self, vx, vy, omega):
        """
        从底盘速度计算四个轮子应有的速度（逆运动学）
        
        Args:
            vx: 机器人前进速度 (m/s)
            vy: 机器人横向速度 (m/s)
            omega: 机器人旋转角速度 (rad/s)
            
        Returns:
            wheel_speeds: 四个轮子应有的速度 [v1, v2, v3, v4]
        """
        chassis_velocity = np.array([vx, vy, omega])
        wheel_speeds = self.J @ chassis_velocity
        
        return wheel_speeds
    
    def get_condition_number(self):
        """
        获取运动学矩阵的条件数
        条件数越小，数值稳定性越好
        """
        cond = np.linalg.cond(self.J)
        return cond
    
    def print_configuration(self):
        """打印当前配置信息"""
        print("四轮万向轮运动学配置:")
        print(f"底盘尺寸: L={self.L:.3f}m, W={self.W:.3f}m")
        print("\n轮子位置 (机器人坐标系，X向前，Y向左):")
        for i in range(4):
            print(f"  轮子{i+1}: ({self.wheel_positions[i, 0]:.3f}, {self.wheel_positions[i, 1]:.3f}) m")
        
        print("\n轮子滚动方向 (单位向量):")
        for i in range(4):
            print(f"  轮子{i+1}: [{self.wheel_directions[i, 0]:.3f}, {self.wheel_directions[i, 1]:.3f}]")
        
        print("\n运动学矩阵 J (4x3):")
        print(self.J)
        
        print(f"\n条件数: {self.get_condition_number():.3f}")
        
        # 验证可逆性
        JtJ = self.J.T @ self.J
        det = np.linalg.det(JtJ)
        print(f"J^T*J的行列式: {det:.6f}")
        if abs(det) < 1e-10:
            print("警告: J^T*J接近奇异，运动学可能不稳定")


# ====================== 使用示例 ======================

def example_basic():
    """基本使用示例"""
    print("=" * 50)
    print("示例1: 基本使用")
    print("=" * 50)
    
    # 创建运动学对象（默认X型45°布局）
    kin = FourOmniWheelKinematics(layout_type='x45', L=0.15, W=0.15)
    kin.print_configuration()
    
    # 测试1: 纯前进运动
    print("\n测试1: 纯前进运动 (vx=1.0 m/s)")
    vx, vy, omega = 1.0, 0.0, 0.0
    wheel_speeds = kin.chassis_to_wheels(vx, vy, omega)
    print(f"所需轮速: {wheel_speeds}")
    
    # 验证: 从轮速反算底盘速度
    chassis_vel = kin.wheels_to_chassis(wheel_speeds)
    print(f"反算底盘速度: {chassis_vel}")
    print(f"误差: {np.linalg.norm(chassis_vel - [vx, vy, omega])}")
    
    # 测试2: 纯旋转
    print("\n测试2: 纯旋转 (omega=1.0 rad/s)")
    vx, vy, omega = 0.0, 0.0, 1.0
    wheel_speeds = kin.chassis_to_wheels(vx, vy, omega)
    print(f"所需轮速: {wheel_speeds}")
    
    chassis_vel = kin.wheels_to_chassis(wheel_speeds)
    print(f"反算底盘速度: {chassis_vel}")
    print(f"误差: {np.linalg.norm(chassis_vel - [vx, vy, omega])}")


def example_measurement():
    """测量示例：从轮子测量速度计算底盘速度"""
    print("\n" + "=" * 50)
    print("示例2: 从轮子测量速度计算底盘速度")
    print("=" * 50)
    
    kin = FourOmniWheelKinematics(layout_type='x45', L=0.15, W=0.15)
    
    # 模拟测量到的轮子速度（可能包含噪声）
    # 假设实际底盘运动是 [vx=0.5, vy=0.3, omega=0.2]
    vx_true, vy_true, omega_true = 0.5, 0.3, 0.2
    wheel_speeds_true = kin.chassis_to_wheels(vx_true, vy_true, omega_true)
    
    # 添加测量噪声
    np.random.seed(42)
    noise = np.random.normal(0, 0.01, 4)  # 标准差0.01 m/s
    wheel_speeds_measured = wheel_speeds_true + noise
    
    print(f"实际底盘速度: vx={vx_true:.3f}, vy={vy_true:.3f}, omega={omega_true:.3f}")
    print(f"理论轮速: {wheel_speeds_true}")
    print(f"测量轮速(含噪声): {wheel_speeds_measured}")
    
    # 从测量轮速计算底盘速度
    chassis_estimated = kin.wheels_to_chassis(wheel_speeds_measured)
    print(f"估计底盘速度: vx={chassis_estimated[0]:.3f}, vy={chassis_estimated[1]:.3f}, omega={chassis_estimated[2]:.3f}")
    
    error = np.linalg.norm(chassis_estimated - [vx_true, vy_true, omega_true])
    print(f"估计误差: {error:.6f}")


def example_layout_comparison():
    """不同布局对比"""
    print("\n" + "=" * 50)
    print("示例3: 不同布局对比")
    print("=" * 50)
    
    layouts = ['x45', 'plus']
    
    for layout in layouts:
        print(f"\n{layout.upper()} 布局:")
        kin = FourOmniWheelKinematics(layout_type=layout, L=0.15, W=0.15)
        cond = kin.get_condition_number()
        print(f"  条件数: {cond:.3f}")
        
        # 测试全向运动能力
        test_velocities = [
            [1.0, 0.0, 0.0],   # 纯前进
            [0.0, 1.0, 0.0],   # 纯横向
            [0.0, 0.0, 1.0],   # 纯旋转
            [0.5, 0.5, 0.5],   # 复合运动
        ]
        
        for vx, vy, omega in test_velocities:
            wheels = kin.chassis_to_wheels(vx, vy, omega)
            max_wheel = np.max(np.abs(wheels))
            if max_wheel > 0:
                wheels_normalized = wheels / max_wheel
                print(f"  [{vx:.1f},{vy:.1f},{omega:.1f}] -> 轮速比例: {wheels_normalized}")


def advanced_custom_layout():
    """自定义布局示例"""
    print("\n" + "=" * 50)
    print("示例4: 自定义布局")
    print("=" * 50)
    
    class CustomOmniRobot(FourOmniWheelKinematics):
        def __init__(self, L=0.2, W=0.15):
            super().__init__(layout_type='custom', L=L, W=W)
            
            # 自定义轮子方向（例如，每个轮子与X轴成不同角度）
            angles = [30, 120, 210, 300]  # 度
            self.wheel_directions = np.zeros((4, 2))
            
            for i, angle_deg in enumerate(angles):
                angle_rad = np.deg2rad(angle_deg)
                self.wheel_directions[i] = [np.cos(angle_rad), np.sin(angle_rad)]
            
            # 重新计算J和伪逆
            self.J = self._calculate_jacobian()
            self.J_pinv = np.linalg.pinv(self.J)
    
    robot = CustomOmniRobot(L=0.2, W=0.15)
    robot.print_configuration()
    
    # 测试运动
    vx, vy, omega = 0.5, 0.3, 0.1
    wheels = robot.chassis_to_wheels(vx, vy, omega)
    print(f"\n底盘速度 [{vx}, {vy}, {omega}] 对应的轮速: {wheels}")
    
    # 验证反算
    chassis_back = robot.wheels_to_chassis(wheels)
    print(f"反算底盘速度: {chassis_back}")
    print(f"误差: {np.linalg.norm(chassis_back - [vx, vy, omega])}")


if __name__ == "__main__":
    # 运行所有示例
    example_basic()
    example_measurement()
    example_layout_comparison()
    advanced_custom_layout()