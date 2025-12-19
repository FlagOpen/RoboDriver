import yaml
import numpy as np
from typing import Dict, Any, List
import os


class ConfigLoader:
    """配置加载器，用于读取和管理仿真配置"""
    
    def __init__(self, config_path: str = "config/simulation_config.yaml"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """加载YAML配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            # 验证必要配置项
            self._validate_config(config)
            return config
            
        except FileNotFoundError:
            print(f"警告: 配置文件 {self.config_path} 未找到，使用默认配置")
            return self._get_default_config()
        except yaml.YAMLError as e:
            print(f"错误: 配置文件解析失败: {e}")
            raise
        except Exception as e:
            print(f"错误: 加载配置文件时发生未知错误: {e}")
            raise
    
    def _validate_config(self, config: Dict[str, Any]):
        """验证配置完整性"""
        required_sections = ['simulation', 'robot', 'keyboard', 'motion', 'performance', 'threading']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"配置缺少必要部分: {section}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """获取默认配置"""
        return {
            'simulation': {
                'plane_enabled': True,
                'cube': {
                    'size': [0.05, 0.05, 0.05],
                    'position': [0.4, 0.0, 0.00]
                },
                'camera': {
                    'resolution': [640, 480],
                    'position': [1.0, 0.0, 1.0],
                    'lookat': [0, 0, 0],
                    'fov': 60,
                    'gui_enabled': False
                }
            },
            'robot': {
                'kp_gains': [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100],
                'kv_gains': [450, 450, 350, 350, 200, 200, 200, 10, 10],
                'force_min': [-87, -87, -87, -87, -12, -12, -12, -100, -100],
                'force_max': [87, 87, 87, 87, 12, 12, 12, 100, 100],
                'motor_dofs': [0, 1, 2, 3, 4, 5, 6],
                'finger_dofs': [7, 8]
            },
            'keyboard': {
                'position_step': 0.002,
                'orientation_step': 0.005,
                'gripper_close_force': [-3.0, -3.0],
                'gripper_open_force': [3.0, 3.0],
                'initial_position': [0.4, 0.0, 0.4],
                'initial_euler': [3.141592653589793, 0.0, 0.0]
            },
            'motion': {
                'initial_path_waypoints': 200,
                'initial_wait_steps': 100,
                'target_frequency_hz': 60
            },
            'performance': {
                'stats_window_size': 100,
                'print_interval': 100,
                'enable_step_stats': True,
                'enable_render_stats': True,
                'enable_tick_stats': True
            },
            'threading': {
                'control_thread_timeout': 2.0,
                'queue_timeout': 0.01
            }
        }
    
    def get_simulation_config(self) -> Dict[str, Any]:
        """获取仿真配置"""
        return self.config['simulation']
    
    def get_robot_config(self) -> Dict[str, Any]:
        """获取机械臂配置"""
        return self.config['robot']
    
    def get_keyboard_config(self) -> Dict[str, Any]:
        """获取键盘控制配置"""
        return self.config['keyboard']
    
    def get_motion_config(self) -> Dict[str, Any]:
        """获取运动规划配置"""
        return self.config['motion']
    
    def get_performance_config(self) -> Dict[str, Any]:
        """获取性能监控配置"""
        return self.config['performance']
    
    def get_threading_config(self) -> Dict[str, Any]:
        """获取线程配置"""
        return self.config['threading']
    
    def get_numpy_array(self, key_path: str) -> np.ndarray:
        """
        获取配置值并转换为numpy数组
        
        Args:
            key_path: 配置键路径，格式如 'robot.kp_gains'
        
        Returns:
            numpy数组
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if key not in value:
                raise KeyError(f"配置键不存在: {key_path}")
            value = value[key]
        
        return np.array(value, dtype=np.float64)
    
    def save_config(self, output_path: str = None):
        """
        保存当前配置到文件
        
        Args:
            output_path: 输出文件路径，如果为None则保存到原始路径
        """
        if output_path is None:
            output_path = self.config_path
        
        # 确保目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
        
        print(f"配置已保存到: {output_path}")


# 全局配置实例
_config_loader = None

def get_config_loader(config_path: str = "config/simulation_config.yaml") -> ConfigLoader:
    """
    获取全局配置加载器实例（单例模式）
    
    Args:
        config_path: 配置文件路径
    
    Returns:
        ConfigLoader实例
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_path)
    
    return _config_loader
