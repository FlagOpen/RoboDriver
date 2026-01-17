import mujoco
import mujoco.viewer
import logging_mp
import numpy as np
import time
import threading
import cv2

from typing import Any
from dataclasses import dataclass


logger = logging_mp.get_logger(__name__)


@dataclass
class SimulatorConfig():
    xml_path: str | None = None
    from_unit: str = "deg"
    render_height: int = 1200
    render_width: int = 1600
    timestep: float = 0.01
    show_viewer: bool = False
    show_local: bool = False
    log_data: bool = False

    def __post_init__(self):
        if self.from_unit != "deg" and self.from_unit != "rad":
            raise ValueError(
                f"from_unit only support \'deg\' or \'rad\' in sim"
            )

class SimulationThread(threading.Thread):
    def __init__(self, model, data, config, running_event, lock):
        super().__init__()
        self.model = model
        self.data = data
        self.config = config
        self.running_event = running_event
        self.lock = lock

    def run(self):
        while self.running_event.is_set():
            with self.lock:
                mujoco.mj_step(self.model, self.data)
            time.sleep(self.model.opt.timestep)
    
    def stop(self):
        """停止线程"""
        self.running_event.clear()

class ViewerRendererThread(threading.Thread):
    """Viewer和Render线程类"""
    
    def __init__(self, model, data, config, running_event, lock):
        """
        初始化线程
        
        Args:
            model: mujoco模型
            data: mujoco数据
            config: 配置参数
            running_event: 运行事件 (threading.Event)
            lock: 线程锁
        """
        super().__init__()
        self.model = model
        self.data = data
        self.config = config
        self.running_event = running_event
        self.lock = lock
        self.viewer = None
        self.renderer = None
        self.latest_image = None
        self.image_lock = threading.Lock()
        
    def run(self):
        """线程主函数"""
        print("Viewer started in thread:", threading.current_thread().name)
        print(f"show_viewer config: {self.config.show_viewer}")
        
        # 创建renderer
        self.renderer = mujoco.Renderer(self.model, height=1200, width=1600)
        
        # 根据配置决定是否创建viewer
        if self.config.show_viewer:
            try:
                print("Attempting to launch viewer...")
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
                print("Viewer launched in passive mode")
                if self.viewer is not None:
                    print(f"Viewer is_running: {self.viewer.is_running()}")
                else:
                    print("Viewer is None after launch!")
            except Exception as e:
                print(f"Failed to launch viewer: {e}")
                import traceback
                traceback.print_exc()
                self.viewer = None
        
        # 配置相机
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        
        try:
            iteration = 0
            while self.running_event.is_set() and (self.viewer is None or self.viewer.is_running()):
                iteration += 1
                if iteration % 100 == 0:
                    print(f"Viewer thread iteration {iteration}, viewer is None: {self.viewer is None}, is_running: {self.viewer.is_running() if self.viewer is not None else 'N/A'}")
                
                with self.lock:
                    # 更新viewer显示
                    if self.viewer is not None:
                        self.viewer.sync()
                    
                    # 在同一线程中进行render操作
                    self.renderer.update_scene(self.data, camera=cam)
                    image = self.renderer.render()
                    
                    # 存储最新图像
                    with self.image_lock:
                        self.latest_image = image.copy() if image is not None else None
                    
                    # 显示渲染图像
                    if image is not None and self.config.show_local == True:
                        cv2.imshow('Render', image)
                        cv2.waitKey(1)
                
                time.sleep(0.016)  # ~60Hz刷新率
                
        except KeyboardInterrupt:
            print("Viewer thread interrupted")
        except Exception as e:
            print(f"Viewer thread error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            print("Viewer thread exiting, cleanup...")
            self.cleanup()
            
    def cleanup(self):
        """清理资源"""
        print("Cleaning up viewer/renderer resources")
        if self.viewer is not None:
            self.viewer.close()
        if self.renderer is not None:
            # 注意：mujoco.Renderer没有close方法，可以设置为None
            self.renderer = None
        if  self.config.show_local == True:
            cv2.destroyAllWindows()
        
    def stop(self):
        """停止线程"""
        self.running_event.clear()
    
    def get_latest_image(self):
        """获取最新渲染图像"""
        with self.image_lock:
            return self.latest_image.copy() if self.latest_image is not None else None

    
class Simulator:
    def __init__(self, config: SimulatorConfig):
        self.config = config

        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)

        self.model.opt.timestep = config.timestep

        self.running_event = threading.Event()
        self.lock = threading.Lock()
        
        self.sim_thread = SimulationThread(self.model, self.data, self.config, self.running_event, self.lock)
        self.view_thread = ViewerRendererThread(self.model, self.data, self.config, self.running_event, self.lock)
        # self._last_update_time = None
        # self._max_steps_per_update = 10  # Limit steps to prevent freezing

    def start(self):
        """启动模拟器线程"""
        self.running_event.set()
        self.sim_thread.start()
        self.view_thread.start()

    def send_action(self, action: dict[str, Any], prefix: str, suffix: str):
        actuators_idx = [self.model.actuator(name.removeprefix(f"{prefix}").removesuffix(f"{suffix}")).id for name in action]
        
        goal_joint = list(action.values())
        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)
        
        if self.config.from_unit == "deg":
            goal_joint_radians = goal_joint_numpy * (np.pi / 180.0)
        elif self.config.from_unit == "rad":
            goal_joint_radians = goal_joint_numpy

        if self.config.log_data == True:
            logger.info(f"action: {action}"), 
            logger.info(f"actuators_idx: {actuators_idx}")
            logger.info(f"goal_joint_numpy: {goal_joint_numpy}")
            logger.info(f"goal_joint_radians: {goal_joint_radians}")

        for dof_id, joint_value in zip(actuators_idx, goal_joint_radians):
            if dof_id >= 0:
                with self.lock:
                    self.data.ctrl[dof_id] = joint_value
    
    def get_render_image(self):
        """获取最新渲染图像"""
        return self.view_thread.get_latest_image()

    
    def stop(self):
        """停止模拟器线程"""
        self.running_event.clear()
        # 等待线程结束
        if self.sim_thread.is_alive():
            self.sim_thread.join(timeout=1.0)
        if self.view_thread.is_alive():
            self.view_thread.join(timeout=1.0)
