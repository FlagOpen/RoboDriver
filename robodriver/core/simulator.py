import mujoco
import mujoco.viewer
import logging_mp
import numpy as np

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
    log_data: bool = False

    def __post_init__(self):
        if self.from_unit != "deg" and self.from_unit != "rad":
            raise ValueError(
                f"from_unit only support \'deg\' or \'rad\' in sim"
            )
    
class Simulator:
    def __init__(self, config: SimulatorConfig):
        self.config = config

        self.model = mujoco.MjModel.from_xml_path(self.config.xml_path)
        self.data = mujoco.MjData(self.model)
        self.cam = mujoco.MjvCamera()
        self.renderer = mujoco.Renderer(self.model, height=self.config.render_height, width=self.config.render_width)
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data) if self.config.show_viewer == True else None

        self.model.opt.timestep = config.timestep
        mujoco.mjv_defaultCamera(self.cam)

    def update(self, action: dict[str, Any], prefix: str, suffix: str):
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
                self.data.ctrl[dof_id] = joint_value

        mujoco.mj_step(self.model, self.data)

        self.viewer.sync()
        self.renderer.update_scene(self.data, self.cam)
        rgb = self.renderer.render()

        return rgb
    
    def stop(self):
        self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()