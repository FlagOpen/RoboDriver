import mujoco
import mujoco.viewer
import logging_mp
import numpy as np
import time

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
        
        # Time tracking for simulation synchronization
        self._last_update_time = None
        self._max_steps_per_update = 10  # Limit steps to prevent freezing

    def update(self, action: dict[str, Any], prefix: str, suffix: str):
        # Start timing the entire update function
        update_start_time = time.perf_counter()
        
        # Apply action to actuators
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

        # Time synchronization: execute multiple physics steps based on elapsed time
        current_time = time.perf_counter()
        
        if self._last_update_time is None:
            # First update, just do one step
            steps_to_do = 1
            self._last_update_time = current_time
        else:
            # Calculate elapsed time since last update
            elapsed_time = current_time - self._last_update_time
            self._last_update_time = current_time
            
            # Calculate how many physics steps we should have done
            steps_to_do = int(elapsed_time / self.config.timestep)
            
            # Limit steps to prevent freezing if too much time has passed
            if steps_to_do > self._max_steps_per_update:
                logger.warning(
                    f"Too many steps ({steps_to_do}) needed, limiting to {self._max_steps_per_update}. "
                    f"Elapsed time: {elapsed_time:.3f}s, timestep: {self.config.timestep:.3f}s"
                )
                steps_to_do = self._max_steps_per_update
            elif steps_to_do < 1:
                # Not enough time has passed for even one step
                steps_to_do = 1

        # Execute physics steps with timing
        mj_step_start_time = time.perf_counter()
        for _ in range(steps_to_do):
            mujoco.mj_step(self.model, self.data)
        mj_step_end_time = time.perf_counter()
        mj_step_duration = mj_step_end_time - mj_step_start_time
        
        if self.config.log_data == True:
            logger.info(f"mj_step loop: {steps_to_do} steps took {mj_step_duration:.6f}s (avg {mj_step_duration/steps_to_do:.6f}s per step)")

        # Sync viewer and render with timing
        if self.viewer is not None:
            self.viewer.sync()
            
        update_scene_start_time = time.perf_counter()
        self.renderer.update_scene(self.data, self.cam)
        update_scene_end_time = time.perf_counter()
        update_scene_duration = update_scene_end_time - update_scene_start_time
        
        render_start_time = time.perf_counter()
        rgb = self.renderer.render()
        render_end_time = time.perf_counter()
        render_duration = render_end_time - render_start_time
        
        if self.config.log_data == True:
            logger.info(f"update_scene took {update_scene_duration:.6f}s")
            logger.info(f"render took {render_duration:.6f}s")
        
        update_end_time = time.perf_counter()
        total_update_duration = update_end_time - update_start_time
        if self.config.log_data == True:
            logger.info(f"Total update function took {total_update_duration:.6f}s")

        return rgb
    
    def stop(self):
        self.renderer.close()
        if self.viewer is not None:
            self.viewer.close()
