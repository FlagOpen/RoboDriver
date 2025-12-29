from typing import Any
import numpy as np
import genesis as gs
import cv2
from pathlib import Path

from dataclasses import dataclass

@dataclass
class SimulatorConfig():
    urdf_path: str | None = None
    mjcf_path: str | None = None
    backend: str = "cpu"
    show_viewer: bool = False
    

class Simulator:
    def __init__(self, backend: str, show_viewer: bool, urdf_path: str | None, mjcf_path: str | None):
        backend_mapping = {
            "cpu": gs.cpu,
            "gpu": gs.gpu,
            "cuda": gs.cuda,
            "vulkan": gs.vulkan,
            "metal": gs.metal,
            "opengl": gs.opengl,
        }

        if backend not in backend_mapping:
            valid_backends = ", ".join(backend_mapping.keys())
            raise ValueError(f"Invalid backend '{backend}'. Valid options are: {valid_backends}")
    
        gs.init(backend=backend_mapping[backend], logging_level="warn")

        self.scene = gs.Scene(
            vis_options = gs.options.VisOptions(
                shadow = False,
                lights = [
                    {"type": "directional", "dir": (-1, -1, -1), "color": (1.0, 1.0, 1.0), "intensity": 1.5},
                    {"type": "directional", "dir": (-1, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 3.0},
                    {"type": "directional", "dir": (1, 1, -1), "color": (1.0, 1.0, 1.0), "intensity": 1.5}
                ]
            ),
            show_viewer=show_viewer,
        )

        _plane = self.scene.add_entity(
            gs.morphs.Plane(),
        )

        if urdf_path is not None:
            self.arm = self.scene.add_entity(
                gs.morphs.URDF(
                    file = urdf_path,
                    pos = (0, 0, 0),
                    fixed = True,
                ),
            )
        elif mjcf_path is not None:
            self.arm = self.scene.add_entity(
                gs.morphs.MJCF(
                    file = mjcf_path,
                ),
            )

        self.cam = self.scene.add_camera(
            res    = (1600, 1200),
            pos    = (-0.5, -0.5, 0.5),
            lookat = (0, 0, 0.1),
            fov    = 60,
            GUI    = False,
        )

        self.scene.build()

        # self.arm.control_dofs_position(
        #     np.zeros(self.arm.n_dofs),
        # )

    def update(self, action: dict[str, Any], prefix: str, suffix: str):
        print("action:", action)
        
        goal_joint = [ val for _key, val in action.items()]

        dofs_idx = [self.arm.get_joint(name.removeprefix(f"{prefix}").removesuffix(f"{suffix}")).dof_idx_local for name in action]
        print("dofs_idx:", dofs_idx)

        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)
        print("goal_joint_numpy:", goal_joint_numpy)


        # 假设 goal_joint 是角度值，需要转换为弧度
        goal_joint_degrees = np.array(goal_joint, dtype=np.float32)  # 角度值
        print("原始角度值 (deg):", goal_joint_degrees)

        # if 需要从角度转化弧度
        # 转换为弧度
        goal_joint_radians = goal_joint_degrees * (np.pi / 180.0)
        print("转换为弧度 (rad):", goal_joint_radians)



        self.arm.control_dofs_position(
            goal_joint_radians,
            dofs_idx,
        )

        self.scene.step()
        rgb, _, _, _ = self.cam.render()

        return rgb