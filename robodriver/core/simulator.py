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

        self.jnt_names = [
            'joint1',
            'joint2',
            'joint3',
            'joint4',
            'joint5',
            'joint6',
            'joint7',
            'finger_joint1',
            'finger_joint2',
        ]
        dofs_idx = [self.arm.get_joint(name).dof_idx_local for name in self.jnt_names]

        self.scene.build()

        self.arm.control_dofs_position(
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            dofs_idx,
        )

    def update(self, action):
        goal_joint = [ val for _key, val in action.items()]

        goal_joint_numpy = np.array(goal_joint, dtype=np.float32)

        self.arm.control_dofs_position(
            goal_joint_numpy,
        )

        self.scene.step()
        rgb, _, _, _ = self.cam.render()

        return rgb