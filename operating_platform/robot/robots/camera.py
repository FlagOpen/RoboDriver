from typing import Protocol

import numpy as np

from operating_platform.robot.robots.com_configs.cameras import (
    CameraConfig,
    IntelRealSenseCameraConfig,
    OpenCVCameraConfig,
)
from typing import Optional


class Camera(Protocol):
    def connect(self): ...
    def read(self, temporary_color: Optional[str] = None) -> np.ndarray: ...
    def async_read(self) -> np.ndarray: ...
    def disconnect(self): ...
