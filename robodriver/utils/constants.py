import logging
import os
from pathlib import Path

DEFAULT_FPS = 30
RERUN_WEB_PORT = 9195
RERUN_WS_PORT = 9285

OBS_ENV = "observation.environment_state"
OBS_ROBOT = "observation.state"
OBS_IMAGE = "observation.image"
OBS_IMAGES = "observation.images"
ACTION = "action"

# files & directories
CHECKPOINTS_DIR = "checkpoints"
LAST_CHECKPOINT_LINK = "last"
PRETRAINED_MODEL_DIR = "pretrained_model"
TRAINING_STATE_DIR = "training_state"
RNG_STATE = "rng_state.safetensors"
TRAINING_STEP = "training_step.json"
OPTIMIZER_STATE = "optimizer_state.safetensors"
OPTIMIZER_PARAM_GROUPS = "optimizer_param_groups.json"
SCHEDULER_STATE = "scheduler_state.json"

user_home = Path.home()
ROBODRIVER_HOME = Path(os.getenv("ROBODRIVER_HOME", str(user_home / "RoboDriver"))).expanduser().resolve()


if "ROBODRIVER_HOME" not in os.environ:
    logging.info(
        f"Environment variable 'ROBODRIVER_HOME' not set. Using default path: {ROBODRIVER_HOME}"
    )
else:
    logging.info(f"Environment variable 'ROBODRIVER_HOME' is set to: {ROBODRIVER_HOME}")

try:
    ROBODRIVER_HOME.mkdir(parents=True, exist_ok=True)
    logging.info(f"Directory ready: {ROBODRIVER_HOME}")
except PermissionError as e:
    logging.error(f"Permission denied! Cannot create {ROBODRIVER_HOME}")
    logging.error(f"Please check permissions for: {user_home}")
    logging.error(f"Or set ROBODRIVER_HOME environment variable to a writable location")
    raise
except Exception as e:
    logging.error(f"Failed to create directory: {e}")
    raise

DOROBOT_DATASET = ROBODRIVER_HOME / "dataset"
