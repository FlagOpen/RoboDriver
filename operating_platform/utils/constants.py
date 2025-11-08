import os
from pathlib import Path
import logging

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

DOROBOT_HOME = Path(os.getenv("DOROBOT_HOME", "~/DoRobot")).expanduser().resolve()


if "DOROBOT_HOME" not in os.environ:
    logging.info(f"Environment variable 'DOROBOT_HOME' not set. Using default path: {DOROBOT_HOME}")
else:
    logging.info(f"Environment variable 'DOROBOT_HOME' is set to: {DOROBOT_HOME}")

if not DOROBOT_HOME.exists():
    DOROBOT_HOME.mkdir(parents=True, exist_ok=True)
    logging.info(f"Created directory: {DOROBOT_HOME}")

DOROBOT_DATASET = DOROBOT_HOME / "dataset"
