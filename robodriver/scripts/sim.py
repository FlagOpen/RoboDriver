import asyncio
from dataclasses import asdict, dataclass
from pprint import pformat
from typing import List
import time
import numpy as np

import cv2
import logging_mp
from robodriver.core.simulator import SimulatorConfig
from robodriver.core.simulator import Simulator
from robodriver.utils import parser
from robodriver.utils.import_utils import register_third_party_devices
from robodriver.utils.utils import git_branch_log

logging_mp.basic_config(level=logging_mp.INFO)
logger = logging_mp.get_logger(__name__)


@dataclass
class SimOnlyConfig:
    """Configuration for simulation-only mode"""
    sim: SimulatorConfig
    
    @classmethod
    def __get_path_fields__(cls) -> List[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return []


def generate_test_action(num_joints: int = 7) -> dict:
    """
    Generate a simple test action for simulation.
    Returns a dictionary with joint positions.
    """
    # Simple sinusoidal motion for testing
    t = time.time()
    action = {}
    for i in range(num_joints):
        # Different frequency for each joint
        freq = 0.5 + i * 0.1
        # Different amplitude for each joint
        amp = 30.0 * (1.0 - i * 0.1)  # degrees
        # Base position
        base = 0.0
        # Sinusoidal motion
        value = base + amp * np.sin(2 * np.pi * freq * t)
        action[f"leader_joint{i+1}.pos"] = value
    
    return action


@parser.wrap()
async def async_main(cfg: SimOnlyConfig):
    logger.info(pformat(asdict(cfg)))

    # Create simulator
    sim = Simulator(cfg.sim) if cfg.sim is not None and cfg.sim.xml_path is not None else None
    
    if sim is None:
        logger.error("Simulator configuration is missing or invalid. Please provide a valid XML path.")
        return
    
    # Start the simulator
    sim.start()
    
    try:
        while True:
            # Generate test action
            # Get the number of actuators from the model
            # num_actuators = sim.model.nu if hasattr(sim, 'model') and hasattr(sim.model, 'nu') else 7
            action = generate_test_action(6)
            
            # Send action to simulator
            sim.send_action(action, prefix="leader_", suffix=".pos")
            
            # Get rendered image
            observation_sim = sim.get_render_image()
            
            # Display the image if available
            if observation_sim is not None:
                # Convert from RGB to BGR for OpenCV display
                img = cv2.cvtColor(observation_sim, cv2.COLOR_RGB2BGR)
                cv2.imshow("Simulation", img)
                cv2.waitKey(1)
            
            # Small delay to control loop frequency
            await asyncio.sleep(0.01)
            
    except KeyboardInterrupt:
        logger.info("Simulation stopped by user")
    finally:
        # Cleanup
        cv2.destroyAllWindows()
        if sim is not None:
            sim.stop()


def main():
    git_branch_log()
    
    register_third_party_devices()
    
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
