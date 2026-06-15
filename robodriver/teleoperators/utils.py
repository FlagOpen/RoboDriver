from typing import cast

import logging_mp
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

from robodriver.utils.import_utils import make_device_from_device_class

logger = logging_mp.getLogger(__name__)


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    logger.info("In make_teleoperator_from_config")
    logger.info(f"make teleoperator type: {config.type}")
    
    try:    
        return cast(Teleoperator, make_device_from_device_class(config))
    except Exception as e:
        logger.critical(f"Can't create teleoperator with config {config}")
        raise ValueError(
            f"Error creating teleoperator with config {config}: {e}"
        ) from e
