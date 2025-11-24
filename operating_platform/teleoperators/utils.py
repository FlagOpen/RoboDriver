from typing import cast

import logging_mp
from lerobot.teleoperators import Teleoperator, TeleoperatorConfig

from operating_platform.utils.import_utils import make_device_from_device_class

logger = logging_mp.get_logger(__name__)


def make_teleoperator_from_config(config: TeleoperatorConfig) -> Teleoperator:
    logger.info("In make_teleoperator_from_config")

    if config.type == "so101_leader":
        raise ValueError("Teleoperator type is not available.")
    else:
        try:
            return cast(Teleoperator, make_device_from_device_class(config))
        except Exception as e:
            logger.critical(f"Can't create teleoperator with config {config}")
            raise ValueError(
                f"Error creating teleoperator with config {config}: {e}"
            ) from e
