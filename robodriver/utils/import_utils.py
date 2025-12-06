#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import importlib
import pkgutil
import re
from typing import Any

import logging_mp
from draccus.choice_types import ChoiceRegistry

logger = logging_mp.get_logger(__name__)


def make_device_from_device_class(config: ChoiceRegistry) -> Any:
    """
    Dynamically instantiates an object from its `ChoiceRegistry` configuration.

    This factory uses the module path and class name from the `config` object's
    type to locate and instantiate the corresponding device class (not the config).
    It derives the device class name by removing a trailing 'Config' from the config
    class name and tries a few candidate modules where the device implementation is
    commonly located.
    """
    if not isinstance(config, ChoiceRegistry):
        raise ValueError(
            f"Config should be an instance of `ChoiceRegistry`, got {type(config)}"
        )

    config_cls = config.__class__
    module_path = (
        config_cls.__module__
    )  # typical: lerobot_teleop_mydevice.config_mydevice
    config_name = config_cls.__name__  # typical: MyDeviceConfig

    # Derive device class name (strip "Config")
    if not config_name.endswith("Config"):
        raise ValueError(
            f"Config class name '{config_name}' does not end with 'Config'"
        )

    device_class_name = config_name[:-6]  # typical: MyDeviceConfig -> MyDevice

    # Build candidate modules to search for the device class
    parts = module_path.split(".")
    parent_module = ".".join(parts[:-1]) if len(parts) > 1 else module_path
    candidates = [
        parent_module,  # typical: lerobot_teleop_mydevice
        parent_module
        + "."
        + device_class_name.lower(),  # typical: lerobot_teleop_mydevice.mydevice
    ]

    # handle modules named like "config_xxx" -> try replacing that piece with "xxx"
    last = parts[-1] if parts else ""
    if last.startswith("config_"):
        candidates.append(".".join(parts[:-1] + [last.replace("config_", "")]))

    # de-duplicate while preserving order
    seen: set[str] = set()
    candidates = [c for c in candidates if not (c in seen or seen.add(c))]

    tried: list[str] = []
    for candidate in candidates:
        tried.append(candidate)
        try:
            module = importlib.import_module(candidate)
        except ImportError:
            continue

        if hasattr(module, device_class_name):
            cls = getattr(module, device_class_name)
            if callable(cls):
                try:
                    return cls(config)
                except TypeError as e:
                    raise TypeError(
                        f"Failed to instantiate '{device_class_name}' from module '{candidate}': {e}"
                    ) from e

    raise ImportError(
        f"Could not locate device class '{device_class_name}' for config '{config_name}'. "
        f"Tried modules: {tried}. Ensure your device class name is the config class name without "
        f"'Config' and that it's importable from one of those modules."
    )


def register_third_party_devices() -> None:
    """
    Discover and import third-party robodriver_* plugins so they can register themselves.

    Scans top-level modules on sys.path for packages starting with
    'robodriver_robot_', 'robodriver_camera_', "robodriver_policy_" or 'robodriver_teleoperator_' and imports them.
    """
    prefixes = (
        "robodriver_robot_",
        "robodriver_camera_",
        "robodriver_policy_",
        "robodriver_teleoperator",
    )
    imported: list[str] = []
    failed: list[str] = []

    for module_info in pkgutil.iter_modules():
        name = module_info.name
        if name.startswith(prefixes):
            _attempt_import(name, imported, failed)

    editable_pattern = re.compile(r"__editable___(.+?)_\d+_\d+_\d+_finder")
    for module_info in pkgutil.iter_modules():
        match = editable_pattern.match(module_info.name)
        if match:
            original_name = match.group(1).replace(
                "_", "-"
            )  # Convert to canonical package name
            if any(
                original_name.startswith(prefix.replace("_", "-"))
                for prefix in prefixes
            ):
                # Convert back to module name format (replace - with _)
                module_name = original_name.replace("-", "_")
                _attempt_import(module_name, imported, failed)

    logger.debug(
        "Third-party plugin import summary: imported=%s failed=%s", imported, failed
    )


def _attempt_import(module_name: str, imported: list, failed: list) -> None:
    """Helper to attempt import and update tracking lists."""
    try:
        importlib.import_module(module_name)
        imported.append(module_name)
        logger.info("✅ Successfully imported plugin: %s", module_name)
    except Exception as e:
        failed.append(module_name)
        logger.warning("❌ FAILED to import plugin: %s - %s", module_name, e)
