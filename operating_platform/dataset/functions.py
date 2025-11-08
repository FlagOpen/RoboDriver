import logging
import packaging.version
from huggingface_hub.errors import RevisionNotFoundError

from operating_platform.dataset.backward_compatibility import (
    V21_MESSAGE,
    BackwardCompatibilityError,
    ForwardCompatibilityError,
)
from operating_platform.robot.robots.utils import Robot
from operating_platform.utils.dataset import get_repo_versions, DEFAULT_FEATURES
from typing import Union, Optional, List


logger = logging.getLogger(__name__)


def check_version_compatibility(
    repo_id: str,
    version_to_check: Union[str, packaging.version.Version],
    current_version: Union[str, packaging.version.Version],
    enforce_breaking_major: bool = True,
) -> None:
    v_check = (
        packaging.version.parse(version_to_check)
        if not isinstance(version_to_check, packaging.version.Version)
        else version_to_check
    )
    v_current = (
        packaging.version.parse(current_version)
        if not isinstance(current_version, packaging.version.Version)
        else current_version
    )
    if v_check.major < v_current.major and enforce_breaking_major:
        raise BackwardCompatibilityError(repo_id, v_check)
    elif v_check.minor < v_current.minor:
        logger.warning(V21_MESSAGE.format(repo_id=repo_id, version=v_check))

def get_features_from_robot(robot: Robot, use_videos: bool = True) -> dict:
    camera_ft = {}
    if robot.cameras:
        camera_ft = {
            key: {"dtype": "video" if use_videos else "image", **ft}
            for key, ft in robot.camera_features.items()
        }
    
    microphone_ft = {}
    if robot.microphones:
        microphone_ft = {
            key: {"dtype": "audio", **ft}
            for key, ft in robot.microphone_features.items()
        }
    return {**robot.motor_features, **camera_ft, **microphone_ft, **DEFAULT_FEATURES}

def get_safe_version(repo_id: str, version: Union[str, packaging.version.Version]) -> str:
    """
    Returns the version if available on repo or the latest compatible one.
    Otherwise, will throw a `CompatibilityError`.
    """
    target_version = (
        packaging.version.parse(version) if not isinstance(version, packaging.version.Version) else version
    )
    hub_versions = get_repo_versions(repo_id)

    if not hub_versions:
        raise RevisionNotFoundError(
            f"""Your dataset must be tagged with a codebase version.
            Assuming _version_ is the codebase_version value in the info.json, you can run this:
            ```python
            from huggingface_hub import HfApi

            hub_api = HfApi()
            hub_api.create_tag("{repo_id}", tag="_version_", repo_type="dataset")
            ```
            """
        )

    if target_version in hub_versions:
        return f"v{target_version}"

    compatibles = [
        v for v in hub_versions if v.major == target_version.major and v.minor <= target_version.minor
    ]
    if compatibles:
        return_version = max(compatibles)
        if return_version < target_version:
            logging.warning(f"Revision {version} for {repo_id} not found, using version v{return_version}")
        return f"v{return_version}"

    lower_major = [v for v in hub_versions if v.major < target_version.major]
    if lower_major:
        raise BackwardCompatibilityError(repo_id, max(lower_major))

    upper_versions = [v for v in hub_versions if v > target_version]
    assert len(upper_versions) > 0
    raise ForwardCompatibilityError(repo_id, min(upper_versions))
