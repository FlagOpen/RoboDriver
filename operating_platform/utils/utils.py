# import logging
import json
import platform
import subprocess
from copy import copy
from datetime import datetime, timezone
from typing import Dict, Union

import logging_mp
import numpy as np
import torch

logger = logging_mp.get_logger(__name__)

# import colored_logging as clog

# import operating_platform.utils.colored_logging as clog


def cameras_to_stream_json(cameras: Dict[str, int]):
    """
    将摄像头字典转换为包含流信息的 JSON 字符串。

    参数:
        cameras (dict[str, int]): 摄像头名称到 ID 的映射

    返回:
        str: 格式化的 JSON 字符串
    """
    stream_list = [{"id": cam_id, "name": name} for name, cam_id in cameras.items()]
    # 修改depth
    result = {"total": len(stream_list), "streams": stream_list}
    return json.dumps(result)


# class CustomFormatter(logging.Formatter):
#     """自定义日志格式化器"""
#     def format(self, record):
#         dt = datetime.now().strftime("%Y-%m-%d %H:%M:%S")  # 可改为 datetime.utcnow() 或带时区时间
#         fnameline = f"{record.pathname}:{record.lineno}"
#         return f"{record.levelname} {dt} {fnameline[-15:]:>15} {record.msg}"

# def init_logging(level=logging.DEBUG, force=False):
#     """
#     初始化日志配置
#     :param level: 日志级别
#     :param force: 是否强制重新配置（默认 False）
#     """
#     logger = logging.getLogger()

#     # 避免重复初始化
#     if not force and logger.handlers:
#         return

#     # 移除所有已存在的 Handler（可选）
#     for handler in logger.handlers[:]:
#         logger.removeHandler(handler)

#     # 创建并添加新的 Handler
#     formatter = CustomFormatter()
#     console_handler = logging.StreamHandler()
#     console_handler.setFormatter(formatter)
#     logger.addHandler(console_handler)
#     logger.setLevel(level)


def auto_select_torch_device() -> torch.device:
    """Tries to select automatically a torch device."""
    if torch.cuda.is_available():
        logger.info("Cuda backend detected, using cuda.")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        logger.info("Metal backend detected, using cuda.")
        return torch.device("mps")
    else:
        logger.warning(
            "No accelerated backend detected. Using default cpu, this will be slow."
        )
        return torch.device("cpu")


def get_container_ip_from_hosts():
    """通过 /etc/hosts 获取容器 IP（Docker 默认写入）"""
    hostname = None
    with open("/proc/sys/kernel/hostname", "r") as f:
        hostname = f.read().strip()

    with open("/etc/hosts", "r") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) > 1 and parts[1] == hostname:
                return parts[0]

    raise RuntimeError("无法从 /etc/hosts 获取 IP")


# TODO(Steven): Remove log. log shouldn't be an argument, this should be handled by the logger level
def get_safe_torch_device(try_device: str, log: bool = False) -> torch.device:
    """Given a string, return a torch.device with checks on whether the device is available."""
    try_device = str(try_device)

    if try_device == "cuda":
        if not torch.cuda.is_available():
            raise AssertionError("CUDA is not available.")
        device = torch.device("cuda")
    elif try_device == "mps":
        if not torch.backends.mps.is_available():
            raise AssertionError("MPS is not available.")
        device = torch.device("mps")
    elif try_device == "cpu":
        device = torch.device("cpu")
        if log:
            logger.warning("Using CPU, this will be slow.")
    else:
        # Try to create a device with the given string (e.g., "cuda:1", "xla:0", etc.)
        device = torch.device(try_device)
        if log:
            logger.warning(f"Using custom device: {try_device}.")

    return device


def get_safe_dtype(dtype: torch.dtype, device: Union[str, torch.device]):
    """
    mps is currently not compatible with float64
    """
    if isinstance(device, torch.device):
        device = device.type
    if device == "mps" and dtype == torch.float64:
        return torch.float32
    else:
        return dtype


def git_branch_log():
    current_branch = get_current_git_branch()

    # 分支匹配规则：优先级从上到下
    branch_patterns = {
        "main": ("🚀", "正在主分支上运行"),
        "release": ("📦", "正在正式分支上运行"),
        "dev": ("🛠️", "正在开发分支上运行"),
        "test": ("🧪", "正在测试分支上运行"),
        "debug": ("🐞", "正在调试分支上运行"),
    }

    if not current_branch:
        logger.warning("❓ 当前分支: 未知分支")
        return

    current_branch = current_branch.lower()

    for branch, (emoji, message) in branch_patterns.items():
        if branch == current_branch:
            logger.info(f"{emoji} {message}")
            return

    # 如果没有精确匹配，再模糊匹配一次
    for branch, (emoji, message) in branch_patterns.items():
        if branch in current_branch:
            logger.info(f"{emoji} 正在包含 '{branch}' 的分支上运行")
            return

    # 完全未知的分支
    logger.warning(f"❓ 当前分支: {current_branch}")


def get_current_git_branch():
    """获取当前 Git 分支名称"""
    try:
        # 执行 git 命令获取当前分支
        branch = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            stderr=subprocess.STDOUT,
            universal_newlines=True,
        ).strip()
        return branch
    except (subprocess.CalledProcessError, FileNotFoundError):
        # 处理异常情况（非 git 仓库或未安装 git）
        return None


def is_torch_device_available(try_device: str) -> bool:
    try_device = str(try_device)  # Ensure try_device is a string
    if try_device == "cuda":
        return torch.cuda.is_available()
    elif try_device == "mps":
        return torch.backends.mps.is_available()
    elif try_device == "cpu":
        return True
    else:
        raise ValueError(
            f"Unknown device {try_device}. Supported devices are: cuda, mps or cpu."
        )


def is_amp_available(device: str):
    if device in ["cuda", "cpu"]:
        return True
    elif device == "mps":
        return False
    else:
        raise ValueError(f"Unknown device '{device}.")


######################################################### old ################
# def none_or_int(value):
#     if value == "None":
#         return None
#     return int(value)


# def inside_slurm():
#     """Check whether the python process was launched through slurm"""
#     # TODO(rcadene): return False for interactive mode `--pty bash`
#     return "SLURM_JOB_ID" in os.environ


# def format_big_number(num, precision=0):
#     suffixes = ["", "K", "M", "B", "T", "Q"]
#     divisor = 1000.0

#     for suffix in suffixes:
#         if abs(num) < divisor:
#             return f"{num:.{precision}f}{suffix}"
#         num /= divisor

#     return num


# def _relative_path_between(path1: Path, path2: Path) -> Path:
#     """Returns path1 relative to path2."""
#     path1 = path1.absolute()
#     path2 = path2.absolute()
#     try:
#         return path1.relative_to(path2)
#     except ValueError:  # most likely because path1 is not a subpath of path2
#         common_parts = Path(osp.commonpath([path1, path2])).parts
#         return Path(
#             "/".join([".."] * (len(path2.parts) - len(common_parts)) + list(path1.parts[len(common_parts) :]))
#         )


def print_cuda_memory_usage():
    """Use this function to locate and debug memory leak."""
    import gc

    gc.collect()
    # Also clear the cache if you want to fully release the memory
    torch.cuda.empty_cache()
    print(
        "Current GPU Memory Allocated: {:.2f} MB".format(
            torch.cuda.memory_allocated(0) / 1024**2
        )
    )
    print(
        "Maximum GPU Memory Allocated: {:.2f} MB".format(
            torch.cuda.max_memory_allocated(0) / 1024**2
        )
    )
    print(
        "Current GPU Memory Reserved: {:.2f} MB".format(
            torch.cuda.memory_reserved(0) / 1024**2
        )
    )
    print(
        "Maximum GPU Memory Reserved: {:.2f} MB".format(
            torch.cuda.max_memory_reserved(0) / 1024**2
        )
    )


def capture_timestamp_utc():
    return datetime.now(timezone.utc)


def say(text, blocking=False):
    system = platform.system()

    if system == "Darwin":
        cmd = ["say", text]

    elif system == "Linux":
        cmd = ["spd-say", text]
        if blocking:
            cmd.append("--wait")

    elif system == "Windows":
        cmd = [
            "PowerShell",
            "-Command",
            "Add-Type -AssemblyName System.Speech; "
            f"(New-Object System.Speech.Synthesis.SpeechSynthesizer).Speak('{text}')",
        ]

    else:
        raise RuntimeError("Unsupported operating system for text-to-speech.")

    if blocking:
        subprocess.run(cmd, check=True)
    else:
        subprocess.Popen(
            cmd, creationflags=subprocess.CREATE_NO_WINDOW if system == "Windows" else 0
        )


def log_say(text, play_sounds, blocking=False):
    logger.info(text)

    if play_sounds:
        say(text, blocking)


def get_channel_first_image_shape(image_shape: tuple) -> tuple:
    shape = copy(image_shape)
    if shape[2] < shape[0] and shape[2] < shape[1]:  # (h, w, c) -> (c, h, w)
        shape = (shape[2], shape[0], shape[1])
    elif not (shape[0] < shape[1] and shape[0] < shape[2]):
        raise ValueError(image_shape)

    return shape


def has_method(cls: object, method_name: str) -> bool:
    return hasattr(cls, method_name) and callable(getattr(cls, method_name))


def is_valid_numpy_dtype_string(dtype_str: str) -> bool:
    """
    Return True if a given string can be converted to a numpy dtype.
    """
    try:
        # Attempt to convert the string to a numpy dtype
        np.dtype(dtype_str)
        return True
    except TypeError:
        # If a TypeError is raised, the string is not a valid dtype
        return False
