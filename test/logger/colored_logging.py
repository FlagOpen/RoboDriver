# colored_logging.py

import logging
import colorama
import os
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from typing import Optional, TextIO, Union

# 初始化 colorama（仅需调用一次）
colorama.init(autoreset=False)


class ColoredFormatter(logging.Formatter):
    """
    支持按日志级别着色的 Formatter
    """
    COLORS = {
        'DEBUG': '\033[36m',    # 青色
        'INFO': '\033[32m',     # 绿色
        'WARNING': '\033[33m',  # 黄色
        'ERROR': '\033[31m',    # 红色
        'CRITICAL': '\033[1;35m', # 加粗紫色
    }
    RESET = '\033[0m'

    def __init__(self, colored=True, fmt=None, datefmt=None):
        super().__init__(fmt, datefmt)
        self.colored = colored
        self.default_format = "%(asctime)s - %(name)s - %(filename)s:%(lineno)d - %(funcName)s() - %(levelname)s - %(message)s"
        self.datefmt = datefmt or "%Y-%m-%d %H:%M:%S"
        self.fmt = fmt or self.default_format

    def format(self, record):
        if self.colored:
            log_color = self.COLORS.get(record.levelname, self.RESET)
            fmt = self.fmt.replace('%(levelname)s', f'{log_color}%(levelname)s{self.RESET}')
        else:
            fmt = self.fmt

        formatter = logging.Formatter(fmt, datefmt=self.datefmt)
        return formatter.format(record)


def setup_colored_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    stream: Optional[TextIO] = None,
    colored: bool = True,
    fmt: Optional[str] = None,
    datefmt: Optional[str] = None,
    log_file: Optional[str] = None,
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    when: Optional[str] = None,          # 如 'D', 'midnight', 'H' 等，用于 TimedRotating
) -> logging.Logger:
    """
    快速设置一个带颜色的日志记录器（支持输出到控制台 + 文件）

    :param name: logger 名称，默认为调用模块名（推荐传入 __name__）
    :param level: 日志级别，默认 INFO
    :param stream: 输出流，默认 sys.stderr
    :param colored: 控制台是否启用颜色，默认 True
    :param fmt: 自定义日志格式
    :param datefmt: 自定义时间格式
    :param log_file: 日志文件路径，如 "app.log"，不传则不写入文件
    :param max_bytes: 每个日志文件最大大小（字节），默认 10MB
    :param backup_count: 保留的备份文件数，默认 5
    :param when: 按时间轮转，如 'D'（天）、'midnight'、'H'（小时）等，与 max_bytes 二选一
    :return: 配置好的 logger 实例
    """
    logger = logging.getLogger(name or __name__)
    logger.setLevel(level)

    # 避免重复添加 handler
    if logger.handlers:
        return logger

    # === 控制台 Handler ===
    ch = logging.StreamHandler(stream)
    console_formatter = ColoredFormatter(colored=colored, fmt=fmt, datefmt=datefmt)
    ch.setFormatter(console_formatter)
    logger.addHandler(ch)

    # === 文件 Handler（可选）===
    if log_file:
        # 创建日志目录（如果不存在）
        os.makedirs(os.path.dirname(log_file) if os.path.dirname(log_file) else '.', exist_ok=True)

        if when:
            # 按时间轮转
            fh = TimedRotatingFileHandler(
                filename=log_file,
                when=when,
                backupCount=backup_count,
                encoding='utf-8'
            )
        else:
            # 按大小轮转
            fh = RotatingFileHandler(
                filename=log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )

        # 文件日志不着色，使用纯文本格式
        file_fmt = fmt or "%(asctime)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(file_fmt, datefmt=datefmt or "%Y-%m-%d %H:%M:%S")
        fh.setFormatter(file_formatter)
        logger.addHandler(fh)

    return logger


# 全局默认 logger（可选，方便快速调用）
_default_logger = None


def get_logger(
    name: Optional[str] = None,
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    **kwargs
) -> logging.Logger:
    """
    获取一个默认配置的彩色 logger（懒加载 + 支持文件日志）
    """
    global _default_logger
    if _default_logger is None:
        _default_logger = setup_colored_logger(
            name=name or "default",
            level=level,
            log_file=log_file,
            **kwargs
        )
    return _default_logger


# 别名，方便直接调用 info/warn/error
def info(msg, *args, **kwargs):
    get_logger().info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    get_logger().warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    get_logger().error(msg, *args, **kwargs)

def debug(msg, *args, **kwargs):
    get_logger().debug(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    get_logger().critical(msg, *args, **kwargs)