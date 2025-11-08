import logging
import colorama

colorama.init()  # 支持 Windows

class ColoredFormatter(logging.Formatter):
    COLORS = {
        'DEBUG': '\033[36m',
        'INFO': '\033[32m',
        'WARNING': '\033[33m',
        'ERROR': '\033[31m',
        'CRITICAL': '\033[35m',
    }
    RESET = '\033[0m'

    def format(self, record):
        log_color = self.COLORS.get(record.levelname, self.RESET)
        format_str = f"%(asctime)s - {log_color}%(levelname)s{self.RESET} - %(message)s"
        formatter = logging.Formatter(format_str, datefmt="%H:%M:%S")
        return formatter.format(record)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setFormatter(ColoredFormatter())
logger.addHandler(ch)

# 测试
logger.info("这是带颜色的 INFO 日志 ✅")
logger.warning("这是带颜色的 WARNING 日志 ⚠️")