# your_script.py
from colored_logging import setup_colored_logger

logger = setup_colored_logger(__name__)

logger.info("启动程序 ✅")
logger.warning("注意：内存使用较高 ⚠️")
logger.error("发生错误 ❌")