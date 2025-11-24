import logging
import sys
import threading
import time

from dora import Node

logger = logging.getLogger(__name__)
node = Node()
dora_stop_event = threading.Event()
survive_close_event = threading.Event()


def test_function() -> None:
    logger.info("Starting test thread.")

    try:
        while not dora_stop_event.is_set():
            time.sleep(0.001)
    except Exception as e:
        logger.exception("Survive error: %s", e)
        survive_close_event.set()
    finally:
        logger.info("Closing pysurvive context.")


def dora_function() -> None:
    logger.info("Starting Dora thread.")
    try:
        for event in node:
            if dora_stop_event.is_set() or survive_close_event.is_set():
                logger.info("Dora loop received stop signal.")
                break

            if event["type"] == "INPUT" and event["id"] == "tick":
                time.sleep(0.01)

            elif event["type"] == "STOP":
                dora_stop_event.set()
                break

    except Exception as e:
        logger.exception("Dora error: %s", e)
    finally:
        logger.info("Exiting Dora thread.")


def main() -> None:
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    # 启动线程
    my_thread = threading.Thread(
        target=test_function,
        daemon=False,
    )
    dora_thread = threading.Thread(
        target=dora_function,
        daemon=False,
    )

    my_thread.start()
    dora_thread.start()

    logger.info("All threads started.")

    try:
        # 等待线程结束（设置超时避免卡死）
        my_thread.join(timeout=5)
        dora_thread.join(timeout=5)

        # 强制退出
        if my_thread.is_alive():
            logger.warning("Survive thread did not terminate, forcing exit.")
        if dora_thread.is_alive():
            logger.warning("Dora thread did not terminate, forcing exit.")
    except KeyboardInterrupt:
        logger.info("Main thread received Ctrl+C again. Forcing exit.")
        sys.exit(1)
    finally:
        logger.info("Program exiting gracefully.")
        sys.exit(0)


if __name__ == "__main__":
    main()
