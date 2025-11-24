import logging

from dora import Node

logger = logging.getLogger(__name__)


def main() -> None:
    sn_set = set()  # 使用集合来避免重复
    try:
        node = Node()
        logger.info("Starting to listen for events...")

        for event in node:
            if event["type"] == "INPUT":
                # 安全处理输入事件
                if event["value"]:
                    struct = event["value"][0]
                    serial_number = struct.get("serial_number").as_py()

                    if serial_number:
                        if serial_number not in sn_set:
                            sn_set.add(serial_number)
                            logger.info(f"Found new serial number: {serial_number}")
                        else:
                            logger.debug(
                                f"Serial number already recorded: {serial_number}"
                            )
                    else:
                        logger.warning("Received input without serial number field")
                else:
                    logger.warning("Received empty input value")

            elif event["type"] in ["STOP", "INPUT_CLOSED"]:
                logger.info("Stopping event listener...")
                break

        # 最终输出收集到的所有序列号
        if sn_set:
            logger.info(f"Collected {len(sn_set)} unique serial numbers:")
            for sn in sorted(sn_set):  # 排序只是为了输出美观
                logger.info(f" - {sn}")
        else:
            logger.info("No serial numbers were collected")

    except KeyboardInterrupt:
        logger.info("\nReceived keyboard interrupt, exiting...")
    except KeyError as e:
        logger.error(f"Missing expected field in input data: {e}", exc_info=True)
    except Exception as e:
        logger.error(f"Unexpected error occurred: {e}", exc_info=True)
        raise

    logger.info("dora_vive_sn terminated normally")


if __name__ == "__main__":
    # 配置日志格式
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    main()
