import platform
import sys
import select


def enter_pressed() -> bool:
    if platform.system() == "Windows":
        import msvcrt

        if msvcrt.kbhit():
            key = msvcrt.getch()
            return key in (b"\r", b"\n")  # enter key
        return False
    else:
        return (
            select.select([sys.stdin], [], [], 0)[0]
            and sys.stdin.readline().strip() == ""
        )


def move_cursor_up(lines):
    """Move the cursor up by a specified number of lines."""
    print(f"\033[{lines}A", end="")


class DeviceNotConnectedError(ConnectionError):
    """Exception raised when the device is not connected."""

    def __init__(
        self, message="This device is not connected. Try calling `connect()` first."
    ):
        self.message = message
        super().__init__(self.message)


class DeviceAlreadyConnectedError(ConnectionError):
    """Exception raised when the device is already connected."""

    def __init__(
        self,
        message="This device is already connected. Try not calling `connect()` twice.",
    ):
        self.message = message
        super().__init__(self.message)
