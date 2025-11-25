"""To enable `robodriver.__version__`"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("robodriver")
except PackageNotFoundError:
    __version__ = "unknown"
