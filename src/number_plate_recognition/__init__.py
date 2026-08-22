"""Production package for Moroccan number-plate recognition.

The package root intentionally avoids importing the computer-vision runtime so
metadata and operational tooling stay lightweight.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("number-plate-recognition")
except PackageNotFoundError:  # source tree before installation
    __version__ = "0.1.0"

__all__ = ["__version__"]
