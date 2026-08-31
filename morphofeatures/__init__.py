"""MorphoFeatures scientific embedding toolkit."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("morphofeatures")
except PackageNotFoundError:
    __version__ = "0.2.0"

__all__ = ["__version__"]
