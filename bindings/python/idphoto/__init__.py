from .idphoto import *  # noqa: F401, F403
from .idphoto import __version__  # noqa: F401

__all__ = [
    "__version__",
    "compress",
    "compress_to_fit",
    "CompressResult",
    "FitResult",
    "FaceBounds",
    "IdPhotoError",
    "DecodeError",
    "UnsupportedFormatError",
    "ZeroDimensionsError",
    "EncodeError",
    "InvalidQualityError",
    "InvalidMaxDimensionError",
]
