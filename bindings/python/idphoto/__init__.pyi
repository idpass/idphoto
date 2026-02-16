"""Type stubs for the idphoto identity photo processing library."""

from typing import Literal, Optional

__version__: str

# Exception hierarchy
class IdPhotoError(Exception): ...
class DecodeError(IdPhotoError): ...
class UnsupportedFormatError(IdPhotoError): ...
class ZeroDimensionsError(IdPhotoError): ...
class EncodeError(IdPhotoError): ...
class InvalidQualityError(IdPhotoError): ...
class InvalidMaxDimensionError(IdPhotoError): ...

class FaceBounds:
    x: float
    y: float
    width: float
    height: float
    confidence: float

class CompressResult:
    data: bytes
    format: str
    width: int
    height: int
    original_size: int
    face_bounds: Optional[FaceBounds]

class FitResult:
    data: bytes
    format: str
    width: int
    height: int
    original_size: int
    face_bounds: Optional[FaceBounds]
    quality_used: float
    reached_target: bool

def compress(
    data: bytes,
    *,
    preset: Optional[Literal["qr-code", "qr-code-match", "print", "display"]] = None,
    max_dimension: Optional[int] = None,
    quality: Optional[float] = None,
    grayscale: Optional[bool] = None,
    crop_mode: Optional[Literal["heuristic", "none", "face-detection"]] = None,
    output_format: Optional[Literal["webp", "jpeg"]] = None,
    face_margin: Optional[float] = None,
) -> CompressResult: ...

def compress_to_fit(
    data: bytes,
    max_bytes: int,
    *,
    preset: Optional[Literal["qr-code", "qr-code-match", "print", "display"]] = None,
    max_dimension: Optional[int] = None,
    grayscale: Optional[bool] = None,
    crop_mode: Optional[Literal["heuristic", "none", "face-detection"]] = None,
    output_format: Optional[Literal["webp", "jpeg"]] = None,
    face_margin: Optional[float] = None,
) -> FitResult: ...
