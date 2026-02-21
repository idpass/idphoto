"""Type stubs for the idphoto identity photo processing library."""

from enum import Enum
from typing import Literal, Optional, Union

__version__: str

# Exception hierarchy
class IdPhotoError(Exception): ...
class DecodeError(IdPhotoError): ...
class UnsupportedFormatError(IdPhotoError): ...
class ZeroDimensionsError(IdPhotoError): ...
class EncodeError(IdPhotoError): ...
class InvalidQualityError(IdPhotoError): ...
class InvalidMaxDimensionError(IdPhotoError): ...

class Preset(str, Enum):
    QR_CODE: Literal["qr-code"]
    QR_CODE_MATCH: Literal["qr-code-match"]
    PRINT: Literal["print"]
    DISPLAY: Literal["display"]

class CropMode(str, Enum):
    HEURISTIC: Literal["heuristic"]
    NONE: Literal["none"]
    FACE_DETECTION: Literal["face-detection"]

class OutputFormat(str, Enum):
    WEBP: Literal["webp"]
    JPEG: Literal["jpeg"]

PresetLiteral = Literal["qr-code", "qr-code-match", "print", "display"]
CropModeLiteral = Literal["heuristic", "none", "face-detection"]
OutputFormatLiteral = Literal["webp", "jpeg"]

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

class CompressOptions:
    preset: Optional[Union[Preset, PresetLiteral]]
    max_dimension: Optional[int]
    quality: Optional[float]
    grayscale: Optional[bool]
    crop_mode: Optional[Union[CropMode, CropModeLiteral]]
    output_format: Optional[Union[OutputFormat, OutputFormatLiteral]]
    face_margin: Optional[float]
    def __init__(
        self,
        preset: Optional[Union[Preset, PresetLiteral]] = ...,
        max_dimension: Optional[int] = ...,
        quality: Optional[float] = ...,
        grayscale: Optional[bool] = ...,
        crop_mode: Optional[Union[CropMode, CropModeLiteral]] = ...,
        output_format: Optional[Union[OutputFormat, OutputFormatLiteral]] = ...,
        face_margin: Optional[float] = ...,
    ) -> None: ...

class CompressToFitOptions:
    preset: Optional[Union[Preset, PresetLiteral]]
    max_dimension: Optional[int]
    grayscale: Optional[bool]
    crop_mode: Optional[Union[CropMode, CropModeLiteral]]
    output_format: Optional[Union[OutputFormat, OutputFormatLiteral]]
    face_margin: Optional[float]
    def __init__(
        self,
        preset: Optional[Union[Preset, PresetLiteral]] = ...,
        max_dimension: Optional[int] = ...,
        grayscale: Optional[bool] = ...,
        crop_mode: Optional[Union[CropMode, CropModeLiteral]] = ...,
        output_format: Optional[Union[OutputFormat, OutputFormatLiteral]] = ...,
        face_margin: Optional[float] = ...,
    ) -> None: ...

def compress(
    data: bytes,
    *,
    options: Optional[CompressOptions] = None,
    preset: Optional[Union[Preset, PresetLiteral]] = None,
    max_dimension: Optional[int] = None,
    quality: Optional[float] = None,
    grayscale: Optional[bool] = None,
    crop_mode: Optional[Union[CropMode, CropModeLiteral]] = None,
    output_format: Optional[Union[OutputFormat, OutputFormatLiteral]] = None,
    face_margin: Optional[float] = None,
) -> CompressResult: ...

def compress_to_fit(
    data: bytes,
    max_bytes: int,
    *,
    options: Optional[CompressToFitOptions] = None,
    preset: Optional[Union[Preset, PresetLiteral]] = None,
    max_dimension: Optional[int] = None,
    grayscale: Optional[bool] = None,
    crop_mode: Optional[Union[CropMode, CropModeLiteral]] = None,
    output_format: Optional[Union[OutputFormat, OutputFormatLiteral]] = None,
    face_margin: Optional[float] = None,
) -> FitResult: ...

class IdPhoto:
    @staticmethod
    def compress(
        data: bytes,
        *,
        options: Optional[CompressOptions] = None,
        preset: Optional[Union[Preset, PresetLiteral]] = None,
        max_dimension: Optional[int] = None,
        quality: Optional[float] = None,
        grayscale: Optional[bool] = None,
        crop_mode: Optional[Union[CropMode, CropModeLiteral]] = None,
        output_format: Optional[Union[OutputFormat, OutputFormatLiteral]] = None,
        face_margin: Optional[float] = None,
    ) -> CompressResult: ...

    @staticmethod
    def compress_to_fit(
        data: bytes,
        max_bytes: int,
        *,
        options: Optional[CompressToFitOptions] = None,
        preset: Optional[Union[Preset, PresetLiteral]] = None,
        max_dimension: Optional[int] = None,
        grayscale: Optional[bool] = None,
        crop_mode: Optional[Union[CropMode, CropModeLiteral]] = None,
        output_format: Optional[Union[OutputFormat, OutputFormatLiteral]] = None,
        face_margin: Optional[float] = None,
    ) -> FitResult: ...
