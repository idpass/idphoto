from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Mapping, Optional, Type, Union

from .idphoto import (
    CompressResult,
    DecodeError,
    EncodeError,
    FaceBounds,
    FitResult,
    IdPhotoError,
    InvalidMaxDimensionError,
    InvalidQualityError,
    UnsupportedFormatError,
    ZeroDimensionsError,
    __version__,
    compress as _compress_native,
    compress_to_fit as _compress_to_fit_native,
)


class Preset(str, Enum):
    """Preset configurations for common use cases."""

    QR_CODE = "qr-code"
    QR_CODE_MATCH = "qr-code-match"
    PRINT = "print"
    DISPLAY = "display"


class CropMode(str, Enum):
    """Crop strategy before resize."""

    HEURISTIC = "heuristic"
    NONE = "none"
    FACE_DETECTION = "face-detection"


class OutputFormat(str, Enum):
    """Output image format."""

    WEBP = "webp"
    JPEG = "jpeg"


_PresetLike = Union[Preset, str]
_CropModeLike = Union[CropMode, str]
_OutputFormatLike = Union[OutputFormat, str]


@dataclass(frozen=True)
class CompressOptions:
    """Typed options for :func:`compress`."""

    preset: Optional[_PresetLike] = None
    max_dimension: Optional[int] = None
    quality: Optional[float] = None
    grayscale: Optional[bool] = None
    crop_mode: Optional[_CropModeLike] = None
    output_format: Optional[_OutputFormatLike] = None
    face_margin: Optional[float] = None


@dataclass(frozen=True)
class CompressToFitOptions:
    """Typed options for :func:`compress_to_fit`."""

    preset: Optional[_PresetLike] = None
    max_dimension: Optional[int] = None
    grayscale: Optional[bool] = None
    crop_mode: Optional[_CropModeLike] = None
    output_format: Optional[_OutputFormatLike] = None
    face_margin: Optional[float] = None


def _normalize_enum(
    value: Optional[Union[str, Enum]],
    enum_cls: Type[Enum],
    label: str,
) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, enum_cls):
        return str(value.value)
    if isinstance(value, str):
        try:
            return str(enum_cls(value).value)
        except ValueError as exc:
            raise ValueError(f"unknown {label}: {value}") from exc
    raise TypeError(
        f"{label} must be a str or {enum_cls.__name__}, got {type(value).__name__}",
    )


def _compact_kwargs(raw: Mapping[str, object]) -> Dict[str, object]:
    return {key: value for key, value in raw.items() if value is not None}


def _compress_options_to_kwargs(options: Optional[CompressOptions]) -> Dict[str, object]:
    if options is None:
        return {}
    if not isinstance(options, CompressOptions):
        raise TypeError(
            f"options must be CompressOptions or None, got {type(options).__name__}",
        )
    return _compact_kwargs(
        {
            "preset": _normalize_enum(options.preset, Preset, "preset"),
            "max_dimension": options.max_dimension,
            "quality": options.quality,
            "grayscale": options.grayscale,
            "crop_mode": _normalize_enum(options.crop_mode, CropMode, "crop_mode"),
            "output_format": _normalize_enum(
                options.output_format,
                OutputFormat,
                "output_format",
            ),
            "face_margin": options.face_margin,
        },
    )


def _fit_options_to_kwargs(options: Optional[CompressToFitOptions]) -> Dict[str, object]:
    if options is None:
        return {}
    if not isinstance(options, CompressToFitOptions):
        raise TypeError(
            "options must be CompressToFitOptions or None, "
            f"got {type(options).__name__}",
        )
    return _compact_kwargs(
        {
            "preset": _normalize_enum(options.preset, Preset, "preset"),
            "max_dimension": options.max_dimension,
            "grayscale": options.grayscale,
            "crop_mode": _normalize_enum(options.crop_mode, CropMode, "crop_mode"),
            "output_format": _normalize_enum(
                options.output_format,
                OutputFormat,
                "output_format",
            ),
            "face_margin": options.face_margin,
        },
    )


def compress(
    data: bytes,
    *,
    options: Optional[CompressOptions] = None,
    preset: Optional[_PresetLike] = None,
    max_dimension: Optional[int] = None,
    quality: Optional[float] = None,
    grayscale: Optional[bool] = None,
    crop_mode: Optional[_CropModeLike] = None,
    output_format: Optional[_OutputFormatLike] = None,
    face_margin: Optional[float] = None,
) -> CompressResult:
    """Compress an identity photo.

    Backward compatible with the original keyword-only argument style.
    """

    kwargs = _compress_options_to_kwargs(options)
    kwargs.update(
        _compact_kwargs(
            {
                "preset": _normalize_enum(preset, Preset, "preset"),
                "max_dimension": max_dimension,
                "quality": quality,
                "grayscale": grayscale,
                "crop_mode": _normalize_enum(crop_mode, CropMode, "crop_mode"),
                "output_format": _normalize_enum(
                    output_format,
                    OutputFormat,
                    "output_format",
                ),
                "face_margin": face_margin,
            },
        ),
    )
    return _compress_native(data, **kwargs)


def compress_to_fit(
    data: bytes,
    max_bytes: int,
    *,
    options: Optional[CompressToFitOptions] = None,
    preset: Optional[_PresetLike] = None,
    max_dimension: Optional[int] = None,
    grayscale: Optional[bool] = None,
    crop_mode: Optional[_CropModeLike] = None,
    output_format: Optional[_OutputFormatLike] = None,
    face_margin: Optional[float] = None,
) -> FitResult:
    """Compress an identity photo to fit within a byte budget."""

    kwargs = _fit_options_to_kwargs(options)
    kwargs.update(
        _compact_kwargs(
            {
                "preset": _normalize_enum(preset, Preset, "preset"),
                "max_dimension": max_dimension,
                "grayscale": grayscale,
                "crop_mode": _normalize_enum(crop_mode, CropMode, "crop_mode"),
                "output_format": _normalize_enum(
                    output_format,
                    OutputFormat,
                    "output_format",
                ),
                "face_margin": face_margin,
            },
        ),
    )
    return _compress_to_fit_native(data, max_bytes, **kwargs)


class IdPhoto:
    """Object-style entry point for idphoto operations."""

    @staticmethod
    def compress(
        data: bytes,
        *,
        options: Optional[CompressOptions] = None,
        preset: Optional[_PresetLike] = None,
        max_dimension: Optional[int] = None,
        quality: Optional[float] = None,
        grayscale: Optional[bool] = None,
        crop_mode: Optional[_CropModeLike] = None,
        output_format: Optional[_OutputFormatLike] = None,
        face_margin: Optional[float] = None,
    ) -> CompressResult:
        return compress(
            data,
            options=options,
            preset=preset,
            max_dimension=max_dimension,
            quality=quality,
            grayscale=grayscale,
            crop_mode=crop_mode,
            output_format=output_format,
            face_margin=face_margin,
        )

    @staticmethod
    def compress_to_fit(
        data: bytes,
        max_bytes: int,
        *,
        options: Optional[CompressToFitOptions] = None,
        preset: Optional[_PresetLike] = None,
        max_dimension: Optional[int] = None,
        grayscale: Optional[bool] = None,
        crop_mode: Optional[_CropModeLike] = None,
        output_format: Optional[_OutputFormatLike] = None,
        face_margin: Optional[float] = None,
    ) -> FitResult:
        return compress_to_fit(
            data,
            max_bytes,
            options=options,
            preset=preset,
            max_dimension=max_dimension,
            grayscale=grayscale,
            crop_mode=crop_mode,
            output_format=output_format,
            face_margin=face_margin,
        )


__all__ = [
    "__version__",
    "Preset",
    "CropMode",
    "OutputFormat",
    "CompressOptions",
    "CompressToFitOptions",
    "IdPhoto",
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
