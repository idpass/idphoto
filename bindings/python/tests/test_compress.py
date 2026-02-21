"""Smoke tests for the idphoto Python bindings."""

import struct
import zlib

import pytest

import idphoto


def make_test_png(width: int, height: int) -> bytes:
    """Create a minimal valid PNG image from scratch."""
    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(chunk) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + chunk + crc

    # IHDR: width, height, bit_depth=8, color_type=2 (RGB), compression=0, filter=0, interlace=0
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT: scanlines with filter byte 0 (None) per row
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"  # filter byte
        for x in range(width):
            r = (x * 255 // max(width - 1, 1)) & 0xFF
            g = (y * 255 // max(height - 1, 1)) & 0xFF
            b = 128
            raw_data += bytes([r, g, b])
    compressed = zlib.compress(raw_data)
    idat = _chunk(b"IDAT", compressed)

    # IEND
    iend = _chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


class TestCompress:
    def test_basic_compress(self):
        png = make_test_png(200, 300)
        result = idphoto.compress(png)
        assert len(result.data) > 0
        assert result.width <= 48
        assert result.height <= 64
        assert result.original_size == len(png)

    def test_compress_with_preset(self):
        png = make_test_png(200, 300)
        result = idphoto.compress(png, preset="qr-code")
        assert len(result.data) > 0
        assert result.format == "webp"
        # WebP RIFF header
        assert result.data[:4] == b"RIFF"

    def test_compress_with_jpeg_format(self):
        png = make_test_png(200, 300)
        result = idphoto.compress(png, output_format="jpeg", quality=0.8)
        assert result.data[:2] == b"\xff\xd8"
        assert result.format == "jpeg"

    def test_compress_with_face_margin(self):
        png = make_test_png(200, 300)
        result = idphoto.compress(
            png, crop_mode="face-detection", face_margin=1.3
        )
        assert len(result.data) > 0

    def test_compress_face_bounds_attribute_present(self):
        png = make_test_png(200, 300)
        result = idphoto.compress(png, crop_mode="heuristic")
        assert hasattr(result, "face_bounds")
        # Heuristic mode -> no face bounds
        assert result.face_bounds is None

    def test_compress_to_fit_basic(self):
        png = make_test_png(200, 300)
        result = idphoto.compress_to_fit(png, 10_000)
        assert result.reached_target is True
        assert len(result.data) <= 10_000
        assert 0.0 < result.quality_used <= 1.0

    def test_compress_to_fit_with_preset(self):
        png = make_test_png(200, 300)
        result = idphoto.compress_to_fit(png, 10_000, preset="qr-code")
        assert result.reached_target is True

    def test_compress_to_fit_face_bounds_attribute(self):
        png = make_test_png(200, 300)
        result = idphoto.compress_to_fit(png, 10_000, crop_mode="heuristic")
        assert hasattr(result, "face_bounds")

    def test_all_presets(self):
        png = make_test_png(200, 300)
        for preset in ["qr-code", "qr-code-match", "print", "display"]:
            result = idphoto.compress(png, preset=preset)
            assert len(result.data) > 0, f"preset {preset} produced empty output"

    def test_enums_and_typed_options(self):
        png = make_test_png(200, 300)
        options = idphoto.CompressOptions(
            preset=idphoto.Preset.QR_CODE,
            crop_mode=idphoto.CropMode.HEURISTIC,
            output_format=idphoto.OutputFormat.WEBP,
            max_dimension=48,
        )
        result = idphoto.compress(png, options=options)
        assert len(result.data) > 0
        assert result.format == "webp"

    def test_object_style_api(self):
        png = make_test_png(200, 300)
        result = idphoto.IdPhoto.compress(
            png,
            preset=idphoto.Preset.DISPLAY,
            output_format=idphoto.OutputFormat.JPEG,
        )
        assert result.data[:2] == b"\xff\xd8"

    def test_fit_typed_options(self):
        png = make_test_png(200, 300)
        options = idphoto.CompressToFitOptions(
            preset=idphoto.Preset.QR_CODE_MATCH,
            crop_mode=idphoto.CropMode.HEURISTIC,
            output_format=idphoto.OutputFormat.WEBP,
        )
        result = idphoto.compress_to_fit(png, 10_000, options=options)
        assert result.reached_target is True
        assert len(result.data) <= 10_000

    def test_compress_result_repr(self):
        png = make_test_png(200, 300)
        result = idphoto.compress(png)
        repr_str = repr(result)
        assert repr_str.startswith("CompressResult(")
        assert "format=" in repr_str
        assert "width=" in repr_str
        assert "height=" in repr_str
        assert "size=" in repr_str
        assert "original_size=" in repr_str


class TestExceptionHierarchy:
    """Test that library errors raise specific IdPhotoError subclasses."""

    def test_decode_error_is_idphoto_error(self):
        """DecodeError should be a subclass of IdPhotoError."""
        with pytest.raises(idphoto.IdPhotoError):
            idphoto.compress(b"not an image")

    def test_decode_error_specific(self):
        """Invalid input should raise DecodeError specifically."""
        with pytest.raises(idphoto.DecodeError):
            idphoto.compress(b"not an image")

    def test_invalid_quality_error(self):
        """Quality outside 0.0-1.0 should raise InvalidQualityError."""
        png = make_test_png(100, 100)
        with pytest.raises(idphoto.InvalidQualityError):
            idphoto.compress(png, quality=1.5)

    def test_invalid_max_dimension_error(self):
        """max_dimension=0 should raise InvalidMaxDimensionError."""
        png = make_test_png(100, 100)
        with pytest.raises(idphoto.InvalidMaxDimensionError):
            idphoto.compress(png, max_dimension=0)

    def test_unknown_preset_raises_value_error(self):
        """Argument validation errors should remain ValueError."""
        png = make_test_png(100, 100)
        with pytest.raises(ValueError):
            idphoto.compress(png, preset="invalid")

    def test_unknown_crop_mode_raises_value_error(self):
        """Argument validation errors should remain ValueError."""
        png = make_test_png(100, 100)
        with pytest.raises(ValueError):
            idphoto.compress(png, crop_mode="invalid")

    def test_unknown_format_raises_value_error(self):
        """Argument validation errors should remain ValueError."""
        png = make_test_png(100, 100)
        with pytest.raises(ValueError):
            idphoto.compress(png, output_format="invalid")


class TestVersionAttribute:
    def test_version_exists(self):
        """idphoto.__version__ should be a dotted string."""
        assert hasattr(idphoto, "__version__")
        version = idphoto.__version__
        assert isinstance(version, str)
        parts = version.split(".")
        assert len(parts) >= 2
        for part in parts:
            assert part.isdigit()
