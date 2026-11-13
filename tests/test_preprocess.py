import pytest

from nutdetect.preprocess import InvalidImage, from_bytes


def test_output_shape_and_dtype(png_bytes):
    arr = from_bytes(png_bytes, 32)
    assert arr.shape == (32, 32, 3)
    assert arr.dtype.name == "uint8"


def test_garbage_raises():
    with pytest.raises(InvalidImage):
        from_bytes(b"definitely not an image", 32)
