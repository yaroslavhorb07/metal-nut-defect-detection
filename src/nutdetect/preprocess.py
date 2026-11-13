import io

import numpy as np
from PIL import Image, UnidentifiedImageError


class InvalidImage(ValueError):
    pass


def prepare(img, size):
    img = img.convert("RGB").resize((size, size), Image.Resampling.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def from_path(path, size):
    with Image.open(path) as img:
        return prepare(img, size)


def from_bytes(data, size):
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.load()
            return prepare(img, size)
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise InvalidImage("could not decode image") from e
