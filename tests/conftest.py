import io

import pytest
from PIL import Image


@pytest.fixture
def png_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (120, 90, 30)).save(buf, format="PNG")
    return buf.getvalue()
