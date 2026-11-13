import pytest
from fastapi.testclient import TestClient

from nutdetect.api import MAX_BYTES, create_app
from nutdetect.preprocess import InvalidImage, from_bytes


class FakePredictor:
    def predict(self, data):
        from_bytes(data, 16)
        return {"label": "good", "prob_defect": 0.1, "threshold": 0.5, "uncertain": False}

    def info(self):
        return {"threshold": 0.5}


@pytest.fixture
def client():
    with TestClient(create_app(FakePredictor())) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_index_page(client):
    assert "Nut defect detector" in client.get("/").text


def test_predict_ok(client, png_bytes):
    r = client.post("/predict", files={"file": ("nut.png", png_bytes, "image/png")})
    assert r.status_code == 200
    assert r.json()["label"] == "good"
    assert r.json()["filename"] == "nut.png"


def test_predict_rejects_non_image(client):
    r = client.post("/predict", files={"file": ("a.png", b"hello", "image/png")})
    assert r.status_code == 400


def test_predict_rejects_huge_upload(client):
    big = b"0" * (MAX_BYTES + 10)
    r = client.post("/predict", files={"file": ("a.png", big, "image/png")})
    assert r.status_code == 413


def test_predict_requires_file(client):
    assert client.post("/predict").status_code == 422
