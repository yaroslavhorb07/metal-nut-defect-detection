import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse

from .preprocess import InvalidImage

MAX_BYTES = 10 * 1024 * 1024
INDEX = Path(__file__).parent / "static" / "index.html"

log = logging.getLogger("api")


def create_app(predictor=None):
    @asynccontextmanager
    async def lifespan(app):
        if predictor is None:
            from .predict import Predictor
            app.state.predictor = Predictor(os.environ.get("MODEL_DIR", "models"))
        else:
            app.state.predictor = predictor
        yield

    app = FastAPI(title="Metal nut defect detector", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index():
        return INDEX.read_text()

    @app.get("/health")
    def health():
        return {"status": "ok", **app.state.predictor.info()}

    @app.post("/predict")
    def predict(file: UploadFile = File(...)):
        data = file.file.read(MAX_BYTES + 1)
        if len(data) > MAX_BYTES:
            raise HTTPException(413, "image too large (max 10MB)")
        try:
            result = app.state.predictor.predict(data)
        except InvalidImage:
            raise HTTPException(400, "file is not a readable image")
        except Exception:
            log.exception("prediction failed")
            raise HTTPException(500, "prediction failed")
        return {"filename": file.filename, **result}

    return app


app = create_app()
