# Nut defect detector

Author: Yaroslav Horb

Classifies photos of metal nuts (MVTec AD "metal nut") as good or defective.
MobileNetV2 transfer learning, served with FastAPI.

## About this project

This is my Extended Project Qualification (EPQ), an AQA Level 3 qualification.

## Dataset

The images come from the "metal nut" category of the MVTec Anomaly Detection (MVTec AD)
dataset: https://www.mvtec.com/company/research/datasets/mvtec-ad

MVTec AD is released under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
license (CC BY-NC-SA 4.0), so it can't be used for commercial purposes.

> Paul Bergmann, Michael Fauser, David Sattlegger, Carsten Steger.
> "MVTec AD — A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection."
> IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2019.

## Instructions to build

Requires Python 3.10 or newer.

1. Create a virtual environment and install the dependencies:

   ```
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements-dev.txt
   pip install -e . --no-deps
   ```

2. Only needed for training: download MVTec AD (see "Dataset" above) and put the metal nut
   images in `dataset/good` and `dataset/bad`. The dataset isn't included in this repository.

3. Train the model (settings are in `configs/default.yaml`):

   ```
   python -m nutdetect.train
   ```

   This writes `models/model.keras` and `models/meta.json`, plus `reports/metrics.json`,
   `reports/splits.csv`, a confusion matrix and a ROC curve. A trained model is already
   included in `models/`, so this step is optional.

4. Optional: build the Docker image. The model must be in `models/` first, since the image copies it.

   ```
   docker build -t nut-detector .
   ```

## Instructions to run

The easiest way is to run `run.py` from the project folder, with your editor's Run button or with:

```
python run.py
```

It starts the server and opens http://127.0.0.1:8000 in your browser once the model has loaded.
It switches to `.venv` on its own, so any Python 3 can start it. Press Ctrl+C or stop the program
to shut the server down. If the server is already running, it just opens the page.

To start the server by hand instead, with the virtual environment active:

```
uvicorn nutdetect.api:app
```

Open http://127.0.0.1:8000 in a browser to upload an image, or send one from the command line:

```
curl -F file=@nut.png http://127.0.0.1:8000/predict
```

The response includes the label (`good` or `defective`), the defect probability, the threshold,
and an `uncertain` flag for predictions close to the threshold. `GET /health` reports the
model's threshold and test metrics.

To load a model from a folder other than `models/`, set `MODEL_DIR`:

```
MODEL_DIR=/path/to/model uvicorn nutdetect.api:app
```

With Docker:

```
docker run -p 8000:8000 nut-detector
```

## Instructions to run the test suite

With the virtual environment active, from the project folder:

```
pytest
```

The tests cover data splitting, preprocessing, evaluation metrics, the uncertainty check and the
API. They use a fake predictor, so they don't need a trained model.
