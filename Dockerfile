FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY pyproject.toml .
COPY src src
RUN pip install --no-cache-dir --no-deps .

COPY models models
ENV MODEL_DIR=/app/models
EXPOSE 8000
CMD ["uvicorn", "nutdetect.api:app", "--host", "0.0.0.0", "--port", "8000"]
