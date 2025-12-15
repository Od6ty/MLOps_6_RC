# docker/Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system deps (opsional, misal untuk OpenCV/Pillow)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements dan install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src ./src
COPY models ./models

# Expose port FastAPI
EXPOSE 8000

# Jalankan FastAPI dengan Uvicorn
# Asumsi app FastAPI ada di src/serving/api.py -> app = FastAPI()
CMD ["uvicorn", "src.serving.api:app", "--host", "0.0.0.0", "--port", "8000"]