FROM python:3.12-slim

WORKDIR /app

# libGL/glib are needed by opencv, pulled in transitively by grad-cam.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install the CPU torch wheels first so the large layer caches independently of the
# app code, and so no CUDA payload is pulled into the image.
RUN pip install --no-cache-dir \
    torch==2.9.0 torchvision==0.24.0 \
    --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code and the trained model. model.py and gradcam.py sit one level up in
# the repo, so the deploy script stages them next to main.py before building.
COPY main.py model.py gradcam.py ./
COPY model.pt ./model.pt

ENV MODEL_PATH=/app/model.pt
EXPOSE 8080

# One worker keeps memory bounded (the lesson from the earlier OOM crash loop); the
# uvicorn worker gives FastAPI its async loop. 300s matches the Cloud Run timeout.
CMD exec gunicorn main:app \
    --bind :$PORT \
    --workers 1 \
    --threads 2 \
    --worker-class uvicorn.workers.UvicornWorker \
    --timeout 300 \
    --keep-alive 5
