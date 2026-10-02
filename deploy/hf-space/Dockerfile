# Hugging Face Space (Docker SDK) serving the large server-only age model.
# Staged and pushed by scripts/deploy_hf_space.sh; not built from this dir.
FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends curl libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY requirements.txt .
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . .
USER user

ENV AGE_MODEL_PATH_REAL=/home/user/app/checkpoints/age_model_server.pt \
    SERVER_MODELS_DIR=/home/user/app/server/models \
    CORS_ORIGINS=https://horgsz.github.io \
    OMP_NUM_THREADS=2
EXPOSE 7860
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]
