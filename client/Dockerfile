FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# CPU-only torch — `--index-url` (not `--extra-index-url`) so CUDA wheels
# from PyPI are NOT preferred. Drops the layer from ~750MB to ~200MB.
RUN pip install --index-url https://download.pytorch.org/whl/cpu \
    "torch>=2.4" \
    "torchvision>=0.19"

# Flower runtime + fl_app's transitive deps.
# Excluded vs server: `datasets` (HF, ~250MB transitive — clients use ImageFolder
# only) and `scikit-learn` (server-only `evaluate()`, lazy-imported there).
RUN pip install \
    "flwr>=1.28.0,<1.29" \
    "timm>=0.9" \
    "Pillow>=10" \
    "numpy>=1.26"

WORKDIR /app
COPY fl_client ./fl_client

ENTRYPOINT ["python", "-m", "fl_client"]
