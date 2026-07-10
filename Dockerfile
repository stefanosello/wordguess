# A small, reproducible CPU-only image so wordguess runs the same on any PC —
# no local Python version to fight (this pins Python 3.11 inside the container).
FROM python:3.11-slim

WORKDIR /app

# Install the CPU build of torch from PyTorch's own wheel index first. This
# keeps the image ~1 GB instead of ~2.5 GB, since the default PyPI wheel bundles
# CUDA libraries we never use on CPU. Copy only requirements first so this heavy
# layer is cached until the pinned versions actually change.
COPY requirements.txt ./
RUN pip install --no-cache-dir torch==2.2.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r requirements.txt

# Application code.
COPY . .

EXPOSE 8000

# Default command serves the API. Override it to train (see docker-compose.yml).
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
