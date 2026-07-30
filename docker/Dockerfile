# syntax=docker/dockerfile:1

FROM python:3.12-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr,
# which matters for timely structured log delivery.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv/app

# Install system build dependencies required by faiss-cpu and other
# compiled wheels, then remove build-only tooling from the final layer
# where possible by relying on prebuilt wheels.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY dashboard ./dashboard
COPY data ./data
COPY pyproject.toml ./

# Create data directory for SQLite database and policy documents
RUN mkdir -p /srv/app/data/policies /srv/app/data/vector_store

# Run as a non-root user.
RUN useradd --create-home --uid 1000 tracelens \
    && chown -R tracelens:tracelens /srv/app
USER tracelens

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/v1/health').read()" || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
