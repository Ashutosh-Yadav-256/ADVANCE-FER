# Multi-stage production Dockerfile for ADVANCE-FER Microservice
FROM python:3.12-slim AS base

# Set production environment flags
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive

WORKDIR /app

# Install system dependencies required for OpenCV and MediaPipe headless execution
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root system user for security
RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/models/checkpoints && \
    chown -R appuser:appuser /app

# Copy application source code
COPY --chown=appuser:appuser . .

# Switch to non-root user
USER appuser

# Pre-download and cache model weights into image layer
RUN python -m models.weights_manager

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/healthz || exit 1

# Default entrypoint starts production FastAPI server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
