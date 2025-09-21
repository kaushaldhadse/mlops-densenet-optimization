# Use NVIDIA CUDA base image compatible with CUDA 13.0 runtime (GTX 1650)
# FROM nvidia/cuda:13.0-devel-ubuntu20.04
# FROM nvidia/cuda:12.6-devel-ubuntu22.04
FROM nvidia/cuda:13.0.1-cudnn-runtime-ubuntu22.04


# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV DEBIAN_FRONTEND=noninteractive
ENV UV_CACHE_DIR=/tmp/.uv-cache

# Install system dependencies including Python 3.12
RUN apt-get update && apt-get install -y \
    software-properties-common \
    curl \
    wget \
    git \
    build-essential \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.12 \
    python3.12-dev \
    python3.12-venv \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && python3.12 -m pip install --upgrade setuptools wheel \
    && rm -rf /var/lib/apt/lists/*



# Create symlinks for python to use Python 3.12
RUN ln -sf /usr/bin/python3.12 /usr/bin/python3 && \
    ln -sf /usr/bin/python3.12 /usr/bin/python

# Install uv (fast Python package manager)
RUN pip3 install uv

# Set working directory
WORKDIR /densenet_optimization

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies using uv
# Note: requirements.txt has duplicate entries with different versions - uv will handle this
RUN uv pip install --system --no-cache-dir -r requirements.txt || \
    (echo "uv failed, falling back to pip" && pip3 install --no-cache-dir -r requirements.txt)

# Copy application code
COPY app/ ./app/
COPY *.py ./
COPY LICENSE ./
COPY README.md ./

# Create necessary directories
RUN mkdir -p data logs results

# Copy entrypoint script
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

# Expose ports for TensorBoard
EXPOSE 6006 6007

# Health check to ensure the application is ready
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import sys; import torch; import torchvision; print(f'Python {sys.version}'); print('Health check passed')" || exit 1

# Set entrypoint
ENTRYPOINT ["./docker-entrypoint.sh"]

# Default command
CMD ["benchmark"]
