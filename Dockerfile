FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    CMAKE_BUILD_PARALLEL_LEVEL=2

WORKDIR /app

# System deps (minimum for llama.cpp build)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    # Drop CUDA-specific packages in Docker CPU build to keep image/build small.
    && grep -Ev '^(nvidia-|triton==)' /app/requirements.txt > /tmp/requirements.docker.txt \
    && pip install -r /tmp/requirements.docker.txt \
    && pip install streamlit

# Copy project
COPY . /app

# Build llama.cpp server binary (CPU build)
RUN cmake -S /app/llama.cpp -B /app/llama.cpp/build \
    && cmake --build /app/llama.cpp/build

EXPOSE 8501

# Streamlit front-end
CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
