FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System deps for building llama.cpp and Python packages
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential \
       cmake \
       git \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps first for better layer caching
COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements.txt \
    && pip install streamlit

# Copy project
COPY . /app

# Build llama.cpp server binary (CPU build)
RUN cmake -S /app/llama.cpp -B /app/llama.cpp/build \
    && cmake --build /app/llama.cpp/build -j

EXPOSE 8501

# Streamlit front-end
CMD ["streamlit", "run", "streamlit/app.py", "--server.address=0.0.0.0", "--server.port=8501"]
