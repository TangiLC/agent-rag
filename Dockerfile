FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Minimal system deps
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install lightweight dependencies first for better layer caching
COPY requirements-light.txt /app/requirements-light.txt
RUN pip install --upgrade pip \
    && pip install -r /app/requirements-light.txt

# Copy project
COPY . /app

EXPOSE 8501

# Streamlit front-end (light deploy: Bedrock)
CMD ["streamlit", "run", "streamlit/app_light.py", "--server.address=0.0.0.0", "--server.port=8501"]
