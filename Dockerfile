FROM python:3.12-slim

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies, font/graphics libraries for Docling & Typst, and Caddy
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgomp1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ca-certificates \
    debian-keyring \
    debian-archive-keyring \
    apt-transport-https \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/gpg.key' | gpg --dearmor -o /usr/share/keyrings/caddy-stable-archive-keyring.gpg \
    && curl -1sLf 'https://dl.cloudsmith.io/public/caddy/stable/debian.deb.txt' | tee /etc/apt/sources.list.d/caddy-stable.list \
    && apt-get update \
    && apt-get install -y caddy \
    && rm -rf /var/lib/apt/lists/*

# Configure non-root user (UID 1000) for security & Hugging Face compatibility
RUN useradd -m -u 1000 user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PORT=7860 \
    INTERNAL_PORT=5000 \
    GROWNXT_OUTPUT_DIR=/app/output \
    HF_HOME=/home/user/.cache/huggingface \
    TORCH_HOME=/home/user/.cache/torch \
    DOCLING_ARTIFACTS_PATH=/home/user/.cache/docling

WORKDIR /app

# Install CPU-only PyTorch to minimize container size & avoid CUDA overhead
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download SentenceTransformer embeddings into image layer
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('Snowflake/snowflake-arctic-embed-m-v1.5')"

# Copy application source code
COPY . .

# Set permissions for user 1000
RUN mkdir -p /app/output /home/user/.cache && \
    chmod +x /app/entrypoint.sh && \
    chown -R user:user /app /home/user

# Switch to non-root user
USER user

EXPOSE 7860 5000

ENTRYPOINT ["/app/entrypoint.sh"]
