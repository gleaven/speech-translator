ARG CUDA_VERSION=12.8.0
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04

ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

WORKDIR /usr/src/app

# Install system packages
RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    espeak-ng \
    && rm -rf /var/lib/apt/lists/*

# Make python3 the default and upgrade pip
RUN ln -sf /usr/bin/python3 /usr/bin/python && \
    pip install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch nightly with CUDA 12.8 (Blackwell/sm_120+ support)
RUN pip install --no-cache-dir --pre torch torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Broadcast mode deps (NLLB tokenizer, API server, TTS phonemizers)
RUN pip install --no-cache-dir sentencepiece "fastapi>=0.100" "uvicorn[standard]" ordered_set pyopenjtalk mojimoji

COPY . .

# Download NLTK data
RUN python -c "import nltk; nltk.download('punkt_tab'); nltk.download('averaged_perceptron_tagger_eng')"

# Download MeloTTS unidic data
RUN python -m unidic download || true
