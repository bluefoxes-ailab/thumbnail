FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    unzip \
    && rm -rf /var/lib/apt/lists/*

# A JavaScript runtime for yt-dlp. YouTube signs its video URLs with
# JavaScript from the watch page, and without something to run it every
# download is refused (see the runtime notes in video_downloader.py, which
# refuses to start a YouTube download without one rather than fail with a
# bare 403). deno is what yt-dlp looks for first, on PATH.
RUN curl -fsSL -o /tmp/deno.zip \
    https://github.com/denoland/deno/releases/latest/download/deno-x86_64-unknown-linux-gnu.zip \
    && unzip -o /tmp/deno.zip -d /usr/local/bin \
    && rm /tmp/deno.zip \
    && deno --version

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip

# Core PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.6.0+cu124 \
    torchvision==0.21.0+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Base requirements. yt-dlp is a floor and not a pin, for the reason given
# beside it in requirements.txt: 2026.7.4, which was pinned here, had already
# stopped working against YouTube, and the version that works moves every few
# weeks. Rebuild the image to pick up a newer one.
RUN pip install --no-cache-dir \
    fastapi==0.139.0 \
    uvicorn==0.50.2 \
    pydantic==2.11.7 \
    imageio-ffmpeg==0.6.0 \
    "yt-dlp>=2026.8.19" \
    onnxruntime-gpu==1.20.0 \
    faster-whisper==1.1.1 \
    ctranslate2==4.5.0 \
    opencv-python==4.13.0.92

# GAN support
RUN pip install --no-cache-dir addict future lmdb yapf filterpy
RUN pip install --no-cache-dir --no-deps \
    basicsr==1.4.2 \
    facexlib==0.3.0 \
    gfpgan==1.3.8 \
    simple-lama-inpainting==0.1.2

# Patch torchvision issue in basicsr
RUN find /usr/local/lib -name "degradations.py" -exec sed -i "s/functional_tensor/functional/g" {} +

# Pre-fetch models
RUN mkdir -p /app/backend/models && \
    wget -q -O /app/backend/models/u2net.onnx \
    https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net.onnx && \
    wget -q -O /app/backend/models/text_detection_en_ppocrv3.onnx \
    https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/text_detection_ppocr/text_detection_en_ppocrv3_2023may.onnx

RUN pip install --no-cache-dir gdown
COPY . /app

EXPOSE 8000

WORKDIR /app
CMD ["python3", "api.py"]