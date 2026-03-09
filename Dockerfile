# Segmentation Pipeline — DGX Spark Container
# ARM64 (aarch64) — uses nvcr.io PyTorch base image for NVIDIA DGX Spark
FROM nvcr.io/nvidia/pytorch:25.11-py3

LABEL maintainer="segmentation-pipeline"
LABEL description="Mask2Former + SAM2-UNet hospital segmentation pipeline (DGX Spark)"

# System dependencies (Ubuntu 24.04 inside nvcr pytorch image)
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Install Python dependencies
# Note: PyTorch and torchvision are already provided by the base image
RUN pip install --upgrade pip && \
    pip install --no-cache-dir \
        transformers>=4.40.0 \
        peft>=0.9.0 \
        accelerate \
        albumentations \
        torchmetrics \
        scipy \
        timm \
        scikit-learn \
        opencv-python-headless \
        tqdm \
        numpy \
        matplotlib \
        pillow \
        pycocotools \
        sentencepiece \
        protobuf \
        roboflow \
        discord

# Install SAM2 from source
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam2.git

# 安裝 SAM3
RUN pip install --no-cache-dir git+https://github.com/facebookresearch/sam3.git || \

# Copy project scripts only (data/output are mounted as volumes)
COPY scripts/ /workspace/scripts/

# Make scripts importable as a package
RUN touch /workspace/scripts/__init__.py 2>/dev/null || true

# Environment: persist HuggingFace model cache on the host
ENV HF_HOME=/workspace/.cache/huggingface
ENV TRANSFORMERS_CACHE=/workspace/.cache/huggingface/transformers

# Default: show help
CMD ["echo", "Use: python -m scripts.mask2former_seg.train OR python -m scripts.sam2_seg.train"]

