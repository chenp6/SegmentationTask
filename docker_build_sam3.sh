#!/bin/bash
# Build and start the Docker container for SAM3 segmentation pipeline
set -e

IMAGE_NAME="sam3_seg"
CONTAINER_NAME="sam3_seg"

# Remove any existing container with same name
docker rm -f ${CONTAINER_NAME} 2>/dev/null && echo "Removed existing container" || true

echo "=== Building SAM3 Docker Image ==="
docker build -t ${IMAGE_NAME}:latest .

echo ""
echo "=== Starting Container ==="
mkdir -p hf_cache output/sam3 data/hospital_coco

docker run -d \
    --name ${CONTAINER_NAME} \
    --gpus all \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e HF_HOME=/workspace/.cache/huggingface \
    -v "$(pwd)/scripts:/workspace/scripts" \
    -v "$(pwd)/data:/workspace/data" \
    -v "$(pwd)/output:/workspace/output" \
    -v "$(pwd)/hf_cache:/workspace/.cache/huggingface" \
    -v "$(pwd)/roboflow_credentials.json:/workspace/roboflow_credentials.json:ro" \
    -w /workspace \
    ${IMAGE_NAME}:latest \
    tail -f /dev/null

echo ""
echo "=== Container '${CONTAINER_NAME}' is running! ==="
echo ""
echo "Quick commands:"
echo "  # Train SAM3"
echo "  docker exec ${CONTAINER_NAME} python -m scripts.sam3_seg.train"
echo ""
echo "  # Evaluate"
echo "  docker exec ${CONTAINER_NAME} python -m scripts.sam3_seg.evaluate --checkpoint output/sam3/best_model.pth"
