#!/bin/bash
# Build and start the DGX Spark Docker container for Mask2Former pipeline
set -e

IMAGE_NAME="mask2former_seg"
CONTAINER_NAME="mask2former_seg"

# Remove any existing container with same name
docker rm -f ${CONTAINER_NAME} 2>/dev/null && echo "Removed existing container" || true

echo "=== Building Mask2Former Docker Image (DGX Spark) ==="
docker build -t ${IMAGE_NAME}:latest .

echo ""
echo "=== Starting Container ==="
mkdir -p hf_cache output/mask2former data/hospital_coco

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
echo "  # Download dataset"
echo "  docker exec ${CONTAINER_NAME} python -m scripts.mask2former_seg.download_dataset"
echo ""
echo "  # Train"
echo "  docker exec ${CONTAINER_NAME} python -m scripts.mask2former_seg.train"
echo ""
echo "  # Evaluate (after training)"
echo "  docker exec ${CONTAINER_NAME} python -m scripts.mask2former_seg.evaluate"
