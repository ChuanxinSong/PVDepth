#!/bin/bash

# ==============================================================================
# General PVDepth inference launcher for images, folders, and videos
# ==============================================================================

# --- 1. Basic configuration (edit these values or use environment variables) ---

# Input path: a single image, an image folder, or a video file
INPUT_PATH=${1:-"examples/100251.mp4"}

# UNet checkpoint path
UNET_PATH=${2:-"Soon122/PVDepth"}

# Output directory
OUTPUT_DIR=${3:-"./outputs"}

# Target width; height is scaled proportionally and aligned to a multiple of 64
RESOLUTION=1024

# GPU configuration
GPU_ID=${GPU_ID:-0}

# Additional options
DECODE_CHUNK_SIZE=8
NUM_WORKERS=8
CPU_OFFLOAD="None" # model | sequential | None

# --- 2. Print configuration ---
echo "========================================"
echo "Starting inference:"
echo "  Input path:  ${INPUT_PATH}"
echo "  UNet path:   ${UNET_PATH}"
echo "  Output dir:  ${OUTPUT_DIR}"
echo "  Resolution:  ${RESOLUTION}"
echo "  GPU ID:      ${GPU_ID}"
echo "========================================"

# --- 3. Build the command ---
CMD="python infer_any_v2.py \
    --input_path ${INPUT_PATH} \
    --output_dir ${OUTPUT_DIR} \
    --unet_path ${UNET_PATH} \
    --resolution ${RESOLUTION} \
    --num_workers ${NUM_WORKERS} \
    --decode_chunk_size ${DECODE_CHUNK_SIZE} \
    --cpu_offload ${CPU_OFFLOAD}"

# --- 4. Run inference ---
CUDA_VISIBLE_DEVICES=${GPU_ID} $CMD
