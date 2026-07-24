#!/bin/bash

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PATHS_ENV="${SCRIPT_DIR}/paths.env"

if [[ ! -f "${PATHS_ENV}" ]]; then
    echo "Error: path configuration file not found: ${PATHS_ENV}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${PATHS_ENV}"

: "${data_root:?Error: data_root is not set in paths.env}"

DECODE_CHUNK_SIZE=8

# ppl_type="depthcrafter" # baseline
# ppl_type="distortion_noise_annealed_weighting" # baseline + PSNI
ppl_type="pvdepth"


OUTPUT_ROOT_DIR="carla_benchmark_results/pvdepth" # PVDepth

# UNET_PATH="/path/to/result/unet"
UNET_PATH="Soon122/PVDepth"


NUM_WORKERS=8
IMAGE_BASE_DIR="${data_root}"
RESOLUTION=1024
cpu_offload="sequential" #model | None | sequential
GPU_ID="${GPU_ID:-0}"


CUDA_VISIBLE_DEVICES="${GPU_ID}" python run_infer_town0210.py \
    --output_root_dir "${OUTPUT_ROOT_DIR}" \
    --resolution "${RESOLUTION}" \
    --unet_path "${UNET_PATH}" \
    --num_workers "${NUM_WORKERS}" \
    --image_base_dir "${IMAGE_BASE_DIR}" \
    --decode_chunk_size "${DECODE_CHUNK_SIZE}" \
    --cpu_offload "${cpu_offload}" \
    --ppl_type "${ppl_type}"
