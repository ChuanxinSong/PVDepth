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
: "${h5_data_root:?Error: h5_data_root is not set in paths.env}"
: "${output_dir:?Error: output_dir is not set in paths.env}"

NOISE_TYPE="distortion_noise_w_weighting_normal_0_1" 

UNET_PATH="/path/to/stage1_result/unet" # baseline + PSNI


accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/deepspeed_config.json" \
    --mixed_precision bf16 \
    --num_processes 2 \
    --gpu_ids 1,0 \
    --main_process_port 21856 \
"${SCRIPT_DIR}/train_stage2_pvdepth.py" \
    \
    --num_frames 6 \
    --gradient_accumulation_steps 1 \
    --num_workers 8 \
    \
    --per_gpu_batch_size 4 \
    --learning_rate 5e-6 \
    --max_train_steps 20000 \
    --mixed_precision "bf16" \
    --width 640 \
    --height 320 \
    \
    --data_root "${data_root}" \
    --h5_data_root "${h5_data_root}" \
    --unet_path "${UNET_PATH}" \
    --noise_type "${NOISE_TYPE}" \
    --output_dir "${output_dir}" \
    --gradient_checkpointing
