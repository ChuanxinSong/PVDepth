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

accelerate launch \
    --use_deepspeed \
    --deepspeed_config_file "${SCRIPT_DIR}/deepspeed_config.json" \
    --mixed_precision bf16 \
    --num_processes 2 \
    --gpu_ids 0,1 \
    --main_process_port 21756 \
    "${SCRIPT_DIR}/train_stage1_psni.py" \
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
    --anneal_end_step 5000.0 \
    \
    --data_root "${data_root}" \
    --h5_data_root "${h5_data_root}" \
    --output_dir "${output_dir}" \
    --noise_type "distortion_noise_annealed_weighting" \
    --gradient_checkpointing
