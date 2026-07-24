#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
PATHS_ENV="${PROJECT_ROOT}/paths.env"

if [[ ! -f "${PATHS_ENV}" ]]; then
    echo "Error: path configuration file not found: ${PATHS_ENV}" >&2
    exit 1
fi

# shellcheck disable=SC1090
source "${PATHS_ENV}"

: "${data_root:?Error: data_root is not set in paths.env}"

GPU_ID="${GPU_ID:-0}"
PRED_BASE_DIR="${PRED_BASE_DIR:-${PROJECT_ROOT}/carla_benchmark_results/pvdepth}"
OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/results/pvdepth}"
ALIGN_METHOD="${ALIGN_METHOD:-scale&shift}"
MAX_DEPTH="${MAX_DEPTH:-80.0}"
RESOLUTION="${RESOLUTION:-1024}"
PYTHON_BIN="${PYTHON_BIN:-python}"

JSON_BASE_DIR="${data_root}/test_benchmark"
JSON_FILES=(
    setting_dynamic_fps02_len50.json
    setting_dynamic_fps10_len90.json
    setting_dynamic_fps20_len110.json
)

echo "Evaluating PVDepth predictions with ${ALIGN_METHOD} alignment on GPU ${GPU_ID}."

CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" "${SCRIPT_DIR}/eval_alone.py" \
    --json_files "${JSON_FILES[@]}" \
    --json_base_dir "${JSON_BASE_DIR}" \
    --pred_base_dir "${PRED_BASE_DIR}" \
    --gt_root "${data_root}" \
    --output_dir "${OUTPUT_DIR}" \
    --town_name "town0210" \
    --resolution "${RESOLUTION}" \
    --align_method "${ALIGN_METHOD}" \
    --max_depth "${MAX_DEPTH}"

echo "Evaluation complete. Results saved to: ${OUTPUT_DIR}"
