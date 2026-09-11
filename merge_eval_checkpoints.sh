#!/bin/bash

set -e

ROOT=/home/kwangryeol/workspace/Qwen3-8B-Instruct
EASYR1=${ROOT}/easyr1
EXP=${EASYR1}/checkpoints/qwen3_vl_gui_specialization/qwen3_vl_8b_screenspot_v2_grpo

cd "${EASYR1}"
source .venv-easyr1/bin/activate

for STEP in 60 213; do
    ACTOR="${EXP}/global_step_${STEP}/actor"

    echo "======================================"
    echo "Merging global_step_${STEP}"
    echo "${ACTOR}"
    echo "======================================"

    python3 scripts/model_merger.py \
        --local_dir "${ACTOR}"

    echo
    echo "Merged model:"
    echo "${ACTOR}/huggingface"

    ls -lh "${ACTOR}/huggingface" | head -20
done
