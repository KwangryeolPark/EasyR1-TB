#!/usr/bin/env bash

set -euo pipefail
set -x

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

EXPERIMENT_NAME=${EXPERIMENT_NAME:-qwen25vl3b_gta1_natural_grpo}

TRAIN_FILE=${TRAIN_FILE:-../datasets/pilot/gta1_grpo_train.parquet}
VAL_FILE=${VAL_FILE:-../datasets/pilot/gta1_grpo_val.parquet}

python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="$TRAIN_FILE" \
    data.val_files="$VAL_FILE" \
    data.prompt_key=prompt \
    data.answer_key=answer \
    data.image_key=images \
    data.max_prompt_length=4096 \
    data.max_response_length=32 \
    data.rollout_batch_size=8 \
    data.val_batch_size=16 \
    data.format_prompt=examples/format_prompt/gta1_grounding.jinja \
    data.min_pixels=$((256 * 28 * 28)) \
    data.max_pixels=$((1280 * 28 * 28)) \
    data.seed=42 \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo \
    algorithm.kl_coef=1.0e-2 \
    worker.actor.global_batch_size=8 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.actor.model.model_path=Qwen/Qwen2.5-VL-3B-Instruct \
    worker.actor.model.trust_remote_code=true \
    worker.actor.optim.lr=1.0e-6 \
    worker.rollout.n=8 \
    worker.rollout.temperature=0.8 \
    worker.rollout.top_p=0.95 \
    worker.rollout.limit_images=1 \
    worker.rollout.tensor_parallel_size=1 \
    worker.reward.reward_function=examples/reward_function/gta1_grounding.py:compute_score \
    trainer.project_name=teacher_balanced_gui \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.logger='["console","file"]' \
    trainer.total_epochs=3 \
    trainer.max_steps=1000 \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_freq=50 \
    trainer.save_freq=50 \
    trainer.save_limit=-1
