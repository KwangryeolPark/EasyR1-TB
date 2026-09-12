#!/usr/bin/env bash
set -euo pipefail

ROOT=${QWEN3_PROJECT_ROOT:-/home/kwangryeol/workspace/Qwen3-8B-Instruct}
EASYR1=${ROOT}/easyr1
MODEL_PATH=${ROOT}/Qwen3-VL-8B-Instruct
TRAIN_FILE=${ROOT}/datasets/groundcua/easyr1/train.parquet
VAL_FILE=${ROOT}/datasets/groundcua/easyr1/validation.parquet

cd "${EASYR1}"
source .venv-easyr1/bin/activate
export TOKENIZERS_PARALLELISM=false

# The official curated GroundCUA split has 623,498 train rows and 8,000 fixed
# validation rows. At rollout_batch_size=16 this is 38,968 steps per epoch;
# a 10,000-step interval observes validation about four times per epoch while
# keeping checkpoint overhead manageable.
python3 -m verl.trainer.main \
    config=examples/config.yaml \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=problem \
    data.answer_key=answer \
    data.image_key=images \
    data.max_prompt_length=8192 \
    data.max_response_length=128 \
    data.rollout_batch_size=16 \
    data.val_batch_size=64 \
    data.format_prompt=examples/format_prompt/groundcua.jinja \
    data.shuffle=true \
    data.seed=42 \
    data.min_pixels=262144 \
    data.max_pixels=2359296 \
    data.filter_overlong_prompts=false \
    algorithm.adv_estimator=grpo \
    algorithm.disable_kl=false \
    algorithm.use_kl_loss=true \
    algorithm.kl_penalty=low_var_kl \
    algorithm.kl_coef=1.0e-2 \
    worker.actor.global_batch_size=16 \
    worker.actor.micro_batch_size_per_device_for_update=1 \
    worker.actor.micro_batch_size_per_device_for_experience=1 \
    worker.actor.max_grad_norm=1.0 \
    worker.actor.model.model_path="${MODEL_PATH}" \
    worker.actor.model.enable_gradient_checkpointing=true \
    worker.actor.model.freeze_vision_tower=false \
    worker.actor.model.lora.rank=0 \
    worker.actor.optim.lr=1.0e-6 \
    worker.actor.optim.weight_decay=1.0e-2 \
    worker.actor.optim.lr_warmup_ratio=0.05 \
    worker.actor.optim.lr_scheduler_type=constant \
    worker.actor.fsdp.enable_full_shard=true \
    worker.actor.fsdp.enable_cpu_offload=false \
    worker.actor.offload.offload_params=true \
    worker.actor.offload.offload_optimizer=true \
    worker.rollout.n=8 \
    worker.rollout.temperature=1.0 \
    worker.rollout.top_p=1.0 \
    worker.rollout.limit_images=1 \
    worker.rollout.gpu_memory_utilization=0.50 \
    worker.rollout.tensor_parallel_size=1 \
    worker.rollout.max_num_batched_tokens=16384 \
    worker.reward.reward_function=examples/reward_function/groundcua.py:compute_score \
    trainer.total_epochs=3 \
    trainer.project_name=qwen3_vl_gui_specialization \
    trainer.experiment_name=qwen3_vl_8b_groundcua_grpo \
    trainer.logger='["file"]' \
    trainer.n_gpus_per_node=4 \
    trainer.nnodes=1 \
    trainer.val_before_train=true \
    trainer.best_metric=accuracy \
    trainer.epoch_end_validation=true \
    trainer.epoch_end_save=true \
    trainer.val_freq=10000 \
    trainer.val_generations_to_log=10 \
    trainer.save_freq=10000 \
    trainer.save_limit=10
