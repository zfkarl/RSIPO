#!/usr/bin/env bash
# Starts SEED's trainer (examples/seed_trainer) with RSIPO enabled. Called by run_rsipo.sh, which sets the
# per-benchmark settings; extra arguments are passed on as Hydra overrides.
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
: "${RSIPO_RUN_ROOT:?}" "${RSIPO_PHASE:?}" "${RSIPO_STEPS:?}" "${RSIPO_SAVE_FREQ:?}" "${RSIPO_MODEL_NAME:?}"
: "${RSIPO_SKILL_COEF:?}"
phase_dir="$RSIPO_RUN_ROOT/$RSIPO_PHASE"
# SEED's launchers activate CONDA_ENV unless it is the active environment; keep the current one.
export CONDA_ENV="${CONDA_DEFAULT_ENV:-}"
export HF_MODEL_PATH="$MODELS_ROOT/$RSIPO_MODEL_NAME"
export MODEL_PATH="$HF_MODEL_PATH"
export EXPERIMENT_NAME="rsipo_$(basename "$RSIPO_RUN_ROOT")_${RSIPO_PHASE}"
export DEFAULT_LOCAL_DIR="$phase_dir/checkpoints"
export DATA_ROOT="$RSIPO_WORK/data/rl-inputs/$EXPERIMENT_NAME" ENV_FILE=/dev/null
export SEED_SKILL_MODE=episode_only SEED_MODE="${RSIPO_SEED_MODE:-mean_std_norm}" SEED_OPD_LOSS_COEF=0.01 SEED_ANALYSIS_BACKEND=policy_vllm
export SEED_ANALYSIS_MAX_COMPLETION_TOKENS="${RSIPO_ANALYSIS_MAX_TOKENS:-4096}"
export TRAIN_DATA_SIZE="${RSIPO_TRAIN_TASKS:-16}" VAL_DATA_SIZE="${RSIPO_VAL_DATA_SIZE:-96}" GROUP_SIZE=8
export PPO_MICRO_BATCH_SIZE_PER_GPU=1 TENSOR_MODEL_PARALLEL_SIZE="${RSIPO_ROLLOUT_TP:-1}"
export VERL_VLLM_SLEEP_PIN_MEMORY=0 OPENAI_API_KEY=EMPTY
export RSI_OPD_METRICS_JSONL="${RSIPO_METRICS_JSONL:-$phase_dir/scalar-metrics.jsonl}"
export RSI_FSDP_NO_SYNC=1 CUDA_DEVICE_MAX_CONNECTIONS=8 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export RSI_ACTOR_PROGRESS_DIR="$phase_dir/actor-progress"
unset RSI_ACTOR_REPLAY_DIR
# Optional: node IPs that the job may use in a shared Ray cluster (rsipo/placement.py); unset means anywhere.
export RSIPO_RAY_NODES="${RSIPO_RAY_NODES:-}"

# Qwen3 runs in non-thinking mode.
thinking=()
if [[ "$RSIPO_MODEL_NAME" == *Qwen3* ]]; then thinking=("+data.apply_chat_template_kwargs.enable_thinking=False"); fi

case "${RSIPO_ENV:-alfworld}" in
  alfworld) launcher="$RSIPO_CODE/examples/seed_trainer/run_alfworld_sft_glm_self.sh" ;;
  search) launcher="$RSIPO_CODE/examples/seed_trainer/_common/search.sh"; export history_length="${RSIPO_HISTORY_LENGTH:-4}" ;;
  *) echo "Unknown RSIPO_ENV=$RSIPO_ENV (alfworld or search)" >&2; exit 2 ;;
esac

mkdir -p "$phase_dir"
exec bash "$launcher" \
 env.seed="${RSIPO_SEED:-0}" +data.seed="${RSIPO_SEED:-0}" ++actor_rollout_ref.rollout.seed="${RSIPO_SEED:-0}" \
 trainer.nnodes="${RSIPO_NNODES:-1}" trainer.n_gpus_per_node="${RSIPO_GPUS_PER_NODE:-8}" trainer.total_training_steps="$RSIPO_STEPS" \
 trainer.resume_mode="${RSIPO_RESUME_MODE:-disable}" trainer.resume_from_path="${RSIPO_RESUME_FROM:-null}" \
 trainer.save_freq="$RSIPO_SAVE_FREQ" trainer.test_freq=-1 trainer.val_before_train="${RSIPO_VAL_BEFORE_TRAIN:-False}" \
 trainer.max_actor_ckpt_to_keep=3 trainer.max_critic_ckpt_to_keep=3 \
 trainer.ray_wait_register_center_timeout=900 "trainer.logger=${RSIPO_LOGGER:-['console']}" \
 actor_rollout_ref.actor.ppo_mini_batch_size="${RSIPO_PPO_MINI_BATCH:-256}" \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="$PPO_MICRO_BATCH_SIZE_PER_GPU" \
 actor_rollout_ref.actor.fsdp_config.fsdp_size="${RSIPO_FSDP_SIZE:-8}" +actor_rollout_ref.ref.fsdp_config.fsdp_size="${RSIPO_FSDP_SIZE:-8}" \
 actor_rollout_ref.actor.fsdp_config.param_offload=True actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
 actor_rollout_ref.actor.use_torch_compile=False actor_rollout_ref.rollout.dtype=bfloat16 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization="${RSIPO_GPU_MEM_UTIL:-0.40}" actor_rollout_ref.rollout.max_num_seqs="${RSIPO_MAX_NUM_SEQS:-16}" \
 actor_rollout_ref.rollout.enforce_eager="${RSIPO_ENFORCE_EAGER:-False}" \
 actor_rollout_ref.rollout.free_cache_engine="${RSIPO_FREE_CACHE_ENGINE:-False}" \
 actor_rollout_ref.rollout.enable_chunked_prefill=True actor_rollout_ref.rollout.max_num_batched_tokens=20480 \
 env.max_steps="${RSIPO_MAX_STEPS:-50}" \
 env.history_length="${RSIPO_HISTORY_LENGTH:-5}" \
 rsipo.enable=True rsipo.seed="${RSIPO_SEED:-0}" rsipo.format_penalty="${RSIPO_FORMAT_PENALTY:-0.2}" \
 actor_rollout_ref.actor.skill_gen_loss_coef="$RSIPO_SKILL_COEF" actor_rollout_ref.actor.skill_gen_same_step=True \
 +ray_init.runtime_env.env_vars.PYTHONPATH="$RSIPO_CODE" \
 +ray_init.runtime_env.env_vars.NCCL_DEBUG=WARN \
 '+ray_init.runtime_env.env_vars.RSI_FSDP_NO_SYNC="1"' \
 '+ray_init.runtime_env.env_vars.CUDA_DEVICE_MAX_CONNECTIONS="8"' \
 '+ray_init.runtime_env.env_vars.TORCH_NCCL_ASYNC_ERROR_HANDLING="1"' \
 +ray_init.runtime_env.env_vars.RSI_OPD_METRICS_JSONL="$RSI_OPD_METRICS_JSONL" \
 +ray_init.runtime_env.env_vars.RSI_ACTOR_PROGRESS_DIR="$RSI_ACTOR_PROGRESS_DIR" \
 "+ray_init.runtime_env.env_vars.RSIPO_RAY_NODES=\"$RSIPO_RAY_NODES\"" \
 ${RSIPO_ANALYSIS_REMINDER:+"+algorithm.seed.analysis_prompt_reminder=True"} \
 ${RSIPO_SKILL_GENERAL_INSTRUCTION:+"+rsipo.skill_general_instruction=True"} \
 ${RSIPO_FORMAT_BONUS:+"+rsipo.format_bonus=$RSIPO_FORMAT_BONUS"} \
 ${RSIPO_VAL_ONLY:+"+trainer.val_only=True"} \
 ${thinking[@]+"${thinking[@]}"} \
 ${RSIPO_RAY_ADDRESS:+"+ray_init.address=$RSIPO_RAY_ADDRESS"} ray_init.num_cpus=null "$@"
