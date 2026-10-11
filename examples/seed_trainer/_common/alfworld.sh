#!/usr/bin/env bash

# Internal implementation shared by the public ALFWorld SFT launchers.

set -x

# Runtime backend.
ENGINE=vllm

ulimit -u 65536
export VLLM_ATTENTION_BACKEND=FLASH_ATTN

# Model, data, and rollout scale.
MODELS_ROOT=${MODELS_ROOT:-}
if [[ -z "${MODEL_PATH:-}" ]]; then
    : "${MODELS_ROOT:?Please set MODEL_PATH through a public launcher, or set MODELS_ROOT}"
    MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct"
fi
TRAIN_DATA_SIZE=${TRAIN_DATA_SIZE:-16}
VAL_DATA_SIZE=${VAL_DATA_SIZE:-128}
GROUP_SIZE=${GROUP_SIZE:-8}
NUM_CPUS_PER_ENV_WORKER=${NUM_CPUS_PER_ENV_WORKER:-0.1}
PPO_MINI_BATCH_SIZE=${PPO_MINI_BATCH_SIZE:-256}
PPO_MICRO_BATCH_SIZE_PER_GPU=${PPO_MICRO_BATCH_SIZE_PER_GPU:-32}
TENSOR_MODEL_PARALLEL_SIZE=${TENSOR_MODEL_PARALLEL_SIZE:-1}

# SEED advantage and OPD teacher/OPD loss schedule.
SEED_MODE=${SEED_MODE:-mean_std_norm}
SEED_STEP_ADV_W=${SEED_STEP_ADV_W:-0.0}
SEED_EPISODE_SKILL_TEACHER_ADV_W=${SEED_EPISODE_SKILL_TEACHER_ADV_W:-0.0}
SEED_STEP_SKILL_TEACHER_ADV_W=${SEED_STEP_SKILL_TEACHER_ADV_W:-0.0}
SEED_SKILL_MODE=${SEED_SKILL_MODE:-episode_step}
SEED_SKILL_TEACHER_MODE=${SEED_SKILL_TEACHER_MODE:-step_priority}
SEED_OPD_START_AFTER_STEPS=${SEED_OPD_START_AFTER_STEPS:-null}
SEED_OPD_STOP_AFTER_STEPS=${SEED_OPD_STOP_AFTER_STEPS:-null}
SEED_OPD_LOSS_COEF=${SEED_OPD_LOSS_COEF:-0.01}
SEED_OPD_GATE_BETA=${SEED_OPD_GATE_BETA:-5.0}
SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU=${SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU:-${SEED_SKILL_GEN_MICRO_BATCH_SIZE:-1}}
SEED_SKILL_GEN_MAX_SAMPLES=${SEED_SKILL_GEN_MAX_SAMPLES:-all}
SEED_SKILL_GEN_VALID_JSON_BONUS=${SEED_SKILL_GEN_VALID_JSON_BONUS:-0.0}
SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS=${SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS:-0.0}
SEED_SKILL_GEN_TOO_LONG_PENALTY=${SEED_SKILL_GEN_TOO_LONG_PENALTY:-0.0}
SEED_SKILL_GEN_MAX_OUTPUT_CHARS=${SEED_SKILL_GEN_MAX_OUTPUT_CHARS:-1200}
SEED_SKILL_GEN_REWARD_CLIP=${SEED_SKILL_GEN_REWARD_CLIP:-2.0}
SEED_SKILL_GEN_FAILED_REWARD_MODE=${SEED_SKILL_GEN_FAILED_REWARD_MODE:-zero}

# SEED episode filtering and teacher prompt construction.
SEED_FAILED_ONLY=${SEED_FAILED_ONLY:-False}
SEED_FAILED_ONLY_AFTER_STEPS=${SEED_FAILED_ONLY_AFTER_STEPS:-null}
SEED_FAILURE_SUCCESS_THRESHOLD=${SEED_FAILURE_SUCCESS_THRESHOLD:-1.0}

# SEED episode + critical-step skill analysis with the on-policy vLLM policy.
SEED_ENABLE_ANALYSIS=${SEED_ENABLE_ANALYSIS:-True}
SEED_SELECTOR=${SEED_SELECTOR:-llm}
SEED_ANALYSIS_BACKEND=${SEED_ANALYSIS_BACKEND:-policy_vllm}
SEED_ANALYSIS_NUM_WORKERS=${SEED_ANALYSIS_NUM_WORKERS:-1}
SEED_ANALYSIS_CONTEXT_LENGTH=${SEED_ANALYSIS_CONTEXT_LENGTH:-16384}
SEED_ANALYSIS_MAX_COMPLETION_TOKENS=${SEED_ANALYSIS_MAX_COMPLETION_TOKENS:-4096}
SEED_ANALYSIS_MAX_MODEL_LEN=${SEED_ANALYSIS_MAX_MODEL_LEN:-20480}
SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ=${SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ:-5}

# Experiment naming and output location.
PROJECT_NAME=${PROJECT_NAME:-agentic_alfworld}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-seed_qwen2.5_3b_alfworld_sft}
DEFAULT_LOCAL_DIR=${DEFAULT_LOCAL_DIR:-$MODELS_ROOT/ckpt/$EXPERIMENT_NAME}

# Prompt observation history.
history_length=${history_length:-5}

python3 -m examples.data_preprocess.prepare \
    --mode text --local_dir "$DATA_ROOT/verl-agent" \
    --train_data_size "$TRAIN_DATA_SIZE" \
    --val_data_size "$VAL_DATA_SIZE"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=seed \
    data.train_files=$DATA_ROOT/verl-agent/text/train.parquet \
    data.val_files=$DATA_ROOT/verl-agent/text/test.parquet \
    data.train_batch_size=$TRAIN_DATA_SIZE \
    data.val_batch_size=$VAL_DATA_SIZE \
    data.max_prompt_length=2048 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$PPO_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.opd_loss_coef=$SEED_OPD_LOSS_COEF \
    actor_rollout_ref.actor.opd_gate_beta=$SEED_OPD_GATE_BETA \
    actor_rollout_ref.actor.skill_gen_micro_batch_size_per_gpu=$SEED_SKILL_GEN_MICRO_BATCH_SIZE_PER_GPU \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$TENSOR_MODEL_PARALLEL_SIZE \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_model_len=$SEED_ANALYSIS_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.max_num_batched_tokens=$SEED_ANALYSIS_MAX_MODEL_LEN \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.seed.step_advantage_w=$SEED_STEP_ADV_W \
    algorithm.seed.episode_skill_teacher_advantage_w=$SEED_EPISODE_SKILL_TEACHER_ADV_W \
    algorithm.seed.step_skill_teacher_advantage_w=$SEED_STEP_SKILL_TEACHER_ADV_W \
    algorithm.seed.skill_mode=$SEED_SKILL_MODE \
    algorithm.seed.skill_teacher_mode=$SEED_SKILL_TEACHER_MODE \
    algorithm.seed.opd_start_after_steps=$SEED_OPD_START_AFTER_STEPS \
    algorithm.seed.opd_stop_after_steps=$SEED_OPD_STOP_AFTER_STEPS \
    algorithm.seed.failed_only=$SEED_FAILED_ONLY \
    algorithm.seed.failed_only_after_steps=$SEED_FAILED_ONLY_AFTER_STEPS \
    algorithm.seed.failure_success_threshold=$SEED_FAILURE_SUCCESS_THRESHOLD \
    algorithm.seed.mode=$SEED_MODE \
    algorithm.seed.enable_analysis=$SEED_ENABLE_ANALYSIS \
    algorithm.seed.selector=$SEED_SELECTOR \
    algorithm.seed.analysis_backend=$SEED_ANALYSIS_BACKEND \
    algorithm.seed.analysis_num_workers=$SEED_ANALYSIS_NUM_WORKERS \
    algorithm.seed.analysis_context_length=$SEED_ANALYSIS_CONTEXT_LENGTH \
    algorithm.seed.analysis_max_completion_tokens=$SEED_ANALYSIS_MAX_COMPLETION_TOKENS \
    algorithm.seed.analysis_max_step_skills_per_traj=$SEED_ANALYSIS_MAX_STEP_SKILLS_PER_TRAJ \
    algorithm.seed.skill_gen.max_samples=$SEED_SKILL_GEN_MAX_SAMPLES \
    algorithm.seed.skill_gen.valid_json_bonus=$SEED_SKILL_GEN_VALID_JSON_BONUS \
    algorithm.seed.skill_gen.non_empty_skill_bonus=$SEED_SKILL_GEN_NON_EMPTY_SKILL_BONUS \
    algorithm.seed.skill_gen.too_long_penalty=$SEED_SKILL_GEN_TOO_LONG_PENALTY \
    algorithm.seed.skill_gen.max_output_chars=$SEED_SKILL_GEN_MAX_OUTPUT_CHARS \
    algorithm.seed.skill_gen.reward_clip=$SEED_SKILL_GEN_REWARD_CLIP \
    algorithm.seed.skill_gen.failed_reward_mode=$SEED_SKILL_GEN_FAILED_REWARD_MODE \
    algorithm.seed.normalize_teacher_adv=False \
    env.history_length=$history_length \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed=0 \
    env.max_steps=30 \
    env.rollout.n=$GROUP_SIZE \
    env.resources_per_worker.num_cpus=$NUM_CPUS_PER_ENV_WORKER \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=5 \
    trainer.total_epochs=160 \
    trainer.val_before_train=False \
    trainer.default_local_dir=$DEFAULT_LOCAL_DIR \
    trainer.rollout_data_dir=$DEFAULT_LOCAL_DIR \
    "$@"
