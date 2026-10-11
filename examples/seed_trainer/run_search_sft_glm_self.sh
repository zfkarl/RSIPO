#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

CONDA_ENV="${CONDA_ENV:-copd}"
if [[ -n "$CONDA_ENV" && "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" ]]; then
    if ! command -v conda >/dev/null 2>&1; then
        echo "conda is required to activate environment: $CONDA_ENV" >&2
        exit 1
    fi
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$CONDA_ENV"
    set -u
fi

if [[ -z "${HF_MODEL_PATH:-}" ]]; then
    if [[ -n "${MODEL_PATH:-}" ]]; then
        HF_MODEL_PATH="$MODEL_PATH"
    elif [[ -n "${MODELS_ROOT:-}" ]]; then
        HF_MODEL_PATH="$MODELS_ROOT/Qwen2.5-3B-Instruct-search-episode-skill-sft-glm-self"
    else
        echo "Please set MODELS_ROOT in $ENV_FILE, or set HF_MODEL_PATH explicitly." >&2
        exit 1
    fi
fi
if [[ ! -f "$HF_MODEL_PATH/config.json" ]]; then
    echo "HF model not found: $HF_MODEL_PATH" >&2
    echo "Run scripts/sft/search/train_sft.sh first, or set HF_MODEL_PATH explicitly." >&2
    exit 1
fi

# SFT checkpoint and self-evolving SEED configuration.
export HF_MODEL_PATH
export MODEL_PATH="${MODEL_PATH:-$HF_MODEL_PATH}"
export SEED_SKILL_MODE="${SEED_SKILL_MODE:-episode_only}"
export SEED_OPD_LOSS_COEF="${SEED_OPD_LOSS_COEF:-0.01}"
export SEED_ANALYSIS_BACKEND="${SEED_ANALYSIS_BACKEND:-policy_vllm}"
export SEED_ANALYSIS_PROMPT_VERSION="${SEED_ANALYSIS_PROMPT_VERSION:-seed}"
export SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY="${SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY:-True}"
export SEED_MODE="${SEED_MODE:-mean_std_norm}"

# Environment-specific runtime scale.
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
N_GPUS_PER_NODE="${N_GPUS_PER_NODE:-8}"
SAVE_FREQ="${SAVE_FREQ:-150}"
TEST_FREQ="${TEST_FREQ:-150}"

export PROJECT_NAME="${PROJECT_NAME:-agentic_search}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-seed_qwen2.5_3b_search_sft_glm_self_mean-std-norm}"
if [[ -n "${MODELS_ROOT:-}" ]]; then
    default_local_dir="$MODELS_ROOT/ckpt/$EXPERIMENT_NAME"
else
    default_local_dir="$PROJECT_ROOT/outputs/rl/$EXPERIMENT_NAME"
fi
export DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$default_local_dir}"

cd "$PROJECT_ROOT"

echo "Running Search QA RL from GLM-self SFT model"
echo "  conda env:       ${CONDA_ENV:-<current shell>}"
echo "  model:           $MODEL_PATH"
echo "  seed mode:       $SEED_MODE"
echo "  experiment:      $EXPERIMENT_NAME"
echo "  output dir:      $DEFAULT_LOCAL_DIR"
if [[ -n "$HISTORY_LENGTH" ]]; then
    echo "  history length:  $HISTORY_LENGTH"
fi
echo "  train/val size:  ${TRAIN_DATA_SIZE:-16}/${VAL_DATA_SIZE:-128}"
echo "  group size:      ${GROUP_SIZE:-8}"
echo "  training steps:  $TOTAL_TRAINING_STEPS"
echo "  GPUs:            $N_GPUS_PER_NODE"

if [[ "${DRY_RUN:-false}" == "true" ]]; then
    echo "Dry run complete; RL training was not started."
    exit 0
fi

launch_args=(
    "algorithm.seed.analysis_include_episode_summary=$SEED_ANALYSIS_INCLUDE_EPISODE_SUMMARY"
    "algorithm.seed.analysis_prompt_version=$SEED_ANALYSIS_PROMPT_VERSION"
    "trainer.n_gpus_per_node=$N_GPUS_PER_NODE"
    "trainer.save_freq=$SAVE_FREQ"
    "trainer.test_freq=$TEST_FREQ"
    "trainer.total_training_steps=$TOTAL_TRAINING_STEPS"
)
if [[ -n "$HISTORY_LENGTH" ]]; then
    export history_length="$HISTORY_LENGTH"
    launch_args+=("env.history_length=$HISTORY_LENGTH")
fi

exec bash "$SCRIPT_DIR/_common/search.sh" "${launch_args[@]}" "$@"
