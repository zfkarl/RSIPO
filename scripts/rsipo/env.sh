#!/usr/bin/env bash
# Environment shared by the RSIPO launchers. Values can be set in the repository's .env file (see .env.example) or in
# the calling shell; the defaults follow the layout in README.md.
RSIPO_CODE="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export RSIPO_CODE
if [[ -f "$RSIPO_CODE/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$RSIPO_CODE/.env"
  set +a
fi
export RSIPO_WORK="${RSIPO_WORK:-$RSIPO_CODE/work}"
export MODELS_ROOT="${MODELS_ROOT:-$RSIPO_WORK/models}"
export SEARCH_DATA="${SEARCH_DATA:-$RSIPO_WORK/data}"
export ALFWORLD_DATA="${ALFWORLD_DATA:-$HOME/.cache/alfworld}"
export WANDB_MODE="${WANDB_MODE:-offline}" WANDB_DIR="${WANDB_DIR:-$RSIPO_WORK/runs}"
export VLLM_USE_V1=0 VLLM_ATTENTION_BACKEND=FLASH_ATTN
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0 PYTHONUNBUFFERED=1 HYDRA_FULL_ERROR=1
export RAY_agent_register_timeout_ms=300000 RAY_worker_register_timeout_seconds=600 RAY_USAGE_STATS_ENABLED=0
export RAY_enable_worker_prestart=0 RAY_prestart_worker_first_driver=0
export NCCL_CUMEM_ENABLE=0
export PYTHONPATH="$RSIPO_CODE${PYTHONPATH:+:$PYTHONPATH}"
