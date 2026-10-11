#!/usr/bin/env bash
# Train RSIPO on one benchmark and backbone with the settings of the paper (one node with 8 GPUs, 150 steps).
# Usage: bash scripts/rsipo/run_rsipo.sh ENV MODEL
#   ENV is alfworld or search; MODEL is the SFT checkpoint under $MODELS_ROOT (see README.md).
#   Search also needs RSIPO_SEARCH_TRAIN_DATA (a train.parquet from scripts/data/) and a retriever at SEARCH_URL.
# Options (environment variables):
#   RSIPO_RUN_SUFFIX                                  appended to the run directory name
#   RSIPO_RESUME_MODE=auto                            resume from the latest checkpoint of the run
#   RSIPO_EVAL_STEP=N                                 evaluate the step-N checkpoint of a search run instead of training
#                                                     (scripts/rsipo/evaluate.sh calls this; ALFWorld uses evaluate.py)
set -euo pipefail
env_name="${1:?env (alfworld or search)}"
model="${2:?model name under MODELS_ROOT}"
code="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$code/scripts/rsipo/env.sh"
# PPO mini-batch and advantage normalization follow SEED's launchers (examples/seed_trainer/_common).
case "$env_name" in
  alfworld) max_steps=50; history=5; tasks=16; mini=256 ;;
  search) max_steps=4; history=4; tasks=128; mini=512 ;;
  *) echo "Unknown env $env_name (alfworld or search)" >&2; exit 2 ;;
esac
seed_mode=mean_std_norm
# vLLM memory share per GPU: a 7B policy needs more room for its weights.
case "$model" in *7B*) util=0.6 ;; *) util=0.40 ;; esac
# Skill analyses of the smaller models on search-based tasks restate the JSON instruction after the trajectory and
# are capped at 2048 tokens.
analysis_tokens=4096; reminder=false
if [[ "$env_name:$model" == search:*3B* || "$env_name:$model" == search:*1.7B* ]]; then
  analysis_tokens=2048; reminder=true
  export RSIPO_ANALYSIS_REMINDER=1
fi
export RSIPO_ANALYSIS_MAX_TOKENS="$analysis_tokens"
# On search-based tasks, unusable analyses are ignored instead of penalized (rho = 0), the skill-regeneration prompt
# asks for a general strategy that does not state the answer, and the tokens around a valid skill get a small bonus.
format_penalty=0.2; general_instruction=false; format_bonus=0
if [[ "$env_name" == search ]]; then
  format_penalty=0.0; general_instruction=true; format_bonus=0.05
  export RSIPO_SKILL_GENERAL_INSTRUCTION=1 RSIPO_FORMAT_BONUS="$format_bonus"
fi
export RSIPO_FORMAT_PENALTY="$format_penalty"
run="$RSIPO_WORK/runs/rsipo-${model}-${env_name}${RSIPO_RUN_SUFFIX:-}"
mkdir -p "$run/rsipo"
export RSIPO_RUN_ROOT="$run" RSIPO_PHASE=rsipo RSIPO_STEPS="${RSIPO_STEPS:-150}" RSIPO_SAVE_FREQ="${RSIPO_SAVE_FREQ:-10}"
export RSIPO_SKILL_COEF=1
export RSIPO_MODEL_NAME="$model" RSIPO_ENV="$env_name"
export RSIPO_MAX_STEPS="$max_steps" RSIPO_HISTORY_LENGTH="$history"
export RSIPO_TRAIN_TASKS="$tasks" RSIPO_NNODES="${RSIPO_NNODES:-1}" RSIPO_PPO_MINI_BATCH="$mini"
export RSIPO_SEED_MODE="$seed_mode" RSIPO_GPU_MEM_UTIL="$util" RSIPO_FSDP_SIZE="${RSIPO_FSDP_SIZE:-8}"
export RSIPO_ROLLOUT_TP=1 RSIPO_RESUME_MODE="${RSIPO_RESUME_MODE:-disable}"
export RSIPO_ENFORCE_EAGER=False RSIPO_FREE_CACHE_ENGINE=False
missing=()
[[ -f "$MODELS_ROOT/$model/config.json" ]] || missing+=("model $MODELS_ROOT/$model")
if [[ "$env_name" == search ]]; then
  export TRAIN_DATA="${RSIPO_SEARCH_TRAIN_DATA:-$SEARCH_DATA/searchR1_processed_direct/train.parquet}"
  export VAL_DATA="$SEARCH_DATA/searchR1_processed_direct/test.parquet"
  # One URL, or a Hydra list "[URL1,URL2]" that the search environments use round-robin.
  export SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
  [[ -f "$TRAIN_DATA" ]] || missing+=("$TRAIN_DATA")
  [[ -f "$VAL_DATA" ]] || missing+=("$VAL_DATA")
  for url in $(tr -d '[]' <<< "$SEARCH_URL" | tr ',' ' '); do
    curl -s -m 60 -X POST "$url" -H 'Content-Type: application/json' -d '{"query": "test", "topk": 1}' | grep -q '"result"' \
      || missing+=("retriever at $url (start it with scripts/rsipo/start_retriever.sh)")
  done
  # 1024 trajectories per step; 32 sequences per engine still fit the KV cache.
  export RSIPO_MAX_NUM_SEQS=32
fi
if (( ${#missing[@]} )); then
  printf 'Not starting; missing:\n'; printf '  %s\n' "${missing[@]}"; exit 3
fi
if [[ "${RSIPO_VAL_ONLY:-0}" == 1 ]]; then
  # Validate MODEL itself (e.g. a released checkpoint) on all test questions before any update; nothing is trained.
  [[ "$env_name" == search ]] || { echo "Evaluate ALFWorld models with scripts/rsipo/evaluate.sh" >&2; exit 2; }
  out="$run/rsipo/eval-model"
  mkdir -p "$out"
  export RSIPO_VAL_BEFORE_TRAIN=True RSIPO_VAL_DATA_SIZE=512 RSIPO_METRICS_JSONL="$out/metrics.jsonl"
  echo "validating $MODELS_ROOT/$model into $out"
  bash "$code/scripts/rsipo/train.sh" >> "$out/evaluation.log" 2>&1
  exit $?
fi
if [[ -n "${RSIPO_EVAL_STEP:-}" ]]; then
  # Evaluate a saved checkpoint through the trainer's validation loop (rsipo/evaluation.py); nothing is trained.
  [[ "$env_name" == search ]] || { echo "Evaluate ALFWorld with scripts/rsipo/evaluate.sh" >&2; exit 2; }
  ckpt="$run/rsipo/checkpoints/global_step_$RSIPO_EVAL_STEP"
  [[ -d "$ckpt/actor" ]] || { echo "No checkpoint at $ckpt" >&2; exit 3; }
  out="$run/rsipo/eval-step$RSIPO_EVAL_STEP${RSIPO_EVAL_TAG:-}"
  mkdir -p "$out"
  # All 51,713 test questions once, temperature 0.4, no skill in the prompt.
  export RSIPO_RESUME_MODE=resume_path RSIPO_RESUME_FROM="$ckpt" RSIPO_VAL_DATA_SIZE=512 RSIPO_METRICS_JSONL="$out/metrics.jsonl"
  echo "evaluating $ckpt into $out"
  bash "$code/scripts/rsipo/train.sh" "+rsipo_eval.dir=$out" "+rsipo_eval.rounds=${RSIPO_EVAL_ROUNDS:-1}" >> "$out/evaluation.log" 2>&1
  exit $?
fi
echo "RSIPO $model on $env_name: $tasks tasks x 8 rollouts, $RSIPO_STEPS steps, max turns $max_steps, history $history," \
     "mini-batch $mini, analysis tokens $analysis_tokens, format penalty $format_penalty, format bonus $format_bonus"
echo "run directory: $run (log: $run/rsipo/training.log)"
cat > "$run/rsipo/protocol.json" <<EOF
{"model": "$model", "env": "$env_name", "train_tasks_per_step": $tasks, "group_size": 8, "training_steps": $RSIPO_STEPS,
 "max_steps": $max_steps, "history_length": $history, "ppo_mini_batch_size": $mini, "seed_mode": "$seed_mode",
 "vllm_gpu_memory_utilization": $util, "analysis_max_completion_tokens": $analysis_tokens,
 "analysis_prompt_reminder": $reminder, "format_penalty": $format_penalty, "format_bonus": $format_bonus,
 "skill_general_instruction": $general_instruction, "train_data": "${TRAIN_DATA:-}", "seed": ${RSIPO_SEED:-0}}
EOF
bash "$code/scripts/rsipo/train.sh" >> "$run/rsipo/training.log" 2>&1
