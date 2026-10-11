#!/usr/bin/env bash
# Evaluate RSIPO policies without skills in the prompt.
# Usage: bash scripts/rsipo/evaluate.sh ENV MODEL [STEP]
#   Evaluates step STEP (default 150) of the run that run_rsipo.sh trained for ENV and MODEL.
# Usage: RSIPO_EVAL_HF=1 bash scripts/rsipo/evaluate.sh ENV MODEL
#   Evaluates the Hugging Face model $MODELS_ROOT/MODEL itself, e.g. a released checkpoint.
# alfworld: scripts/rsipo/evaluate.py runs the seen (140) and unseen (134) splits with seeds 1, 2 and 3 (temperature
#           0.4, at most 50 turns, one vLLM engine per GPU of this node) and writes summary.json under the run.
# search:   the trainer's validation loop over all test questions (temperature 0.4, at most 4 turns); needs a
#           retriever at SEARCH_URL.
set -euo pipefail
env_name="${1:?env (alfworld or search)}"
model="${2:?model name under MODELS_ROOT}"
step="${3:-150}"
code="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$code/scripts/rsipo/env.sh"
run="$RSIPO_WORK/runs/rsipo-${model}-${env_name}${RSIPO_RUN_SUFFIX:-}"
hf=()
[[ "${RSIPO_EVAL_HF:-0}" == 1 ]] && hf=(--model-dir "$MODELS_ROOT/$model") && step=0
case "$env_name" in
  alfworld)
    mkdir -p "$run/rsipo"
    # evaluate.py stripes every round over one engine per GPU of the nodes listed in ray-nodes.txt.
    hostname -I | awk '{print $1}' > "$run/ray-nodes.txt"
    cd "$code"
    python scripts/rsipo/evaluate.py --phase-dir "$run/rsipo" --step "$step" --max-steps 50 \
      --name "RSIPO-${model}-alfworld" ${hf[@]+"${hf[@]}"} ;;
  search)
    if (( ${#hf[@]} )); then
      RSIPO_VAL_ONLY=1 bash "$code/scripts/rsipo/run_rsipo.sh" search "$model"
    else
      RSIPO_EVAL_STEP="$step" bash "$code/scripts/rsipo/run_rsipo.sh" search "$model"
    fi ;;
  *) echo "Unknown env $env_name (alfworld or search)" >&2; exit 2 ;;
esac
