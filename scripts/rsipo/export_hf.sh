#!/usr/bin/env bash
# Merge the FSDP actor of one training step into a Hugging Face model (bf16) and optionally upload it.
# Usage: bash scripts/rsipo/export_hf.sh RUN_DIR [STEP] [HF_REPO]
#   RUN_DIR is $RSIPO_WORK/runs/<run>. The model goes to RUN_DIR/rsipo/model-hf-step<STEP> (reused if it exists, e.g.
#   after the ALFWorld evaluation); HF_REPO (user/name) uploads it with the hf CLI, so run `hf auth login` first.
set -euo pipefail
run="${1:?run directory}"
step="${2:-150}"
repo="${3:-}"
code="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$code/scripts/rsipo/env.sh"
ckpt="$run/rsipo/checkpoints/global_step_$step/actor"
merged="$run/rsipo/model-hf-step$step"
if [[ ! -f "$merged/config.json" ]]; then
  [[ -d "$ckpt" ]] || { echo "No checkpoint at $ckpt" >&2; exit 3; }
  CUDA_VISIBLE_DEVICES="" python "$code/scripts/model_merger.py" merge --backend fsdp --local_dir "$ckpt" --target_dir "$merged"
fi
echo "Hugging Face model (bf16): $merged"
if [[ -n "$repo" ]]; then
  hf upload "$repo" "$merged" . --repo-type model
fi
