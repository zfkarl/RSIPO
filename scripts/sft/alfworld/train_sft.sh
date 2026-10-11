#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET_NAME="alfworld"
DATASET_LABEL="ALFWorld"
MODEL_BASENAME="Qwen2.5-3B-Instruct"
MODEL_TAG="qwen25_3b"
DEFAULT_SFT_CONDA_ENV="skillrl"
DEFAULT_LR="5e-6"
DEFAULT_MAX_LENGTH="8192"
MULTIMODAL="false"

# shellcheck source=../_common/trainer.sh
source "$PROJECT_ROOT/scripts/sft/_common/trainer.sh"
