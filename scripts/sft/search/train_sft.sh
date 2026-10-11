#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

DATASET_NAME="search"
DATASET_LABEL="Search QA"
MODEL_BASENAME="Qwen2.5-3B-Instruct"
MODEL_TAG="qwen25_3b"
DEFAULT_SFT_CONDA_ENV="copd"
DEFAULT_LR="5e-6"
DEFAULT_MAX_LENGTH="12288"
MULTIMODAL="false"

# shellcheck source=../_common/trainer.sh
source "$PROJECT_ROOT/scripts/sft/_common/trainer.sh"
