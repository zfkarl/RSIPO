#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-copd}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-$CONDA_ENV}"
RUN_MODE="${RUN_MODE:-full}"  # full or smoke
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
ALF_CONFIG="${ALF_CONFIG:-$PROJECT_ROOT/agent_system/environments/env_package/alfworld/configs/config_tw.yaml}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

MODEL_PATH="${MODEL_PATH:-${POLICY_MODEL_PATH:-${MODELS_ROOT:-Qwen2.5-3B-Instruct}}}"
POLICY_MODEL_PATH="$MODEL_PATH"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

HOST="${HOST:-${POLICY_HOST:-127.0.0.1}}"
PORT="${PORT:-${POLICY_PORT:-60001}}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://${HOST}:${PORT}/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL="${POLICY_MODEL:-$MODEL_NAME}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.4}"
POLICY_MAX_COMPLETION_TOKENS="${POLICY_MAX_COMPLETION_TOKENS:-512}"
POLICY_TIMEOUT="${POLICY_TIMEOUT:-120}"
POLICY_RETRIES="${POLICY_RETRIES:-2}"
POLICY_RETRY_DELAY="${POLICY_RETRY_DELAY:-1.0}"

SKILL_BASE_URL="${SKILL_BASE_URL:-${OPENAI_BASE_URL:?Please set OPENAI_BASE_URL in .env or SKILL_BASE_URL.}}"
SKILL_API_KEY="${SKILL_API_KEY:-${OPENAI_API_KEY:?Please set OPENAI_API_KEY in .env or SKILL_API_KEY.}}"
SKILL_MODEL="${SKILL_MODEL:-${OPENAI_MODEL:?Please set OPENAI_MODEL in .env or SKILL_MODEL.}}"
SKILL_TEMPERATURE="${SKILL_TEMPERATURE:-0.0}"
SKILL_MAX_COMPLETION_TOKENS="${SKILL_MAX_COMPLETION_TOKENS:-1024}"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-120}"
SKILL_RETRIES="${SKILL_RETRIES:-5}"
SKILL_RETRY_DELAY="${SKILL_RETRY_DELAY:-1.0}"
SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-128}"
# shellcheck source=../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

START_VLLM="${START_VLLM:-1}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
STOP_VLLM_AFTER_API="${STOP_VLLM_AFTER_API:-${STOP_VLLM_AFTER_BASELINE:-1}}"
STOP_EXISTING_VLLM_AFTER_API="${STOP_EXISTING_VLLM_AFTER_API:-${STOP_EXISTING_VLLM_AFTER_BASELINE:-$STOP_VLLM_AFTER_API}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$PROJECT_ROOT/logs/vllm}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$VLLM_LOG_DIR/alfworld_policy_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"

MAX_STEPS="${MAX_STEPS:-30}"
HISTORY_LENGTH="${HISTORY_LENGTH:-5}"
SEED="${SEED:-2026}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-128}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
FALLBACK_ACTION="${FALLBACK_ACTION:-look}"
CHECK_ALFWORLD_DATA="${CHECK_ALFWORLD_DATA:-true}"
RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"
REGENERATE_CANDIDATES="${REGENERATE_CANDIDATES:-false}"
PROGRESS_MONITOR="${PROGRESS_MONITOR:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"

if [[ "$RUN_MODE" == "smoke" ]]; then
    TASKS_PER_TYPE="${TASKS_PER_TYPE:-1}"
    ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-1}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-2}"
    OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/alfworld_episode_skill_pipeline_smoke_${SFT_SELF_DIR_SUFFIX}}"
elif [[ "$RUN_MODE" == "full" ]]; then
    TASKS_PER_TYPE="${TASKS_PER_TYPE:-30}"
    ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
    MAX_CANDIDATES="${MAX_CANDIDATES:-}"
    OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT/outputs/alfworld_episode_skill_pipeline_qwen25_3b_${SFT_SELF_DIR_SUFFIX}}"
else
    echo "Unsupported RUN_MODE='$RUN_MODE'. Use RUN_MODE=full or RUN_MODE=smoke." >&2
    exit 2
fi

DEFAULT_REQUEST_WORKERS=$((TASK_BATCH_SIZE * ROLLOUTS_PER_TASK))
if (( DEFAULT_REQUEST_WORKERS < 1 )); then
    DEFAULT_REQUEST_WORKERS=1
fi
REQUEST_WORKERS="${REQUEST_WORKERS:-$DEFAULT_REQUEST_WORKERS}"

args=(
    "$SCRIPT_DIR/pipeline.py"
    --env-file "$ENV_FILE"
    --alf-config "$ALF_CONFIG"
    --output-dir "$OUTPUT_DIR"
    --tasks-per-type "$TASKS_PER_TYPE"
    --rollouts-per-task "$ROLLOUTS_PER_TASK"
    --task-batch-size "$TASK_BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --history-length "$HISTORY_LENGTH"
    --request-workers "$REQUEST_WORKERS"
    --policy-base-url "$POLICY_BASE_URL"
    --policy-api-key "$POLICY_API_KEY"
    --policy-model "$POLICY_MODEL"
    --policy-temperature "$POLICY_TEMPERATURE"
    --policy-max-completion-tokens "$POLICY_MAX_COMPLETION_TOKENS"
    --policy-timeout "$POLICY_TIMEOUT"
    --policy-retries "$POLICY_RETRIES"
    --policy-retry-delay "$POLICY_RETRY_DELAY"
    --fallback-action "$FALLBACK_ACTION"
    --skill-base-url "$SKILL_BASE_URL"
    --skill-api-key "$SKILL_API_KEY"
    --skill-model "$SKILL_MODEL"
    --skill-temperature "$SKILL_TEMPERATURE"
    --skill-max-completion-tokens "$SKILL_MAX_COMPLETION_TOKENS"
    --skill-timeout "$SKILL_TIMEOUT"
    --skill-retries "$SKILL_RETRIES"
    --skill-retry-delay "$SKILL_RETRY_DELAY"
    --skill-gen-workers "$SKILL_GEN_WORKERS"
    --sft-val-ratio "$SFT_VAL_RATIO"
    --seed "$SEED"
)

export SKILL_OPENAI_API_KEY="$SKILL_API_KEY"
export SKILL_OPENAI_BASE_URL="$SKILL_BASE_URL"
export SKILL_OPENAI_MODEL="$SKILL_MODEL"

if [[ -n "${BASELINE_ROLLOUTS:-}" ]]; then
    args+=(--baseline-rollouts "$BASELINE_ROLLOUTS")
fi

if [[ -n "$MAX_CANDIDATES" ]]; then
    args+=(--max-candidates "$MAX_CANDIDATES")
fi

if [[ -n "${MAX_TASKS:-}" ]]; then
    args+=(--max-tasks "$MAX_TASKS")
fi

if [[ -n "${POLICY_EXTRA_BODY_JSON:-}" ]]; then
    args+=(--policy-extra-body-json "$POLICY_EXTRA_BODY_JSON")
fi

if [[ -n "${SKILL_EXTRA_BODY_JSON:-}" ]]; then
    args+=(--skill-extra-body-json "$SKILL_EXTRA_BODY_JSON")
fi

if [[ "$OVERWRITE" == "true" ]]; then
    args+=(--overwrite)
elif [[ "$REGENERATE_CANDIDATES" == "true" ]]; then
    args+=(--regenerate-candidates)
elif [[ "$RESUME" == "true" ]]; then
    args+=(--resume)
fi

cd "$PROJECT_ROOT"

if [[ "$CHECK_ALFWORLD_DATA" == "true" && -z "${BASELINE_ROLLOUTS:-}" ]]; then
    if [[ -z "${ALFWORLD_DATA:-}" ]]; then
        echo "Please set ALFWORLD_DATA in .env. Expected: \$ALFWORLD_DATA/json_2.1.1/train" >&2
        exit 1
    fi
    if [[ ! -d "$ALFWORLD_DATA/json_2.1.1/train" ]]; then
        echo "ALFWorld train split not found: $ALFWORLD_DATA/json_2.1.1/train" >&2
        exit 1
    fi
fi

if [[ -n "${BASELINE_ROLLOUTS:-}" && ! -f "$BASELINE_ROLLOUTS" ]]; then
    echo "Baseline rollouts not found: $BASELINE_ROLLOUTS" >&2
    exit 1
fi

server_pid=""
progress_monitor_pid=""

is_server_ready() {
    curl -fsS "${POLICY_BASE_URL}/models" >/dev/null 2>&1
}

find_local_vllm_pid() {
    if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" && "$HOST" != "0.0.0.0" ]]; then
        return
    fi
    lsof -nP -iTCP:"$PORT" -sTCP:LISTEN -t 2>/dev/null | head -n 1 || true
}

print_progress() {
    local progress_file="$OUTPUT_DIR/progress.json"
    if [[ ! -f "$progress_file" ]]; then
        echo "[progress] waiting for $progress_file"
        return
    fi

    python3 - "$progress_file" <<'PY' || true
import json
import sys

with open(sys.argv[1], "r", encoding="utf-8") as f:
    progress = json.load(f)

parts = [
    f"[progress] {progress.get('updated_at', '')}",
    f"stage={progress.get('stage', 'unknown')}",
    f"status={progress.get('status', 'unknown')}",
]
for key in (
    "sampled_tasks",
    "expected_tasks",
    "task_types",
    "wave",
    "total_waves",
    "completed_rollouts",
    "expected_rollouts",
    "baseline_rollouts",
    "used_for_generation",
    "completed_skills",
    "expected_skills",
    "completed_in_current_run",
    "pending_in_current_run",
    "parse_ok_skills",
    "sft_records",
    "last_skill_id",
    "last_parse_ok",
):
    if key in progress:
        parts.append(f"{key}={progress[key]}")
print(" ".join(parts), flush=True)
PY
}

cleanup() {
    if [[ -n "$progress_monitor_pid" ]]; then
        kill "$progress_monitor_pid" >/dev/null 2>&1 || true
        wait "$progress_monitor_pid" >/dev/null 2>&1 || true
    fi
    stop_vllm_server
}

stop_vllm_server() {
    if [[ -n "$server_pid" && "$KEEP_VLLM_ALIVE" != "1" ]]; then
        echo "Stopping policy vLLM server pid=$server_pid"
        kill "$server_pid" >/dev/null 2>&1 || true
        wait "$server_pid" >/dev/null 2>&1 || true
        server_pid=""
    fi
}
trap cleanup EXIT

if [[ "$START_VLLM" == "1" && -z "${BASELINE_ROLLOUTS:-}" ]]; then
    if is_server_ready; then
        echo "Using existing policy vLLM server at $POLICY_BASE_URL"
        if [[ "$STOP_EXISTING_VLLM_AFTER_API" == "1" && "$KEEP_VLLM_ALIVE" != "1" ]]; then
            server_pid="$(find_local_vllm_pid)"
            if [[ -n "$server_pid" ]]; then
                echo "Will stop existing local policy vLLM pid=$server_pid after API calls."
            else
                echo "Could not find a local vLLM pid for $POLICY_BASE_URL; it will not be stopped automatically."
            fi
        fi
    else
        if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
            echo "Policy model path not found: $MODEL_PATH" >&2
            echo "Set MODELS_ROOT in $ENV_FILE, or set MODEL_PATH/POLICY_MODEL_PATH explicitly." >&2
            exit 1
        fi
        mkdir -p "$VLLM_LOG_DIR"
        read -r -a VLLM_EXTRA_ARGS_ARRAY <<< "${VLLM_EXTRA_ARGS:-}"
        VLLM_SERVE_ARGS=(
            "$MODEL_PATH"
            --host "$HOST"
            --port "$PORT"
            --served-model-name "$POLICY_MODEL"
            --tensor-parallel-size "$TENSOR_PARALLEL_SIZE"
            --gpu-memory-utilization 0.6
            --max-model-len "$MAX_MODEL_LEN"
            --dtype "$DTYPE"
        )
        if [[ -n "$DATA_PARALLEL_SIZE" ]]; then
            VLLM_SERVE_ARGS+=(--data-parallel-size "$DATA_PARALLEL_SIZE")
        fi

        echo "Starting local policy vLLM server"
        echo "  model path:    $MODEL_PATH"
        echo "  served name:   $POLICY_MODEL"
        echo "  base url:      $POLICY_BASE_URL"
        echo "  vLLM log:      $VLLM_LOG_FILE"

        conda run -n "$VLLM_CONDA_ENV" --no-capture-output \
            "$VLLM_BIN" serve "${VLLM_SERVE_ARGS[@]}" \
            "${VLLM_EXTRA_ARGS_ARRAY[@]}" \
            >"$VLLM_LOG_FILE" 2>&1 &
        server_pid=$!

        deadline=$((SECONDS + VLLM_STARTUP_TIMEOUT))
        until is_server_ready; do
            if ! kill -0 "$server_pid" >/dev/null 2>&1; then
                echo "vLLM server exited before becoming ready. See $VLLM_LOG_FILE" >&2
                exit 1
            fi
            if (( SECONDS >= deadline )); then
                echo "Timed out waiting for vLLM at $POLICY_BASE_URL. See $VLLM_LOG_FILE" >&2
                exit 1
            fi
            sleep 2
        done
    fi
fi

if [[ "$PROGRESS_MONITOR" == "1" ]]; then
    mkdir -p "$OUTPUT_DIR"
    print_progress
    (
        while true; do
            sleep "$PROGRESS_INTERVAL"
            print_progress
        done
    ) &
    progress_monitor_pid=$!
fi

echo "Running ALFWorld episode-skill pipeline"
echo "  mode:              $RUN_MODE"
echo "  conda env:         $CONDA_ENV"
echo "  vLLM conda env:    $VLLM_CONDA_ENV"
echo "  output dir:        $OUTPUT_DIR"
echo "  policy model:      $MODEL_PATH"
echo "  policy endpoint:   $POLICY_BASE_URL"
echo "  skill model:       $SKILL_MODEL @ $SKILL_BASE_URL"
echo "  scale:             tasks/type=$TASKS_PER_TYPE rollouts/task=$ROLLOUTS_PER_TASK"
echo "  wave size:         task_batch=$TASK_BATCH_SIZE request_workers=$REQUEST_WORKERS"
echo "  stop vLLM after API:$STOP_VLLM_AFTER_API"
echo "  skill gen workers: $SKILL_GEN_WORKERS"
echo "  baseline source:   ${BASELINE_ROLLOUTS:-generated}"

run_pipeline() {
    conda run -n "$CONDA_ENV" python "${args[@]}" "$@"
}

run_remaining_pipeline() {
    local remaining_args=()
    local arg
    local has_resume=0
    for arg in "${args[@]}"; do
        if [[ "$arg" == "--overwrite" || "$arg" == "--regenerate-candidates" ]]; then
            continue
        fi
        if [[ "$arg" == "--resume" ]]; then
            has_resume=1
        fi
        remaining_args+=("$arg")
    done
    if [[ "$has_resume" != "1" ]]; then
        remaining_args+=(--resume)
    fi
    conda run -n "$CONDA_ENV" python "${remaining_args[@]}"
}

if [[ "$STOP_VLLM_AFTER_API" == "1" && -n "$server_pid" && "$KEEP_VLLM_ALIVE" != "1" ]]; then
    echo "Running baseline rollouts and skill API generation before stopping policy vLLM."
    set +e
    run_pipeline --stop-after-skill-generation
    run_status=$?
    set -e
    print_progress
    if [[ "$run_status" -ne 0 ]]; then
        exit "$run_status"
    fi
    stop_vllm_server
    echo "Running SFT export after policy vLLM has stopped."
    set +e
    run_remaining_pipeline
    run_status=$?
    set -e
else
    if [[ "$STOP_VLLM_AFTER_API" == "1" && "$KEEP_VLLM_ALIVE" != "1" && -z "$server_pid" && -z "${BASELINE_ROLLOUTS:-}" ]]; then
        echo "Policy vLLM was not started by this script; it will not be stopped after API calls."
    fi
    set +e
    run_pipeline
    run_status=$?
    set -e
fi

print_progress
exit "$run_status"
