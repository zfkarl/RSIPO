#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-copd}"
VLLM_CONDA_ENV="${VLLM_CONDA_ENV:-copd}"
RETRIEVER_CONDA_ENV="${RETRIEVER_CONDA_ENV:-$CONDA_ENV}"
ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"

if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

SKILL_PROMPT_VERSION="${SKILL_PROMPT_VERSION:-seed}"

: "${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE.}"
DATA_ROOT="${DATA_ROOT:-$HOME/data}"
MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/Qwen2.5-3B-Instruct}"
MODEL_NAME="${MODEL_NAME:-$(basename "$MODEL_PATH")}"

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-60003}"
POLICY_BASE_URL="${POLICY_BASE_URL:-http://${HOST}:${PORT}/v1}"
POLICY_API_KEY="${POLICY_API_KEY:-EMPTY}"
POLICY_MODEL="${POLICY_MODEL:-$MODEL_NAME}"
POLICY_TEMPERATURE="${POLICY_TEMPERATURE:-0.4}"
POLICY_MAX_COMPLETION_TOKENS="${POLICY_MAX_COMPLETION_TOKENS:-512}"
POLICY_TIMEOUT="${POLICY_TIMEOUT:-120}"
POLICY_RETRIES="${POLICY_RETRIES:-2}"
POLICY_RETRY_DELAY="${POLICY_RETRY_DELAY:-1.0}"

if [[ -z "${SKILL_BASE_URL:-}" ]]; then
    : "${OPENAI_BASE_URL:?Please set OPENAI_BASE_URL in .env or SKILL_BASE_URL.}"
    SKILL_BASE_URL="$OPENAI_BASE_URL"
fi
if [[ -z "${SKILL_API_KEY:-}" ]]; then
    : "${OPENAI_API_KEY:?Please set OPENAI_API_KEY in .env or SKILL_API_KEY.}"
    SKILL_API_KEY="$OPENAI_API_KEY"
fi
if [[ -z "${SKILL_MODEL:-}" ]]; then
    : "${OPENAI_MODEL:?Please set OPENAI_MODEL in .env or SKILL_MODEL.}"
    SKILL_MODEL="$OPENAI_MODEL"
fi
# shellcheck source=../_common/teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"
if [[ -z "${OUTPUT_DIR:-}" ]]; then
    OUTPUT_DIR="$PROJECT_ROOT/outputs/search_episode_skill_pipeline_qwen25_3b_${SFT_SELF_DIR_SUFFIX}"
fi
SKILL_TEMPERATURE="${SKILL_TEMPERATURE:-0.0}"
SKILL_MAX_COMPLETION_TOKENS="${SKILL_MAX_COMPLETION_TOKENS:-1024}"
SKILL_TIMEOUT="${SKILL_TIMEOUT:-120}"
SKILL_RETRIES="${SKILL_RETRIES:-5}"
SKILL_RETRY_DELAY="${SKILL_RETRY_DELAY:-1.0}"
SKILL_GEN_WORKERS="${SKILL_GEN_WORKERS:-128}"

START_VLLM="${START_VLLM:-1}"
KEEP_VLLM_ALIVE="${KEEP_VLLM_ALIVE:-0}"
STOP_VLLM_AFTER_API="${STOP_VLLM_AFTER_API:-${STOP_VLLM_AFTER_BASELINE:-1}}"
STOP_EXISTING_VLLM_AFTER_API="${STOP_EXISTING_VLLM_AFTER_API:-${STOP_EXISTING_VLLM_AFTER_BASELINE:-$STOP_VLLM_AFTER_API}}"
VLLM_BIN="${VLLM_BIN:-vllm}"
VLLM_STARTUP_TIMEOUT="${VLLM_STARTUP_TIMEOUT:-600}"
VLLM_LOG_DIR="${VLLM_LOG_DIR:-$PROJECT_ROOT/logs/vllm}"
VLLM_LOG_FILE="${VLLM_LOG_FILE:-$VLLM_LOG_DIR/search_policy_${MODEL_NAME}_$(date +%Y%m%d_%H%M%S).log}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
DATA_PARALLEL_SIZE="${DATA_PARALLEL_SIZE:-8}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
DTYPE="${DTYPE:-auto}"

SEARCH_URL="${SEARCH_URL:-http://127.0.0.1:8000/retrieve}"
SEARCH_TOPK="${SEARCH_TOPK:-3}"
SEARCH_TIMEOUT="${SEARCH_TIMEOUT:-180}"
SEARCH_LOG_REQUESTS="${SEARCH_LOG_REQUESTS:-0}"
TRAIN_DATA="${TRAIN_DATA:-$HOME/data/searchR1_processed_direct/train.parquet}"

START_RETRIEVER="${START_RETRIEVER:-0}"
KEEP_RETRIEVER_ALIVE="${KEEP_RETRIEVER_ALIVE:-1}"
RETRIEVER_STARTUP_TIMEOUT="${RETRIEVER_STARTUP_TIMEOUT:-600}"
RETRIEVER_LOG_DIR="${RETRIEVER_LOG_DIR:-$PROJECT_ROOT/logs/search_retriever}"
RETRIEVER_LOG_FILE="${RETRIEVER_LOG_FILE:-$RETRIEVER_LOG_DIR/retriever_$(date +%Y%m%d_%H%M%S).log}"
RETRIEVER_PORT="${RETRIEVER_PORT:-8000}"
RETRIEVER_INDEX_PATH="${RETRIEVER_INDEX_PATH:-$DATA_ROOT/searchR1/e5_Flat.index}"
RETRIEVER_CORPUS_PATH="${RETRIEVER_CORPUS_PATH:-$DATA_ROOT/searchR1/wiki-18.jsonl}"
RETRIEVER_NAME="${RETRIEVER_NAME:-e5}"
RETRIEVER_MODEL="${RETRIEVER_MODEL:-intfloat/e5-base-v2}"
RETRIEVER_FAISS_GPU="${RETRIEVER_FAISS_GPU:-0}"

NUM_TASKS="${NUM_TASKS:-180}"
ROLLOUTS_PER_TASK="${ROLLOUTS_PER_TASK:-8}"
TASK_BATCH_SIZE="${TASK_BATCH_SIZE:-128}"
MAX_STEPS="${MAX_STEPS:-4}"
HISTORY_LENGTH="${HISTORY_LENGTH:-4}"
REQUEST_WORKERS="${REQUEST_WORKERS:-8}"
SFT_VAL_RATIO="${SFT_VAL_RATIO:-0.1}"
SEED="${SEED:-2026}"
RESUME="${RESUME:-false}"
OVERWRITE="${OVERWRITE:-false}"
REGENERATE_CANDIDATES="${REGENERATE_CANDIDATES:-false}"
PROGRESS_MONITOR="${PROGRESS_MONITOR:-1}"
PROGRESS_INTERVAL="${PROGRESS_INTERVAL:-30}"
if [[ -z "${FALLBACK_ANSWER:-}" ]]; then
    FALLBACK_ANSWER="I don't know"
fi

if [[ ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "Policy model path not found: $MODEL_PATH" >&2
    exit 1
fi
if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Search train parquet not found: $TRAIN_DATA" >&2
    exit 1
fi

args=(
    "$SCRIPT_DIR/pipeline.py"
    --env-file "$ENV_FILE"
    --output-dir "$OUTPUT_DIR"
    --train-data "$TRAIN_DATA"
    --num-tasks "$NUM_TASKS"
    --rollouts-per-task "$ROLLOUTS_PER_TASK"
    --task-batch-size "$TASK_BATCH_SIZE"
    --max-steps "$MAX_STEPS"
    --history-length "$HISTORY_LENGTH"
    --request-workers "$REQUEST_WORKERS"
    --search-url "$SEARCH_URL"
    --search-topk "$SEARCH_TOPK"
    --search-timeout "$SEARCH_TIMEOUT"
    --policy-base-url "$POLICY_BASE_URL"
    --policy-api-key "$POLICY_API_KEY"
    --policy-model "$POLICY_MODEL"
    --policy-temperature "$POLICY_TEMPERATURE"
    --policy-max-completion-tokens "$POLICY_MAX_COMPLETION_TOKENS"
    --policy-timeout "$POLICY_TIMEOUT"
    --policy-retries "$POLICY_RETRIES"
    --policy-retry-delay "$POLICY_RETRY_DELAY"
    --fallback-answer "$FALLBACK_ANSWER"
    --skill-base-url "$SKILL_BASE_URL"
    --skill-model "$SKILL_MODEL"
    --skill-temperature "$SKILL_TEMPERATURE"
    --skill-max-completion-tokens "$SKILL_MAX_COMPLETION_TOKENS"
    --skill-timeout "$SKILL_TIMEOUT"
    --skill-retries "$SKILL_RETRIES"
    --skill-retry-delay "$SKILL_RETRY_DELAY"
    --skill-gen-workers "$SKILL_GEN_WORKERS"
    --skill-prompt-version "$SKILL_PROMPT_VERSION"
    --sft-val-ratio "$SFT_VAL_RATIO"
    --seed "$SEED"
)

if [[ "$SEARCH_LOG_REQUESTS" == "1" || "$SEARCH_LOG_REQUESTS" == "true" || "$SEARCH_LOG_REQUESTS" == "True" ]]; then
    args+=(--search-log-requests)
fi
if [[ -n "${MAX_TASKS:-}" ]]; then
    args+=(--max-tasks "$MAX_TASKS")
fi
if [[ -n "${MAX_CANDIDATES:-}" ]]; then
    args+=(--max-candidates "$MAX_CANDIDATES")
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
set +u
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV"
set -u
export SKILL_OPENAI_API_KEY="$SKILL_API_KEY"

server_pid=""
retriever_pid=""
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

is_retriever_ready() {
    curl -fsS -X POST "$SEARCH_URL" \
        -H 'Content-Type: application/json' \
        -d '{"query":"test","topk":1,"return_scores":false}' \
        >/dev/null 2>&1
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

with open(sys.argv[1], encoding="utf-8") as f:
    progress = json.load(f)

keys = (
    "sampled_tasks",
    "completed_rollouts",
    "expected_rollouts",
    "wave",
    "total_waves",
    "completed_skills",
    "expected_skills",
    "completed_in_current_run",
    "pending_in_current_run",
    "parse_ok_skills",
    "sft_records",
    "last_skill_id",
    "last_parse_ok",
)
parts = [
    "[progress] " + str(progress.get("updated_at", "")),
    "stage=" + str(progress.get("stage", "unknown")),
    "status=" + str(progress.get("status", "unknown")),
]
parts.extend(f"{key}={progress[key]}" for key in keys if key in progress)
print(" ".join(parts), flush=True)
PY
}

cleanup() {
    if [[ -n "$progress_monitor_pid" ]]; then
        kill "$progress_monitor_pid" >/dev/null 2>&1 || true
        wait "$progress_monitor_pid" >/dev/null 2>&1 || true
    fi
    stop_vllm_server
    if [[ -n "$retriever_pid" && "$KEEP_RETRIEVER_ALIVE" != "1" ]]; then
        kill "$retriever_pid" >/dev/null 2>&1 || true
        wait "$retriever_pid" >/dev/null 2>&1 || true
    elif [[ -n "$retriever_pid" && "$KEEP_RETRIEVER_ALIVE" == "1" ]]; then
        disown "$retriever_pid" >/dev/null 2>&1 || true
    fi
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

if [[ "$START_RETRIEVER" == "1" || "$START_RETRIEVER" == "true" || "$START_RETRIEVER" == "True" ]]; then
    if is_retriever_ready; then
        echo "Using existing Search retriever at $SEARCH_URL"
    else
        mkdir -p "$RETRIEVER_LOG_DIR"
        RETRIEVER_ARGS=(
            examples/search/retriever/retrieval_server.py
            --index_path "$RETRIEVER_INDEX_PATH"
            --corpus_path "$RETRIEVER_CORPUS_PATH"
            --topk "$SEARCH_TOPK"
            --retriever_name "$RETRIEVER_NAME"
            --retriever_model "$RETRIEVER_MODEL"
            --port "$RETRIEVER_PORT"
        )
        if [[ "$RETRIEVER_FAISS_GPU" == "1" || "$RETRIEVER_FAISS_GPU" == "true" || "$RETRIEVER_FAISS_GPU" == "True" ]]; then
            RETRIEVER_ARGS+=(--faiss_gpu)
        fi
        conda run -n "$RETRIEVER_CONDA_ENV" --no-capture-output \
            python "${RETRIEVER_ARGS[@]}" \
            >"$RETRIEVER_LOG_FILE" 2>&1 &
        retriever_pid=$!
    fi
fi

deadline=$((SECONDS + RETRIEVER_STARTUP_TIMEOUT))
until is_retriever_ready; do
    if [[ -n "$retriever_pid" ]] && ! kill -0 "$retriever_pid" >/dev/null 2>&1; then
        echo "Search retriever exited before becoming ready. See $RETRIEVER_LOG_FILE" >&2
        exit 1
    fi
    if (( SECONDS >= deadline )); then
        echo "Timed out waiting for Search retriever at $SEARCH_URL." >&2
        echo "Start it first, or rerun with START_RETRIEVER=1 and valid RETRIEVER_* paths." >&2
        exit 1
    fi
    sleep 2
done

if [[ "$START_VLLM" == "1" ]]; then
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

echo "Running Search QA episode-skill pipeline"
echo "  conda env:          $CONDA_ENV"
echo "  vLLM conda env:     $VLLM_CONDA_ENV"
echo "  output dir:         $OUTPUT_DIR"
echo "  train data:         $TRAIN_DATA"
echo "  search endpoint:    $SEARCH_URL"
echo "  policy model:       $MODEL_PATH"
echo "  policy endpoint:    $POLICY_BASE_URL"
echo "  skill model:        $SKILL_MODEL @ $SKILL_BASE_URL"
echo "  skill prompt:       $SKILL_PROMPT_VERSION"
echo "  sampled tasks:      $NUM_TASKS"
echo "  rollouts per task:  $ROLLOUTS_PER_TASK"
echo "  task batch size:    $TASK_BATCH_SIZE"
echo "  stop vLLM after API:$STOP_VLLM_AFTER_API"
echo "  skill gen workers:  $SKILL_GEN_WORKERS"

run_pipeline() {
    python "${args[@]}" "$@"
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
    python "${remaining_args[@]}"
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
    if [[ "$STOP_VLLM_AFTER_API" == "1" && "$KEEP_VLLM_ALIVE" != "1" && -z "$server_pid" ]]; then
        echo "Policy vLLM was not started by this script; it will not be stopped after API calls."
    fi
    set +e
    run_pipeline
    run_status=$?
    set -e
fi

print_progress
exit "$run_status"
