#!/usr/bin/env bash

# Shared implementation sourced by scripts/sft/<dataset>/train_sft.sh.

ENV_FILE="${ENV_FILE:-$PROJECT_ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
    set -a
    # shellcheck disable=SC1090
    source "$ENV_FILE"
    set +a
fi

# shellcheck source=teacher_naming.sh
source "$PROJECT_ROOT/scripts/sft/_common/teacher_naming.sh"

: "${MODELS_ROOT:?Please set MODELS_ROOT in $ENV_FILE or the environment.}"

SFT_CONDA_ENV="${SFT_CONDA_ENV:-${CONDA_ENV:-$DEFAULT_SFT_CONDA_ENV}}"
SFT_CUDA_VISIBLE_DEVICES="${SFT_CUDA_VISIBLE_DEVICES:-}"
NPROC_PER_NODE="${NPROC_PER_NODE:-8}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
LR="${LR:-$DEFAULT_LR}"
MAX_LENGTH="${MAX_LENGTH:-$DEFAULT_MAX_LENGTH}"
TRUNCATION="${TRUNCATION:-error}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
MICRO_BATCH_SIZE_PER_GPU="${MICRO_BATCH_SIZE_PER_GPU:-1}"
ULYSSES_SEQUENCE_PARALLEL_SIZE="${ULYSSES_SEQUENCE_PARALLEL_SIZE:-1}"
TRAINER_LOGGER="${TRAINER_LOGGER:-['console','wandb']}"
EXPORT_MODEL_AFTER_TRAIN="${EXPORT_MODEL_AFTER_TRAIN:-true}"

MODEL_PATH="${MODEL_PATH:-$MODELS_ROOT/$MODEL_BASENAME}"
DATA_DIR_NAME="${DATASET_NAME}_episode_skill_pipeline_${MODEL_TAG}_${SFT_SELF_DIR_SUFFIX}${DATA_DIR_EXTRA_SUFFIX:-}"
DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/outputs/$DATA_DIR_NAME}"
TRAIN_DATA="${TRAIN_DATA:-$DATA_DIR/sft_episode_skill_train.parquet}"
VAL_DATA="${VAL_DATA:-$DATA_DIR/sft_episode_skill_val.parquet}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
PROJECT_NAME="${PROJECT_NAME:-${DATASET_NAME}-episode-skill-sft}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-${MODEL_TAG}-${DATASET_NAME}-episode-skill-sft-${SFT_SELF_SUFFIX}-ep${TOTAL_EPOCHS}}"
OUTPUT_STEM="${OUTPUT_STEM:-${DATASET_NAME}_episode_skill_sft_${MODEL_TAG}_${SFT_SELF_DIR_SUFFIX}_ep${TOTAL_EPOCHS}_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$MODELS_ROOT/outputs/sft/$OUTPUT_STEM}"
LOG_DIR="${LOG_DIR:-$MODELS_ROOT/logs/sft}"
LOG_FILE="${LOG_FILE:-$LOG_DIR/$OUTPUT_STEM.log}"
EXPORT_MODEL_DIR="${EXPORT_MODEL_DIR:-$MODELS_ROOT/$MODEL_BASENAME-${DATASET_NAME}-episode-skill-sft-${SFT_SELF_SUFFIX}}"

if [[ ! -f "$TRAIN_DATA" ]]; then
    echo "Train parquet not found: $TRAIN_DATA" >&2
    echo "Run scripts/sft/$DATASET_NAME/prepare_data.sh first, or set DATA_DIR." >&2
    exit 1
fi
if [[ ! -f "$VAL_DATA" ]]; then
    echo "Validation parquet not found: $VAL_DATA" >&2
    exit 1
fi
if [[ "$MODEL_PATH" == /* && ! -d "$MODEL_PATH" && ! -f "$MODEL_PATH" ]]; then
    echo "Model path not found: $MODEL_PATH" >&2
    exit 1
fi

if [[ -n "$SFT_CUDA_VISIBLE_DEVICES" ]]; then
    export CUDA_VISIBLE_DEVICES="$SFT_CUDA_VISIBLE_DEVICES"
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR"
cd "$PROJECT_ROOT"

echo "Running $DATASET_LABEL episode-skill SFT"
echo "  conda env:      ${SFT_CONDA_ENV:-current environment}"
echo "  model:          $MODEL_PATH"
echo "  train data:     $TRAIN_DATA"
echo "  val data:       $VAL_DATA"
echo "  output dir:     $OUTPUT_DIR"
echo "  epochs:         $TOTAL_EPOCHS"
echo "  learning rate:  $LR"
echo "  max length:     $MAX_LENGTH"
echo "  nproc:          $NPROC_PER_NODE"
echo "  multimodal:     $MULTIMODAL"
echo "  export model:   $EXPORT_MODEL_AFTER_TRAIN"
if [[ "$EXPORT_MODEL_AFTER_TRAIN" == "true" ]]; then
    echo "  export dir:     $EXPORT_MODEL_DIR"
fi

if [[ -n "$SFT_CONDA_ENV" ]]; then
    set +u
    eval "$(conda shell.bash hook)"
    conda activate "$SFT_CONDA_ENV"
    set -u
fi

trainer_args=(
    "data.train_files=$TRAIN_DATA"
    "data.val_files=$VAL_DATA"
    "data.train_batch_size=$TRAIN_BATCH_SIZE"
    "data.micro_batch_size_per_gpu=$MICRO_BATCH_SIZE_PER_GPU"
    "data.prompt_key=prompt"
    "data.response_key=response"
    "data.prompt_dict_keys=[]"
    "data.response_dict_keys=[]"
    "data.max_length=$MAX_LENGTH"
    "data.truncation=$TRUNCATION"
    "model.partial_pretrain=$MODEL_PATH"
    "model.enable_gradient_checkpointing=True"
    "optim.lr=$LR"
    "trainer.default_local_dir=$OUTPUT_DIR"
    "trainer.project_name=$PROJECT_NAME"
    "trainer.experiment_name=$EXPERIMENT_NAME"
    "trainer.logger=$TRAINER_LOGGER"
    "trainer.total_epochs=$TOTAL_EPOCHS"
    "trainer.default_hdfs_dir=null"
)

if [[ "$MULTIMODAL" == "true" ]]; then
    trainer_args+=(
        "data.image_key=images"
        "model.trust_remote_code=True"
        "ulysses_sequence_parallel_size=1"
        "use_remove_padding=false"
    )
else
    trainer_args+=(
        "ulysses_sequence_parallel_size=$ULYSSES_SEQUENCE_PARALLEL_SIZE"
        "use_remove_padding=true"
    )
fi

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC_PER_NODE" \
    -m verl.trainer.fsdp_sft_trainer \
    "${trainer_args[@]}" \
    "$@" \
    2>&1 | tee "$LOG_FILE"

if [[ "$EXPORT_MODEL_AFTER_TRAIN" != "true" ]]; then
    exit 0
fi

latest_checkpoint="$(
    find "$OUTPUT_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' \
        | sort -V \
        | tail -n 1
)"
if [[ -z "$latest_checkpoint" ]]; then
    echo "No global_step_* checkpoint found in $OUTPUT_DIR" >&2
    exit 1
fi

latest_checkpoint="$OUTPUT_DIR/$latest_checkpoint"
if [[ ! -f "$latest_checkpoint/config.json" ]]; then
    echo "Latest checkpoint does not look like an HF model: $latest_checkpoint" >&2
    exit 1
fi

export_parent="$(dirname "$EXPORT_MODEL_DIR")"
export_tmp="${EXPORT_MODEL_DIR}.tmp.${RUN_ID}"
export_backup="${EXPORT_MODEL_DIR}.bak.${RUN_ID}"
mkdir -p "$export_parent"

if [[ -e "$export_tmp" ]]; then
    echo "Temporary export path already exists: $export_tmp" >&2
    exit 1
fi
if [[ -e "$EXPORT_MODEL_DIR" && -e "$export_backup" ]]; then
    echo "Backup export path already exists: $export_backup" >&2
    exit 1
fi

mkdir -p "$export_tmp"
if ! cp -al "$latest_checkpoint"/. "$export_tmp"/ 2>/dev/null; then
    cp -a "$latest_checkpoint"/. "$export_tmp"/
fi
if [[ ! -f "$export_tmp/config.json" ]]; then
    echo "Exported directory does not look like an HF model: $export_tmp" >&2
    exit 1
fi

if [[ -e "$EXPORT_MODEL_DIR" ]]; then
    mv "$EXPORT_MODEL_DIR" "$export_backup"
    echo "Previous export moved to: $export_backup"
fi
mv "$export_tmp" "$EXPORT_MODEL_DIR"
echo "Exported SFT model to $EXPORT_MODEL_DIR"
