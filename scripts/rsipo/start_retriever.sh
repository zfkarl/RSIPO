#!/usr/bin/env bash
# Serve the Search-R1 e5 retriever (2018 Wikipedia, top 3 passages) on this node through retrieval_server.py, which is
# SEED's server with request batching. Runs on CPU by default so that it can share a node with training; pass
# --faiss_gpu with RETRIEVER_GPUS set to put the index on those GPUs instead.
# Usage: bash scripts/rsipo/start_retriever.sh [--port 8000] [--faiss_gpu]
set -euo pipefail
code="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "$code/scripts/rsipo/env.sh"
data="${RETRIEVER_DATA:-$SEARCH_DATA/searchR1}"
export CUDA_VISIBLE_DEVICES="${RETRIEVER_GPUS:-}"
# env.sh pins BLAS to one thread for training; exact search over 21M passages is a large matrix product.
threads="${RETRIEVER_THREADS:-64}"
export OMP_NUM_THREADS="$threads" MKL_NUM_THREADS="$threads" OPENBLAS_NUM_THREADS="$threads"
# Keep idle BLAS/OpenMP threads from spinning while training shares the node.
export OMP_WAIT_POLICY=PASSIVE GOMP_SPINCOUNT=0
cd "$code"
exec python scripts/rsipo/retrieval_server.py \
  --index_path "$data/e5_Flat.index" \
  --corpus_path "$data/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
  --port "${RETRIEVER_PORT:-8000}" "$@"
