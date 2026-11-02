#!/usr/bin/env bash
# Axion relocation: canonical package/resource paths; training behavior unchanged.
# Start the Search HTTP service on a separate machine, with embeddings on GPU and FAISS on CPU.
# Run directly. By default, it reads config/tools.yaml in this repo, including faiss_similarity_threshold and Serper settings.
# Override config: export SEARCH_SERVICE_CONFIG=/other/path/tools.yaml
# Override port: export SEARCH_SERVICE_PORT=8001
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Repository root: scripts -> rl -> training -> axion -> repository.
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
QUEST_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$VERL_ROOT"

DEEPRESEARCH_SECRETS_ENV="${DEEPRESEARCH_SECRETS_ENV:-$QUEST_ROOT/.secrets/deepresearch_api_keys.env}"
if [[ -f "$DEEPRESEARCH_SECRETS_ENV" ]]; then
  set -a
  source "$DEEPRESEARCH_SECRETS_ENV"
  set +a
fi

# Force offline HF loads: the embedding model is pre-staged on NAS (see
# DEEPRESEARCH_EMBEDDING_MODEL -> model/all-MiniLM-L6-v2). This node may not reach HF
# Hub; without offline mode a stray HF lookup hangs on the network and uvicorn never
# binds :8000, so the launcher's wait_for_port times out and the run aborts. A local
# model dir needs no network, so this is safe. Override by exporting HF_HUB_OFFLINE=0.
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"

# By default, use all visible GPUs: cuda:0/1/2 for reads (search_top1), cuda:3 for the writer thread.
# To use only one GPU, export CUDA_VISIBLE_DEVICES=0 before launch.
if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="0,1,2,3"
fi

# Default: 3 GPUs for reads and 1 GPU for writes. The writer flushes batches every 30s, up to 512 items per batch.
export SEARCH_FAISS_READ_GPUS="${SEARCH_FAISS_READ_GPUS:-0,1,2}"
export SEARCH_FAISS_WRITE_GPUS="${SEARCH_FAISS_WRITE_GPUS:-3}"
export SEARCH_FAISS_WRITE_FLUSH_MS="${SEARCH_FAISS_WRITE_FLUSH_MS:-30000}"
export SEARCH_FAISS_WRITE_BATCH_SIZE="${SEARCH_FAISS_WRITE_BATCH_SIZE:-512}"

# By default, use this repo tools.yaml so threshold/cache_dir match the training side.
CONFIG_PATH="${SEARCH_SERVICE_CONFIG:-$SCRIPT_DIR/../config/tools.yaml}"
PORT="${SEARCH_SERVICE_PORT:-8000}"

# Bind dual-stack (::), NOT 0.0.0.0. On an IPv6-only cluster, 0.0.0.0 binds
# ONLY IPv4, so the head's own 127.0.0.1 works but REMOTE rollout nodes connecting to
# [head-ipv6]:8000 get "[Errno 111] Connection refused" -> their searches all fail.
# :: is dual-stack (IPv6 + IPv4-mapped), so both local 127.0.0.1 and remote IPv6 reach
# it. Override with SEARCH_SERVICE_HOST for an IPv4-only cluster.
exec python -m axion.training.rl.tools.search_service \
  --host "${SEARCH_SERVICE_HOST:-::}" \
  --port "$PORT" \
  --config "$CONFIG_PATH" \
  "$@"
