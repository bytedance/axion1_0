#!/usr/bin/env bash
# Modified by the Axion authors for package and resource paths.
# Initialize the FAISS search store: merge shards -> build FAISS index.
# Read paths from config/tools.yaml; optionally use GPU to accelerate index building.
# Before running, install: pip install faiss-cpu sentence-transformers pyyaml

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$VERL_ROOT"

# Optional: choose which GPU to use for embeddings. GPU is recommended for index building.
# export CUDA_VISIBLE_DEVICES=0

# By default, read cache_dir / cache_file / shards / faiss_embedding_model from axion/training/rl/config/tools.yaml.
# Use GPU for index building, automatically selecting cuda when available.
python -m axion.training.rl.scripts.build_search_faiss --device cuda "$@"

# To build only FAISS without merging, for example after a previous merge:
# python -m axion.training.rl.scripts.build_search_faiss --skip-merge --device cuda
