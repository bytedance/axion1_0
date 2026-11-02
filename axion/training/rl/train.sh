#!/usr/bin/env bash
# Axion paper RL entry point. Uses the active Python/RL environment.
set -euo pipefail
AXION_RL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
AXION_REPO_ROOT="$(cd -- "$AXION_RL_DIR/../../.." && pwd)"
export PYTHONPATH="$AXION_REPO_ROOT:$AXION_RL_DIR${PYTHONPATH:+:$PYTHONPATH}"
exec python -m axion.training.rl "$@"
