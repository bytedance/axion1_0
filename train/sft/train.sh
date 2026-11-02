#!/usr/bin/env bash
set -euo pipefail
exec "${PYTHON:-python3}" -B "$(dirname "${BASH_SOURCE[0]}")/launcher.py" "$@"
