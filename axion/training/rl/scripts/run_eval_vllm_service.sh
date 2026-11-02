#!/usr/bin/env bash
# Start one or more local OpenAI-compatible vLLM eval/judge services.
# Run on a dedicated service node and supply its endpoints in the RL deployment
# configuration for judge and summarizer requests.
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#

set -euo pipefail

MODEL_PATH="${EVAL_VLLM_MODEL_PATH:?EVAL_VLLM_MODEL_PATH must be set by the launcher}"
SERVED_MODEL_NAME="${EVAL_VLLM_SERVED_MODEL_NAME:?EVAL_VLLM_SERVED_MODEL_NAME must be set by the launcher}"
PORTS_RAW="${EVAL_VLLM_PORTS:?EVAL_VLLM_PORTS must be set by the launcher}"
GPU_GROUPS_RAW="${EVAL_VLLM_GPU_GROUPS:?EVAL_VLLM_GPU_GROUPS must be set by the launcher}"
TP_SIZES_RAW="${EVAL_VLLM_TENSOR_PARALLEL_SIZES:-}"
MAX_MODEL_LEN="${EVAL_VLLM_MAX_MODEL_LEN:?EVAL_VLLM_MAX_MODEL_LEN must be set by the launcher}"
GPU_MEMORY_UTILIZATION="${EVAL_VLLM_GPU_MEMORY_UTILIZATION:?EVAL_VLLM_GPU_MEMORY_UTILIZATION must be set by the launcher}"
HOST="${EVAL_VLLM_HOST:-0.0.0.0}"
WAIT_RETRIES="${EVAL_VLLM_WAIT_RETRIES:-300}"
WAIT_INTERVAL_SECONDS="${EVAL_VLLM_WAIT_INTERVAL_SECONDS:-2}"
EXTRA_ARGS_RAW="${EVAL_VLLM_EXTRA_ARGS:-}"
# Per-instance overrides (comma-separated, one entry per port). If unset, the
# single MODEL_PATH / SERVED_MODEL_NAME / MAX_MODEL_LEN is used for every
# instance (backward compatible). Used to run a mixed fleet, e.g. 3x27B + 1x9B
# where the 9B endpoint serves the obj judge.
MODEL_PATHS_RAW="${EVAL_VLLM_MODEL_PATHS:-}"
SERVED_NAMES_RAW="${EVAL_VLLM_SERVED_MODEL_NAMES:-}"
MAX_MODEL_LENS_RAW="${EVAL_VLLM_MAX_MODEL_LENS:-}"
GPU_UTILS_RAW="${EVAL_VLLM_GPU_MEMORY_UTILIZATIONS:-}"

IFS=',' read -r -a PORTS <<< "${PORTS_RAW}"
IFS=';' read -r -a GPU_GROUPS <<< "${GPU_GROUPS_RAW}"

if [[ ${#PORTS[@]} -ne ${#GPU_GROUPS[@]} ]]; then
  echo "[ERROR] EVAL_VLLM_PORTS count (${#PORTS[@]}) must match EVAL_VLLM_GPU_GROUPS count (${#GPU_GROUPS[@]})." >&2
  exit 1
fi

# Build per-instance model/served/max-len arrays (fall back to the single value).
MODELS=(); SERVED_NAMES=(); MAX_LENS=()
if [[ -n "${MODEL_PATHS_RAW}" ]]; then
  IFS=',' read -r -a MODELS <<< "${MODEL_PATHS_RAW}"
  [[ ${#MODELS[@]} -eq ${#PORTS[@]} ]] || { echo "[ERROR] EVAL_VLLM_MODEL_PATHS count (${#MODELS[@]}) must match ports (${#PORTS[@]})." >&2; exit 1; }
else
  for _ in "${PORTS[@]}"; do MODELS+=("${MODEL_PATH}"); done
fi
if [[ -n "${SERVED_NAMES_RAW}" ]]; then
  IFS=',' read -r -a SERVED_NAMES <<< "${SERVED_NAMES_RAW}"
  [[ ${#SERVED_NAMES[@]} -eq ${#PORTS[@]} ]] || { echo "[ERROR] EVAL_VLLM_SERVED_MODEL_NAMES count (${#SERVED_NAMES[@]}) must match ports (${#PORTS[@]})." >&2; exit 1; }
else
  for _ in "${PORTS[@]}"; do SERVED_NAMES+=("${SERVED_MODEL_NAME}"); done
fi
if [[ -n "${MAX_MODEL_LENS_RAW}" ]]; then
  IFS=',' read -r -a MAX_LENS <<< "${MAX_MODEL_LENS_RAW}"
  [[ ${#MAX_LENS[@]} -eq ${#PORTS[@]} ]] || { echo "[ERROR] EVAL_VLLM_MAX_MODEL_LENS count (${#MAX_LENS[@]}) must match ports (${#PORTS[@]})." >&2; exit 1; }
else
  for _ in "${PORTS[@]}"; do MAX_LENS+=("${MAX_MODEL_LEN}"); done
fi
GPU_UTILS=()
if [[ -n "${GPU_UTILS_RAW:-}" ]]; then
  IFS=',' read -r -a GPU_UTILS <<< "${GPU_UTILS_RAW}"
  [[ ${#GPU_UTILS[@]} -eq ${#PORTS[@]} ]] || { echo "[ERROR] EVAL_VLLM_GPU_MEMORY_UTILIZATIONS count (${#GPU_UTILS[@]}) must match ports (${#PORTS[@]})." >&2; exit 1; }
else
  for _ in "${PORTS[@]}"; do GPU_UTILS+=("${GPU_MEMORY_UTILIZATION}"); done
fi
for _mp in "${MODELS[@]}"; do
  [[ -e "${_mp}" ]] || { echo "[ERROR] eval vLLM model path does not exist: ${_mp}" >&2; exit 1; }
done

TP_SIZES=()
if [[ -n "${TP_SIZES_RAW}" ]]; then
  IFS=',' read -r -a TP_SIZES <<< "${TP_SIZES_RAW}"
  if [[ ${#TP_SIZES[@]} -ne ${#PORTS[@]} ]]; then
    echo "[ERROR] EVAL_VLLM_TENSOR_PARALLEL_SIZES count (${#TP_SIZES[@]}) must match EVAL_VLLM_PORTS count (${#PORTS[@]})." >&2
    exit 1
  fi
fi

EXTRA_ARGS=()
if [[ -n "${EXTRA_ARGS_RAW}" ]]; then
  read -r -a EXTRA_ARGS <<< "${EXTRA_ARGS_RAW}"
fi

count_csv_items() {
  python - "$1" <<'PY'
import sys
items = [x.strip() for x in sys.argv[1].split(',') if x.strip()]
print(len(items))
PY
}

wait_for_vllm_endpoint() {
  local name="$1"
  local host="$2"
  local port="$3"
  local pid="$4"
  local attempt=1
  while (( attempt <= WAIT_RETRIES )); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] ${name} exited before becoming ready on ${host}:${port}" >&2
      return 1
    fi
    if curl -s -f --max-time 5 "http://${host}:${port}/v1/models" >/dev/null 2>&1; then
      echo "[INFO] ${name} is ready on ${host}:${port}"
      return 0
    fi
    sleep "$WAIT_INTERVAL_SECONDS"
    attempt=$((attempt + 1))
  done
  echo "[ERROR] ${name} did not become ready on ${host}:${port}" >&2
  return 1
}

PIDS=()
cleanup() {
  local exit_code=$?
  for pid in "${PIDS[@]:-}"; do
    if [[ -n "${pid}" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
    fi
  done
  wait 2>/dev/null || true
  return "$exit_code"
}
trap cleanup EXIT INT TERM

# Start mode: parallel (default) launches all instances at once, then waits for
# them together - total startup ~= slowest single load instead of the sum. This
# matters a lot for many-instance layouts (e.g. 4 x TP1), where serial startup
# meant ~4x the wait. Each instance is on its own GPU group + port so they don't
# conflict; the only cost is transient NAS read + host-RAM pressure from loading
# several model copies at once. Set EVAL_VLLM_PARALLEL_START=0 to go back to the
# conservative serial (start -> wait -> next) behavior.
PARALLEL_START="${EVAL_VLLM_PARALLEL_START:-1}"
STARTED_PORTS=()

#   idx0 61000-61199 … idx11 63200-63399
VLLM_PORT_BASE="${VLLM_PORT:-61000}"
VLLM_PORT_STRIDE="${VLLM_PORT_STRIDE:-200}"
echo "[INFO] VLLM_PORT windows: base=${VLLM_PORT_BASE} stride=${VLLM_PORT_STRIDE} -> idx0 ${VLLM_PORT_BASE}-$((VLLM_PORT_BASE+VLLM_PORT_STRIDE-1)) … idx$(( ${#PORTS[@]} - 1 )) $((VLLM_PORT_BASE+(${#PORTS[@]}-1)*VLLM_PORT_STRIDE))-$((VLLM_PORT_BASE+${#PORTS[@]}*VLLM_PORT_STRIDE-1))"

for idx in "${!PORTS[@]}"; do
  port="${PORTS[$idx]}"
  gpu_group="${GPU_GROUPS[$idx]}"
  if [[ -n "${TP_SIZES_RAW}" ]]; then
    tp_size="${TP_SIZES[$idx]}"
  else
    tp_size="$(count_csv_items "${gpu_group}")"
  fi
  if [[ -z "${port}" || -z "${gpu_group}" || -z "${tp_size}" ]]; then
    echo "[ERROR] Invalid eval vLLM configuration at index ${idx}: port='${port}' gpu_group='${gpu_group}' tp='${tp_size}'" >&2
    exit 1
  fi

  inst_model="${MODELS[$idx]}"
  inst_served="${SERVED_NAMES[$idx]}"
  inst_maxlen="${MAX_LENS[$idx]}"
  inst_util="${GPU_UTILS[$idx]}"
  start_retries="${EVAL_VLLM_START_RETRIES:-3}"
  attempt_i=1
  while :; do
    inst_vllm_port=$(( VLLM_PORT_BASE + idx * VLLM_PORT_STRIDE + (attempt_i - 1) * 60 ))
    echo "[INFO] Starting eval vLLM endpoint port=${port} gpus=${gpu_group} tp=${tp_size} model=${inst_model} served=${inst_served} max_len=${inst_maxlen} util=${inst_util} VLLM_PORT=${inst_vllm_port} (attempt ${attempt_i}/${start_retries})"
    CUDA_VISIBLE_DEVICES="${gpu_group}" \
    VLLM_PORT="${inst_vllm_port}" \
      vllm serve "${inst_model}" \
        --served-model-name "${inst_served}" \
        --host "${HOST}" \
        --port "${port}" \
        --tensor-parallel-size "${tp_size}" \
        --gpu-memory-utilization "${inst_util}" \
        --max-model-len "${inst_maxlen}" \
        "${EXTRA_ARGS[@]}" &
    pid=$!
    PIDS+=("$pid")
    if [[ "${PARALLEL_START}" == "1" ]]; then
      STARTED_PORTS+=("$port")
      break
    fi
    if wait_for_vllm_endpoint "eval_vllm_${port}" "127.0.0.1" "${port}" "$pid"; then
      STARTED_PORTS+=("$port")
      break
    fi
    attempt_i=$((attempt_i + 1))
    if (( attempt_i > start_retries )); then
      echo "[ERROR] eval_vllm_${port} failed to start ${start_retries} times, giving up" >&2
      exit 1
    fi
    echo "[WARN] eval_vllm_${port} failed to start, cleaning up and retrying (next ${attempt_i}/${start_retries})" >&2
    kill -9 "$pid" 2>/dev/null || true
    pkill -9 -f -- "--port ${port} --tensor-parallel-size" 2>/dev/null || true
    wait "$pid" 2>/dev/null || true
    sleep 20
  done
done

if [[ "${PARALLEL_START}" == "1" ]]; then
  # Parallel: all instances are already launching; now wait for each to be ready.
  echo "[INFO] Launched ${#STARTED_PORTS[@]} eval vLLM instances in parallel; waiting for all to become ready."
  for i in "${!STARTED_PORTS[@]}"; do
    wait_for_vllm_endpoint "eval_vllm_${STARTED_PORTS[$i]}" "127.0.0.1" "${STARTED_PORTS[$i]}" "${PIDS[$i]}"
  done
fi

echo "[INFO] All eval vLLM endpoints are ready for served model ${SERVED_MODEL_NAME}."

while true; do
  wait -n
  status=$?
  echo "[ERROR] An eval vLLM process exited with status ${status}." >&2
  exit "$status"
done
