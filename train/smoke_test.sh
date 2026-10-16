#!/usr/bin/env bash
# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

# Smoke test for the RL stack. Checks what can be checked without GPUs:
# the recipes parse, the launcher parses, every Python module imports, and the
# code-execution sandbox actually runs code with isolation in place.
#
#   bash train/smoke_test.sh
#
# Exit code 0 means everything passed.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DR="$HERE/verl/recipe/deepresearch"
PORT="${SMOKE_PORT:-19321}"
fail=0
ok()   { printf '  ok    %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }

echo "== 1. shell syntax"
for f in "$HERE"/rl/rl_*.sh "$DR"/run_deepresearch_fully_async_megatron.sh \
         "$DR"/tools/*.sh "$DR"/scripts/*.sh; do
  [ -e "$f" ] || continue
  bash -n "$f" 2>/dev/null && ok "$(basename "$f")" || bad "$(basename "$f")"
done

echo "== 2. python syntax"
n=0; b=0
while IFS= read -r f; do
  n=$((n+1)); python3 -m py_compile "$f" 2>/dev/null || { b=$((b+1)); echo "  FAIL  ${f#$DR/}"; }
done < <(find "$DR" -name '*.py')
[ "$b" -eq 0 ] && ok "$n files compile" || fail=1

echo "== 3. yaml parses"
for f in "$DR"/config/*.yaml; do
  [ -e "$f" ] || continue
  python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$f" 2>/dev/null \
    && ok "$(basename "$f")" || bad "$(basename "$f")"
done

echo "== 4. recipes refuse to run without their required inputs"
for s in 122b 35b; do
  r="$HERE/rl/rl_$s.sh"; [ -e "$r" ] || continue
  # Capture first: the shell exits non-zero here by design, and with pipefail a
  # pipeline straight into grep would report that as a failure of the check.
  out=$( env -u MODEL_PATH -u DATASET -u RAY_DATA_HOME \
           bash -c "source '$r'" 2>&1 )
  case "$out" in
    *MODEL_PATH*) ok  "rl_$s.sh rejects a missing MODEL_PATH" ;;
    *)            bad "rl_$s.sh accepted a missing MODEL_PATH" ;;
  esac
done

echo "== 5. recipes export a topology when inputs are present"
for s in 122b 35b; do
  r="$HERE/rl/rl_$s.sh"; [ -e "$r" ] || continue
  out=$( MODEL_PATH=/tmp/x DATASET=/tmp/x.parquet RAY_DATA_HOME=/tmp/x \
         SERPER_KEY_ID=x JINA_API_KEYS=x SUMMARY_MODEL_NAME=x \
         bash -c "source '$r'" 2>&1 | tail -1 )
  case "$out" in *topology*) ok "rl_$s.sh -> $out" ;; *) bad "rl_$s.sh: $out" ;; esac
done

echo "== 6. SFT config and data scripts"
if [ -e "$HERE/sft/sft_stage1.yaml" ]; then
  for y in sft_stage1 sft_stage2; do
    python3 -c "import yaml,sys; yaml.safe_load(open(sys.argv[1]))" "$HERE/sft/$y.yaml" 2>/dev/null \
      && ok "$y.yaml parses" || bad "$y.yaml parses"
  done
  # The patch that keeps sequence_parallel from failing on an odd-length batch.
  grep -q '_pad_to_tp_multiple_wrapper' \
    "$HERE/sft/LlamaFactory/src/llamafactory/train/mca/workflow.py" 2>/dev/null \
    && ok "LlamaFactory carries the padding patch" \
    || bad "LlamaFactory is missing the padding patch"
  python3 "$HERE/sft/check_data.py" --help >/dev/null 2>&1 \
    && ok "check_data.py --help" || bad "check_data.py --help"
  # The dataset names the configs use must exist in dataset_info.json.
  python3 - "$HERE/sft" <<'PY' && ok "configs and dataset_info.json agree" || bad "configs and dataset_info.json agree"
import json, sys, yaml, pathlib
d = pathlib.Path(sys.argv[1])
registered = set(json.load(open(d / "data" / "dataset_info.json")))
for cfg in sorted(d.glob("sft_stage*.yaml")):
    for name in str(yaml.safe_load(open(cfg))["dataset"]).split(","):
        if name.strip() not in registered:
            sys.exit(f"{cfg.name}: dataset {name.strip()!r} is not in dataset_info.json")
PY
  python3 -c "import json,sys; json.load(open(sys.argv[1]))" "$HERE/sft/data/dataset_info.json" 2>/dev/null \
    && ok "dataset_info.json parses" || bad "dataset_info.json parses"
fi

echo "== 7. documented commands match their CLIs"
if out=$( cd "$HERE/.." && python3 train/check_readme_commands.py 2>&1 ); then
  ok "$(echo "$out" | tail -1)"
else
  echo "$out" | sed 's/^/  /'
  bad "README commands do not match the scripts"
fi

echo "== 8. lint"
if command -v ruff >/dev/null 2>&1; then
  ( cd "$HERE/.." && ruff check . >/dev/null 2>&1 ) \
    && ok "ruff check" || bad "ruff check (run: ruff check .)"
else
  echo "  skip  ruff is not installed"
fi

echo "== 9. sandbox runs code"
srv="$DR/tools/local_sandbox_server.py"
if [ ! -e "$srv" ]; then bad "local_sandbox_server.py missing"; else
  python3 "$srv" --port "$PORT" --workers 4 >/tmp/smoke_sandbox.log 2>&1 &
  pid=$!
  # --noproxy: the sandbox is on the loopback interface, but a machine with
  # http_proxy or ALL_PROXY set sends even 127.0.0.1 through the proxy, which
  # answers with an empty body and exit status 0 -- every check below then
  # fails on a machine whose sandbox is in fact fine.
  for _ in $(seq 1 20); do
    curl -s -f --noproxy '*' --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && break
    sleep 1
  done
  run() {   # run() <code> -> "<status> <run_status> <stdout>"
    curl -s --noproxy '*' --max-time 40 -X POST "http://127.0.0.1:$PORT/run_code" \
      -H 'Content-Type: application/json' \
      -d "$(python3 -c 'import json,sys; print(json.dumps({"code":sys.argv[1],"language":"python","run_timeout":8.0,"compile_timeout":10,"stdin":None,"files":{},"fetch_files":[]}))' "$1")" \
      | python3 -c 'import json,sys
d=json.load(sys.stdin); r=d.get("run_result") or {}
print(d.get("status"), r.get("status"), (r.get("stdout") or "").strip())'
  }
  check() {  # check <label> <code> <expected prefix>
    got="$(run "$2")"
    case "$got" in "$3"*) ok "$1" ;; *) bad "$1 (got: $got)" ;; esac
  }
  check "computes"                 'print(2**100)'                  'Success Finished 1267650600228229401496703205376'
  check "sympy available"          'import sympy; print(sympy.factorint(1234567))' 'Success Finished {127: 1, 9721: 1}'
  check "user error is not a sandbox error" '1/0'                   'Success Error'
  check "enforces the timeout"     'while True: pass'               'Success TimeLimitExceeded'
  check "shared storage read-only" "open('/mnt/x','w').write('x'); print('WROTE')" 'Success Error'
  check "network cut off"          "import socket; socket.create_connection(('1.1.1.1',53),2); print('NET')" 'Success Error'
  kill "$pid" 2>/dev/null
  wait "$pid" 2>/dev/null
fi

echo
if [ "$fail" -eq 0 ]; then echo "all checks passed"; else echo "some checks FAILED"; fi
exit "$fail"
