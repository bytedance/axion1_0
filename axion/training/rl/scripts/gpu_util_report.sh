#!/usr/bin/env bash
# Axion relocation: canonical package/resource paths; training behavior unchanged.
# Aggregate the per-node GPU util logs written by the launcher's util monitor.
# Usage: bash gpu_util_report.sh [EXP_NAME] [LOG_ROOT]
#   defaults: EXP_NAME=quest4b-grpo, LOG_ROOT=<rl>/logs/quest4b_grpo_multinode_with_services
# Prints: overall average util, per-node average, and per-GPU average, plus the
# fraction of samples below the 45% kill threshold.
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
set -euo pipefail

EXP_NAME="${1:-quest4b-grpo}"
RL_ROOT="${RL_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)}"
LOG_ROOT="${2:-${RL_ROOT}/logs/quest4b_grpo_multinode_with_services}"
UTIL_DIR="${LOG_ROOT}/${EXP_NAME}/gpu_util"

if ! ls "${UTIL_DIR}"/node*.log >/dev/null 2>&1; then
  echo "[ERROR] No util logs found under ${UTIL_DIR}" >&2
  echo "        (Was the run launched with UTIL_MONITOR_ENABLED=1?)" >&2
  exit 1
fi

echo "=== GPU util report: ${UTIL_DIR} ==="
python - "${UTIL_DIR}" <<'PY'
import sys, glob, os, re
from collections import defaultdict
d = sys.argv[1]
# line: <ts> node=0 gpu=0 total=55 keeper=47 rl=8 mem_mb=1479
pat = re.compile(r'node=(\S+)\s+gpu=(\S+)\s+total=(\d+)\s+keeper=(\d+)\s+rl=(\d+)')
tot_by_node = defaultdict(list); rl_by_node = defaultdict(list)
rl_by_gpu = defaultdict(list); tot_all = []; keep_all = []; rl_all = []
lo_thresh = int(os.environ.get("UTIL_KILL_THRESHOLD", "45"))
for f in sorted(glob.glob(os.path.join(d, "node*.log"))):
    for line in open(f):
        if line.startswith("#"): continue
        m = pat.search(line)
        if not m: continue
        node, gpu, tot, keep, rl = m.group(1), m.group(2), int(m.group(3)), int(m.group(4)), int(m.group(5))
        tot_all.append(tot); keep_all.append(keep); rl_all.append(rl)
        tot_by_node[node].append(tot); rl_by_node[node].append(rl); rl_by_gpu[f"{node}:{gpu}"].append(rl)
if not tot_all:
    print("no samples parsed"); sys.exit()
avg = lambda xs: sum(xs)/len(xs)
below_total = sum(1 for v in tot_all if v < lo_thresh)/len(tot_all)*100
print(f"samples: {len(tot_all)}")
print(f"  TOTAL   avg util (what the platform sees): {avg(tot_all):.1f}%   (<{lo_thresh}%: {below_total:.0f}% of samples)")
print(f"  keeper  avg util (gpu_keeper baseline):    {avg(keep_all):.1f}%")
print(f"  >> RL   avg util (real training, keeper subtracted): {avg(rl_all):.1f}% <<")
print("per-node RL avg:")
for n in sorted(rl_by_node):
    print(f"   node {n}: rl={avg(rl_by_node[n]):.1f}%  total={avg(tot_by_node[n]):.1f}%  ({len(rl_by_node[n])} samples)")
print("per-gpu RL avg:")
for g in sorted(rl_by_gpu): print(f"   gpu {g}: rl={avg(rl_by_gpu[g]):.1f}%")
PY
