#!/bin/bash
#
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF="${PYTHON_NODES_CONF:-$HERE/../config/python_nodes.conf}"
PRUNE=0; [[ "${1:-}" == "--prune" ]] && PRUNE=1

# Use the explicitly configured server-side root, including paths with quotes.
if ! PROBE_CODE=$(python3 - <<'PY_PROBE'
import os
roots = [part.strip() for part in os.environ.get('SANDBOX_RO_ROOT', '').split(',') if part.strip()]
if not roots or any(not os.path.isabs(root) for root in roots):
    raise SystemExit('SANDBOX_RO_ROOT must name absolute server-side storage directories')
print("open(%r,'w').write('x');print('WRITABLE')" % (roots[0].rstrip('/') + '/__probe'))
PY_PROBE
); then exit 1; fi

mapfile -t EPS < <(grep -oE "https?://[^ ]+" "$CONF" 2>/dev/null | sed 's#/run_code$##' | grep -v '<ip>' | sort -u)
if [[ ${#EPS[@]} -eq 0 ]]; then echo "no endpoint found in $CONF" >&2; exit 1; fi
echo "node $(hostname) | conf: $CONF | ${#EPS[@]} endpoints"

post() {  # $1=endpoint $2=code $3=timeout
  curl -s --max-time "${3:-25}" -X POST "$1/run_code" -H 'Content-Type: application/json' \
    -d "$(printf '{"code":%s,"language":"python","run_timeout":%s}' "$(printf '%s' "$2" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" "${4:-20}")"
}
field() { python3 -c "import json,sys;d=json.load(sys.stdin).get('run_result') or {};print(d.get('$1',''))" 2>/dev/null; }

DEAD=()
for ep in "${EPS[@]}"; do
  echo; echo "── $ep"
  out=$(post "$ep" 'import sympy;print(sympy.isprime(2**61-1))')
  st=$(printf '%s' "$out" | field status); so=$(printf '%s' "$out" | field stdout)
  if [[ "$st" != "Finished" || "${so// /}" != "True" ]]; then
    echo "  unreachable or wrong result (status=${st:-no response})"; DEAD+=("$ep"); continue
  fi
  echo "  reachable, compute correct"
  w=$(post "$ep" "$PROBE_CODE" | field status)
  n=$(post "$ep" "import socket;socket.create_connection(('1.1.1.1',53),timeout=3);print('NET')" | field status)
  [[ "$w" == "Error" ]] && echo "  storage read-only" || echo "  storage WRITABLE -- unsafe, do not use this endpoint"
  [[ "$n" == "Error" ]] && echo "  network cut off" || echo "  network NOT isolated"
  t=$(post "$ep" 'while True: pass' 30 6 | field status)
  [[ "$t" == "TimeLimitExceeded" ]] && echo "  timeout enforced" || echo "  timeout NOT enforced (status=$t)"
  python3 - "$ep" <<'PY'
import json,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
ep=sys.argv[1]; N=48
def one(_):
    r=urllib.request.Request(ep+"/run_code",data=json.dumps({"code":"import time;time.sleep(2);print('x')","language":"python","run_timeout":30}).encode(),headers={"Content-Type":"application/json"})
    try:
        return json.load(urllib.request.urlopen(r,timeout=120))["run_result"]["status"]=="Finished"
    except Exception: return False
t0=time.time()
with ThreadPoolExecutor(max_workers=N) as ex: ok=sum(ex.map(one,range(N)))
w=time.time()-t0
print(f"  {'ok' if ok==N else 'FAIL'} concurrency {N}: {ok}/{N} succeeded, wall clock {w:.1f}s"
      f"{' (single wave, enough capacity)' if w<4 else ' (several waves, too few workers)'}")
PY
done

echo
if [[ ${#DEAD[@]} -gt 0 ]]; then
  echo "unreachable from this node: ${#DEAD[@]} endpoints: ${DEAD[*]}"
  echo "   the client shuffles and tries them in turn, so dead addresses waste retries."
  if [[ $PRUNE -eq 1 ]]; then
    for d in "${DEAD[@]}"; do sed -i "\#${d}#d" "$CONF"; done
    echo "   removed from $CONF"
  else
    echo "   pass --prune to remove them automatically"
  fi
else
  echo "all endpoints reachable from this node"
fi
