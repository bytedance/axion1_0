#!/bin/bash
#
#
# Modified by the Axion authors for package paths, deployment settings and runtime fixes.
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
print("import errno, os, tempfile\n"
      "for root in %r:\n"
      "    try:\n"
      "        fd, path = tempfile.mkstemp(prefix='.axion_probe_', dir=root)\n"
      "    except OSError as exc:\n"
      "        if exc.errno not in (errno.EROFS, errno.EACCES, errno.EPERM): raise\n"
      "    else:\n"
      "        os.close(fd); os.unlink(path)\n"
      "        raise RuntimeError('storage is writable: ' + root)\n"
      "print('READ_ONLY')" % roots)
PY_PROBE
); then exit 1; fi

mapfile -t EPS < <(sed 's/#.*//' "$CONF" 2>/dev/null | grep -oE "https?://[^ ,;]+" | sed 's#/run_code/*$##' | grep -v '<ip>' | sort -u)
if [[ ${#EPS[@]} -eq 0 ]]; then echo "no endpoint found in $CONF" >&2; exit 1; fi
echo "node $(hostname) | conf: $CONF | ${#EPS[@]} endpoints"

post() {  # $1=endpoint $2=code $3=timeout
  curl -sf --noproxy '*' --max-time "${3:-25}" -X POST "$1/run_code" -H 'Content-Type: application/json' \
    -d "$(printf '{"code":%s,"language":"python","run_timeout":%s}' "$(printf '%s' "$2" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" "${4:-20}")"
}
field() { python3 -c "import json,sys;d=json.load(sys.stdin).get('run_result') or {};print(d.get('$1',''))" 2>/dev/null; }

NETWORK_CODE="$(cat <<'PY_NETWORK'
import errno, socket
try:
    sock = socket.create_connection(('1.1.1.1', 53), timeout=3)
except OSError as exc:
    if exc.errno not in (errno.ENETUNREACH, errno.EACCES, errno.EPERM): raise
    print('NETWORK_BLOCKED')
else:
    sock.close()
    raise RuntimeError('network is reachable')
PY_NETWORK
)"

FAILED=0
DEAD=()
for ep in "${EPS[@]}"; do
  echo; echo "── $ep"
  out=$(post "$ep" 'import sympy;print(sympy.isprime(2**61-1))')
  st=$(printf '%s' "$out" | field status); so=$(printf '%s' "$out" | field stdout)
  if [[ "$st" != "Finished" || "${so// /}" != "True" ]]; then
    echo "  unreachable or wrong result (status=${st:-no response})"; DEAD+=("$ep"); FAILED=1; continue
  fi
  echo "  reachable, compute correct"
  w=$(post "$ep" "$PROBE_CODE")
  n=$(post "$ep" "$NETWORK_CODE")
  if [[ "$(printf '%s' "$w" | field status)" == Finished && "$(printf '%s' "$w" | field stdout)" == READ_ONLY ]]; then
    echo "  storage read-only"
  else
    echo "  storage read-only check FAILED"; FAILED=1
  fi
  if [[ "$(printf '%s' "$n" | field status)" == Finished && "$(printf '%s' "$n" | field stdout)" == NETWORK_BLOCKED ]]; then
    echo "  network cut off"
  else
    echo "  network isolation check FAILED"; FAILED=1
  fi
  t=$(post "$ep" 'while True: pass' 30 6 | field status)
  if [[ "$t" == "TimeLimitExceeded" ]]; then
    echo "  timeout enforced"
  else
    echo "  timeout NOT enforced (status=$t)"; FAILED=1
  fi
  if ! python3 - "$ep" <<'PY'
import json,sys,time,urllib.request
from concurrent.futures import ThreadPoolExecutor
ep=sys.argv[1]; N=48
opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
def one(_):
    r=urllib.request.Request(ep+"/run_code",data=json.dumps({"code":"import time;time.sleep(2);print('x')","language":"python","run_timeout":30}).encode(),headers={"Content-Type":"application/json"})
    try:
        rr=json.load(opener.open(r,timeout=120))["run_result"]
        return rr["status"]=="Finished" and rr["stdout"].strip()=="x"
    except Exception: return False
t0=time.time()
with ThreadPoolExecutor(max_workers=N) as ex: ok=sum(ex.map(one,range(N)))
w=time.time()-t0
print(f"  {'ok' if ok==N else 'FAIL'} concurrency {N}: {ok}/{N} succeeded, wall clock {w:.1f}s"
      f"{' (single wave, enough capacity)' if w<4 else ' (several waves, too few workers)'}")
sys.exit(0 if ok==N else 1)
PY
  then FAILED=1; fi
done

echo
if [[ ${#DEAD[@]} -gt 0 ]]; then
  echo "unreachable from this node: ${#DEAD[@]} endpoints: ${DEAD[*]}"
  echo "   the client shuffles and tries them in turn, so dead addresses waste retries."
  if [[ $PRUNE -eq 1 ]]; then
    python3 - "$CONF" "${DEAD[@]}" <<'PY_PRUNE'
from pathlib import Path
import re, sys
path=Path(sys.argv[1]); dead=set(sys.argv[2:]); lines=[]
for line in path.read_text().splitlines(keepends=True):
    body, mark, comment=line.partition('#')
    urls=re.findall(r'https?://[^\s,;]+', body)
    remove=[url for url in urls if url.rstrip('/').removesuffix('/run_code') in dead]
    if not remove:
        lines.append(line); continue
    kept=[url for url in urls if url not in remove]
    if kept:
        key=body.split('=',1)[0]+'=' if '=' in body else ''
        lines.append(key+','.join(kept)+(' #'+comment.rstrip('\n') if mark else '')+'\n')
    elif mark:
        lines.append('#'+comment.rstrip('\n')+'\n')
path.write_text(''.join(lines))
PY_PRUNE
    [[ $? -eq 0 ]] || FAILED=1
    echo "   removed from $CONF"
  else
    echo "   pass --prune to remove them automatically"
  fi
else
  echo "all endpoints reachable from this node"
fi
exit "$FAILED"
