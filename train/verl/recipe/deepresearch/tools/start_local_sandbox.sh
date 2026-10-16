#!/bin/bash
#
#      /usr/bin/python3 -m pip install scipy pandas
#
#
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
for pid in $(ps -eo pid,cmd | grep "local_sandbox_serve[r].py" | awk '{print $1}'); do
  echo "[sandbox] stopping old instance pid=$pid"
  kill "$pid" 2>/dev/null
done
sleep 1

_port_busy() { (ss -tln 2>/dev/null || netstat -tln 2>/dev/null) | grep -qE "[:.]$1[[:space:]]"; }
_port_forbidden() {
  local p="$1"
  [[ "$p" -ge 20000 && "$p" -le 29999 ]] && return 0
  [[ "$p" -ge 6000  && "$p" -le 6003  ]] && return 0
  [[ "$p" == "8000" || "$p" == "8001" ]] && return 0            # search / scholar
  local v
  for v in $(env | grep -oE "^(ARNOLD|METIS)_WORKER_[0-9]+_PORT=[0-9]+" | cut -d= -f2); do
    [[ "$p" == "$v" ]] && return 0
  done
  return 1
}
PORT="${SANDBOX_PORT:-}"
if [[ -n "$PORT" ]]; then
  if _port_forbidden "$PORT"; then
    echo "[sandbox] SANDBOX_PORT=$PORT is in the reserved range (Ray / judge / search services); pick another or leave it unset" >&2; exit 1
  fi
  if _port_busy "$PORT"; then echo "[sandbox] SANDBOX_PORT=$PORT is already in use" >&2; exit 1; fi
else
  PORT=""
  for _cand in $(env | grep -oE "^ARNOLD_RESERVED_[0-9]+_CURRENT_PORT=[0-9]+" | cut -d= -f2 | sort -n) \
               $(seq 9100 9199); do
    _port_forbidden "$_cand" && continue
    _port_busy "$_cand" && continue
    PORT="$_cand"; break
  done
  if [[ -z "$PORT" ]]; then echo "[sandbox] no free port available" >&2; exit 1; fi
  echo "[sandbox] selected port $PORT (from this trial's reserved pool or the fallback range)"
fi
WORKERS="${WORKERS:-64}"
LOG="${SANDBOX_LOG:-/tmp/local_sandbox_${PORT}.log}"
SERVER_PY="${SERVER_PY:-/usr/bin/python3}"
export SANDBOX_PYTHON="${SANDBOX_PYTHON:-/usr/bin/python3}"
export SANDBOX_ISOLATE="${SANDBOX_ISOLATE:-1}"
if [[ -z "${SANDBOX_RO_ROOT:-}" ]]; then
  _ro_roots=""
  for _d in /mnt/bn/*/; do
    [[ -d "$_d" ]] || continue
    _ro_roots="${_ro_roots}${_ro_roots:+,}${_d%/}"
  done
  export SANDBOX_RO_ROOT="$_ro_roots"
fi
if [[ -z "$SANDBOX_RO_ROOT" ]]; then
  echo "[sandbox] no shared-storage mount detected, read-only isolation cannot be guaranteed;" >&2
  echo "[sandbox]   set SANDBOX_RO_ROOT explicitly before running (comma-separated for several)." >&2
  exit 1
fi
echo "[sandbox] read-only roots: $SANDBOX_RO_ROOT"
# The server accepts loopback only unless SANDBOX_ALLOW names wider networks.
# A single-node run needs nothing; a multi-node run has to opt in, because the
# other trainer nodes reach this service over the cluster network.
if [[ -n "${SANDBOX_ALLOW:-}" ]]; then
  echo "[sandbox] accepting code from: $SANDBOX_ALLOW"
  [[ "${SANDBOX_ALLOW,,}" == "any" ]] && \
    echo "[sandbox]   WARNING: the access check is off; only do this on an isolated network" >&2
else
  echo "[sandbox] accepting code from loopback only"
  echo "[sandbox]   a multi-node run must set SANDBOX_ALLOW to the CIDRs of its trainer nodes,"
  echo "[sandbox]   e.g. export SANDBOX_ALLOW=10.0.0.0/8 -- otherwise they get HTTP 403"
fi
export SANDBOX_MEM_LIMIT_MB="${SANDBOX_MEM_LIMIT_MB:-8192}"
_sbx_site="${SANDBOX_EXTRA_SITE:-}"
if [[ -z "${SANDBOX_EXTRA_SITE:-}" && -d "$_sbx_site" ]]; then
  export SANDBOX_EXTRA_SITE="$_sbx_site"
fi
[[ -n "${SANDBOX_EXTRA_SITE:-}" ]] && echo "[sandbox] extra site-packages: $SANDBOX_EXTRA_SITE"
unset _sbx_site

# Without SANDBOX_ALLOW the server answers only loopback, so that is the
# address worth advertising. Set both ADVERTISE_HOST and SANDBOX_ALLOW to put
# this sandbox on the cluster network.
ADVERTISE_HOST="${ADVERTISE_HOST:-}"
if [[ -z "$ADVERTISE_HOST" && -z "${SANDBOX_ALLOW:-}" ]]; then
  ADVERTISE_HOST="127.0.0.1"
fi
if [[ -z "$ADVERTISE_HOST" ]]; then
  _peer="${HEAD_NODE_IP:-${ARNOLD_WORKER_0_HOST:-}}"
  if [[ -n "$_peer" ]]; then
    ADVERTISE_HOST="$(ip route get "$_peer" 2>/dev/null | sed -n 's/.* src \([0-9a-f:.]*\).*/\1/p' | head -1)"
  else
    _dev="$(ip -4 route show default 2>/dev/null | sed -n 's/.* dev \([^ ]*\).*/\1/p' | head -1)"
    [[ -n "$_dev" ]] && ADVERTISE_HOST="$(ip -4 -o addr show dev "$_dev" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1)"
  fi
  [[ -z "$ADVERTISE_HOST" ]] && ADVERTISE_HOST="$(hostname -I 2>/dev/null | awk '{print $1}')"
  [[ -z "$ADVERTISE_HOST" ]] && ADVERTISE_HOST="127.0.0.1"
fi
case "$ADVERTISE_HOST" in *:*) ADVERTISE_HOST="[$ADVERTISE_HOST]";; esac
ENDPOINT="http://${ADVERTISE_HOST}:${PORT}/run_code"


echo "[sandbox] starting port=$PORT workers=$WORKERS exec-python=$SANDBOX_PYTHON isolate=$SANDBOX_ISOLATE"
setsid nohup "$SERVER_PY" "$HERE/local_sandbox_server.py" \
  --host "::" --port "$PORT" --workers "$WORKERS" > "$LOG" 2>&1 < /dev/null &

for _ in $(seq 1 30); do
  curl -sf --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null && break
  sleep 1
done
if ! curl -sf --noproxy '*' --max-time 2 "http://127.0.0.1:${PORT}/health" >/dev/null; then
  echo "[sandbox] failed to start, see $LOG" >&2; tail -20 "$LOG" >&2; exit 1
fi
if ! curl -sf --noproxy '*' --max-time 5 "http://${ADVERTISE_HOST}:${PORT}/health" >/dev/null 2>&1; then
  echo "[sandbox] advertised address http://${ADVERTISE_HOST}:${PORT} is unreachable (127.0.0.1 works)." >&2
  echo "[sandbox]   other nodes could not reach this sandbox; refusing to write the conf." >&2
  exit 1
fi
echo "[sandbox] advertised address reachable: http://${ADVERTISE_HOST}:${PORT}"

"$SERVER_PY" - "$PORT" "${SANDBOX_RO_ROOT%%,*}" <<'PY'
import json, sys, urllib.request
port = sys.argv[1]
ro_root = sys.argv[2]
_op = urllib.request.build_opener(urllib.request.ProxyHandler({}))
def run(code):
    req = urllib.request.Request(f"http://127.0.0.1:{port}/run_code",
        data=json.dumps({"code": code, "language": "python", "run_timeout": 20}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(_op.open(req, timeout=60))["run_result"]
a = run("import sympy;print(sympy.isprime(2**61-1))")
c = run("import scipy,pandas;print('EXTRA_OK',scipy.__version__,pandas.__version__)")
b = run("open(%r,'w').write('x');print('WRITABLE')" % (ro_root + "/__sbx_probe"))
ok = a["status"] == "Finished" and a["stdout"].strip() == "True"
iso_broken = any(x.get("return_code") == 97 for x in (a, b, c))
ro = (b["status"] == "Error") and b.get("return_code") != 97
extra = "EXTRA_OK" in (c.get("stdout") or "")
if iso_broken:
    print("[sandbox] smoke test: isolation setup failed (return_code=97); the checks below are meaningless")
    print("[sandbox]   usually SANDBOX_RO_ROOT points at a mount that does not exist on this host.")
    print("[sandbox]   stderr: " + (a.get("stderr") or b.get("stderr") or "")[:200])
    sys.exit(1)
print(f"[sandbox] smoke compute={'OK' if ok else 'FAIL'} storage-read-only={'OK' if ro else 'FAIL (unsafe)'} "
      f"scipy/pandas={'OK' if extra else 'missing (not fatal)'}")
if not ok:
    print("[sandbox]   compute failure stderr: " + (a.get("stderr") or "")[:300])
sys.exit(0 if (ok and ro) else 1)
PY
[[ $? -ne 0 ]] && { echo "[sandbox] smoke test failed" >&2; exit 1; }

mkdir -p "$(dirname "$CONF")"
#
_conf_tmp="$(mktemp "${CONF}.tmp.XXXXXX")"
printf 'endpoint=%s\n' "$ENDPOINT" > "$_conf_tmp"
mv -f "$_conf_tmp" "$CONF"
unset _conf_tmp
echo "[sandbox] ready  $ENDPOINT"
echo "[sandbox]   wrote $CONF"
echo "[sandbox]   remember to set: export PYTHON_SANDBOX_BACKEND=sandbox_fusion"
echo "[sandbox]   log $LOG    health check: curl http://127.0.0.1:${PORT}/health"
if [[ -z "${SANDBOX_ALLOW:-}" ]]; then
  echo "[sandbox]   note: $ENDPOINT answers 403 to every peer except loopback"
fi
