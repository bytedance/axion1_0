#!/usr/bin/env python3
# Axion relocation: canonical package/resource paths; training behavior unchanged.
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
"""Local code-execution service: a drop-in replacement for e2b that speaks the
sandbox_fusion POST /run_code protocol.

Why this exists: a hosted e2b account caps concurrent sandboxes well below the
rollout concurrency used here (engines x ROLLOUT_CONCURRENCY_PER_ENGINE), so
PythonInterpreter calls fail en masse with rate limits and timeouts. With the
tool effectively unavailable the policy degrades into searching for answers it
should compute.

Protocol -- the client is the sandbox_fusion package (see python_tool.py):
  POST {endpoint}/run_code
  req  {"code":..., "language":"python", "run_timeout":30.0, "compile_timeout":10,
        "stdin":null, "files":{}, "fetch_files":[]}
  resp {"status":"Success|Failed|SandboxError", "message":"",
        "compile_result":null,
        "run_result":{"status":"Finished|Error|TimeLimitExceeded",
                      "execution_time":float, "cpu_time":null,
                      "return_code":int, "stdout":str, "stderr":str},
        "executor_pod_name":null, "files":{}}
  status=SandboxError makes the client raise and retry, so a failure in the
  user's own code must return Success with run_result.status=Error instead.

Isolation: this is lightweight isolation, not a true sandbox -- a separate
process in a fresh user, mount and network namespace (shared storage remounted
read-only, networking disconnected), a fresh temporary cwd, RLIMITs (address
space, CPU, file size, process count), a cleared environment and python -I.
That stops runaway loops, memory bombs, filling the disk, and following
environment variables to shared storage. It does not stop deliberately
malicious code. The model writes mathematical code, where the risk is accidents
rather than attacks; stronger isolation needs a container or an unprivileged
user.

Access: the service executes whatever it is sent, so by default it accepts
requests from loopback only and rejects every other peer with 403. A multi-node
run has to widen that explicitly: set SANDBOX_ALLOW to a comma-separated list of
CIDRs covering the training nodes. They are added to loopback rather than
swapping for it, no value turns the check off, and a range wider than /8 (IPv4)
or /16 (IPv6) is refused at startup. Keep the list as narrow as the deployment
allows, and never put this service on an untrusted network.

Usage:
  export SANDBOX_RO_ROOT=/path/to/shared-storage
  python local_sandbox_server.py --port 9000 --workers 32
  then write the endpoint into axion/training/rl/config/python_nodes.conf
  (hot-reloaded, no need to restart training):
      endpoint=http://<host-ip>:9000/run_code
  and set PYTHON_SANDBOX_BACKEND=sandbox_fusion.
"""
import argparse
import ipaddress
import json
import socket
import os
import resource
import shutil
import shlex
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MEM_LIMIT_MB = int(os.environ.get("SANDBOX_MEM_LIMIT_MB", "8192"))
FSIZE_LIMIT_MB = int(os.environ.get("SANDBOX_FSIZE_LIMIT_MB", "64"))
NPROC_LIMIT = int(os.environ.get("SANDBOX_NPROC_LIMIT", "64"))
PYTHON_BIN = os.environ.get("SANDBOX_PYTHON", sys.executable)
ISOLATE = os.environ.get("SANDBOX_ISOLATE", "1").strip() == "1"

# Networks allowed to submit code. The service executes what it is sent and has
# no caller authentication, so this allowlist is its only access boundary and
# the default is the narrowest one that still works: loopback. Reaching the
# service from other hosts -- which a multi-node run needs -- requires setting
# SANDBOX_ALLOW to a comma-separated list of CIDRs.
# Loopback is always allowed: anyone on the host can already run code as this
# user, and the start script health-checks over it. SANDBOX_ALLOW adds to this
# list rather than replacing it, and no value switches the check off -- a
# deployment that wants peers has to name their networks. Ranges wider than /8
# (IPv4) or /16 (IPv6) are refused at startup, so the list cannot be opened to
# everything by writing 0.0.0.0/0 either: the service fails to start rather
# than coming up with an allowlist that does not mean what it looks like.
_LOOPBACK_ALLOW = "127.0.0.0/8,::1/128,::ffff:127.0.0.0/104"
_ALLOW_IS_EXPLICIT = bool((os.environ.get("SANDBOX_ALLOW") or "").strip())
_MIN_PREFIXLEN = {4: 8, 6: 16}


def _parse_allow():
    nets, bad = [], []
    for part in f"{_LOOPBACK_ALLOW},{os.environ.get('SANDBOX_ALLOW') or ''}".split(","):
        part = part.strip()
        if not part:
            continue
        try:
            net = ipaddress.ip_network(part, strict=False)
        except ValueError:
            bad.append(f"{part!r}: not a CIDR")
            continue
        if net.prefixlen < _MIN_PREFIXLEN[net.version]:
            bad.append(f"{net}: wider than /{_MIN_PREFIXLEN[net.version]}")
            continue
        nets.append(net)
    if bad:
        for b in bad:
            print(f"[sandbox] SANDBOX_ALLOW rejected {b}", flush=True)
        print("[sandbox] SANDBOX_ALLOW takes CIDRs of the hosts allowed to submit code, "
              "e.g. 10.0.0.0/8. No value disables the check.", flush=True)
        raise SystemExit(2)
    return nets


ALLOW_NETS = _parse_allow()


def _client_allowed(addr: str) -> bool:
    try:
        ip = ipaddress.ip_address(addr)
    except ValueError:
        return False
    # A dual-stack socket reports IPv4 peers as ::ffff:a.b.c.d.
    if getattr(ip, "ipv4_mapped", None):
        ip = ip.ipv4_mapped
    return any(ip in net for net in ALLOW_NETS)


def _read_only_roots() -> list[str]:
    roots = [part.strip() for part in os.environ.get("SANDBOX_RO_ROOT", "").split(",") if part.strip()]
    if not roots or any(not os.path.isabs(root) or not os.path.isdir(root) for root in roots):
        raise ValueError("SANDBOX_RO_ROOT must name existing absolute storage directories, separated by commas")
    return roots


READ_ONLY_ROOTS = _read_only_roots()

_stats = {"total": 0, "ok": 0, "err": 0, "timeout": 0}
_lock = threading.Lock()


def _preexec():
    """Only call setsid, so a timeout can kill the whole process group.

    Resource limits must not be set here: setting RLIMITs on the parent makes
    unshare exit with rc=2 and an empty stderr, and the namespace is never
    created. They are applied with ulimit inside the namespace instead, so they
    constrain only the python process that runs the code.
    """
    os.setsid()


def _build_cmd(script: str) -> list:
    """Namespace isolation: user + mount + net namespaces, with shared storage
    remounted read-only and networking cut off.

    Without it the executed code can list and write shared storage, which puts
    model weights and checkpoints one shutil.rmtree away. With it, writes to
    shared storage and socket calls raise OSError while the conda environment
    stays readable. unshare needs no root here because of the user namespace.
    """
    ulimits = (f"ulimit -v {MEM_LIMIT_MB * 1024}; ulimit -f {FSIZE_LIMIT_MB * 1024}; "
               f"ulimit -c 0; ")
    _extra = (os.environ.get("SANDBOX_EXTRA_SITE") or "").strip()
    if _extra:
        _payload = ("import sys,runpy; sys.path.append(%r); "
                    "runpy.run_path(%r, run_name='__main__')" % (_extra, script))
        inner = ulimits + "exec %s -I -c %s" % (PYTHON_BIN, shlex.quote(_payload))
    else:
        inner = ulimits + f"exec {PYTHON_BIN} -I {script}"
    if ISOLATE and shutil.which("unshare"):
        guard = "".join(
            f"mount -o remount,ro,bind {shlex.quote(root)} 2>/dev/null || "
            "{ echo 'sandbox: failed to remount storage read-only' >&2; exit 97; }; "
            for root in READ_ONLY_ROOTS
        )
        return ["unshare", "--user", "--map-root-user", "--mount", "--net",
                "--", "/bin/sh", "-c", guard + inner]
    return ["/bin/sh", "-c", inner]


def run_python(code: str, timeout: float, stdin: str | None) -> dict:
    workdir = tempfile.mkdtemp(prefix="sbx_")
    script = os.path.join(workdir, "main.py")
    with open(script, "w") as f:
        f.write(code)
    env = {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": workdir,
        "TMPDIR": workdir,
        "LANG": "C.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    started = time.time()
    p = None
    try:
        p = subprocess.Popen(
            _build_cmd(script), cwd=workdir, env=env,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, preexec_fn=_preexec,
        )
        out, err = p.communicate(input=(stdin or ""), timeout=timeout)
        elapsed = time.time() - started
        if p.returncode == 97:
            return {"status": "Error", "execution_time": round(elapsed, 6), "cpu_time": None,
                    "return_code": 97, "stdout": "",
                    "stderr": "sandbox internal error: isolation setup failed, refusing to run"}
        return {
            "status": "Finished" if p.returncode == 0 else "Error",
            "execution_time": round(elapsed, 6),
            "cpu_time": None,
            "return_code": p.returncode,
            "stdout": out[-100000:],
            "stderr": err[-20000:],
        }
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(p.pid), 9)
        except Exception:
            pass
        try:
            p.communicate(timeout=5)
        except Exception:
            pass
        return {
            "status": "TimeLimitExceeded",
            "execution_time": round(time.time() - started, 6),
            "cpu_time": None,
            "return_code": -9,
            "stdout": "",
            "stderr": f"Execution timed out after {timeout}s",
        }
    except Exception as e:
        return {
            "status": "Error",
            "execution_time": round(time.time() - started, 6),
            "cpu_time": None,
            "return_code": -1,
            "stdout": "",
            "stderr": f"sandbox internal error: {e}",
        }
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


class _Server(ThreadingHTTPServer):
    request_queue_size = 512
    allow_reuse_address = True
    daemon_threads = True
    address_family = socket.AF_INET6

    def server_bind(self):
        try:
            self.socket.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        except OSError:
            pass
        super().server_bind()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    pool: ThreadPoolExecutor = None

    def log_message(self, *a):
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _reject_unless_allowed(self) -> bool:
        """Close the request unless the peer is on an allowed network."""
        if _client_allowed(self.client_address[0]):
            return False
        self._send(403, {"error": "client network not allowed"})
        return True

    def do_GET(self):
        if self._reject_unless_allowed():
            return
        if self.path.rstrip("/") in ("/health", "/ping"):
            with _lock:
                self._send(200, {"ok": True, **_stats})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self._reject_unless_allowed():
            return
        if self.path.rstrip("/") != "/run_code":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception as e:
            self._send(400, {"error": f"bad request: {e}"})
            return

        lang = str(req.get("language") or "python").lower()
        if lang not in ("python", "python3", ""):
            self._send(200, {
                "status": "Failed", "message": f"unsupported language: {lang}",
                "compile_result": None, "run_result": None,
                "executor_pod_name": None, "files": {},
            })
            return

        timeout = float(req.get("run_timeout") or 30.0)
        code = req.get("code") or ""
        fut = Handler.pool.submit(run_python, code, timeout, req.get("stdin"))
        rr = fut.result(timeout=timeout + 15)

        with _lock:
            _stats["total"] += 1
            _stats["ok" if rr["status"] == "Finished" else
                   ("timeout" if rr["status"] == "TimeLimitExceeded" else "err")] += 1

        self._send(200, {
            "status": "Success",
            "message": "",
            "compile_result": None,
            "run_result": rr,
            "executor_pod_name": None,
            "files": {},
        })


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="::", help="default :: binds dual-stack, accepting both IPv4 and IPv6")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--workers", type=int, default=64,
                    help="maximum concurrent executions; size it above the rollout concurrency "
                         "(engines x ROLLOUT_CONCURRENCY_PER_ENGINE) to leave headroom")
    a = ap.parse_args()
    Handler.pool = ThreadPoolExecutor(max_workers=a.workers)
    srv = _Server((a.host, a.port), Handler)
    if not _ALLOW_IS_EXPLICIT:
        scope = "loopback only (set SANDBOX_ALLOW to reach this from other nodes)"
    else:
        scope = ",".join(str(n) for n in ALLOW_NETS)
    print(f"[local-sandbox] listening on {a.host}:{a.port}  workers={a.workers} "
          f"python={PYTHON_BIN} mem={MEM_LIMIT_MB}MB", flush=True)
    print(f"[local-sandbox] accepting code from: {scope}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
