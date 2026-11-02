#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
"""
Synthetic GPU load generator — for CONTENTION / STRESS testing.

Runs a synthetic load with a controllable utilisation target next to a real
job, to measure how much that job slows down under contention. This is not a
training script and does not pretend to be one: it is a plain GEMM stress load
whose logs report both the target and the measured utilisation.

Tunables (env):
  LOADGEN_GPUS         comma list of GPU ids (default "0,1,2,3,4,5,6,7")
  LOADGEN_TARGET_UTIL  target utilisation %% (default 35); closed-loop mode converges to it.
  LOADGEN_MODE         "closed" (default, feeds back the measured utilisation
                       via pynvml) or "duty" (open-loop duty cycle, no pynvml)
  LOADGEN_MEM_GB       per-process memory cap in GB (default 4.0)
  LOADGEN_MATRIX       square matrix side N; the load is an N x N matmul (default 4096)
  LOADGEN_DTYPE        "bf16" (default) or "tf32"
  LOADGEN_LOG_EVERY_S  logging interval in seconds (default 5)
"""
import os
import time
import multiprocessing as mp


def _gpu_ids():
    raw = os.getenv("LOADGEN_GPUS", "0,1,2,3,4,5,6,7")
    ids = [int(c) for c in raw.split(",") if c.strip() != ""]
    if not ids:
        raise ValueError("LOADGEN_GPUS must list at least one GPU id")
    return ids


def _read_util(gpu_id):
    """Instantaneous utilisation %% for the GPU, or None when unavailable."""
    try:
        import pynvml
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        u = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
        return float(u)
    except Exception:
        return None


def worker(gpu_id: int):
    import torch

    torch.cuda.set_device(gpu_id)
    device = f"cuda:{gpu_id}"

    dtype_env = os.getenv("LOADGEN_DTYPE", "bf16").lower()
    if dtype_env == "tf32":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        mm_dtype = torch.float32
    else:
        mm_dtype = torch.bfloat16

    mem_gb = float(os.getenv("LOADGEN_MEM_GB", "4.0"))
    try:
        total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)
        frac = max(0.005, min(0.5, mem_gb / total_gb))
        torch.cuda.set_per_process_memory_fraction(frac, gpu_id)
    except Exception as e:
        print(f"[loadgen] gpu{gpu_id}: memory fraction skip: {e}", flush=True)

    n = int(os.getenv("LOADGEN_MATRIX", "4096"))
    target = float(os.getenv("LOADGEN_TARGET_UTIL", "35"))
    mode = os.getenv("LOADGEN_MODE", "closed").lower()
    log_every_s = float(os.getenv("LOADGEN_LOG_EVERY_S", "5"))

    a = torch.randn(n, n, device=device, dtype=mm_dtype)
    b = torch.randn(n, n, device=device, dtype=mm_dtype)

    iters_per_block = 20

    sleep_s = 0.005
    have_nvml = _read_util(gpu_id) is not None
    if mode == "closed" and not have_nvml:
        print(f"[loadgen] gpu{gpu_id}: pynvml unavailable, falling back to open-loop duty mode", flush=True)
        mode = "duty"

    print(f"[loadgen] gpu{gpu_id} SYNTHETIC LOAD start: "
          f"target_util={target:.0f}% mode={mode} matmul={n}x{n} dtype={mm_dtype}", flush=True)

    last_log = time.time()
    ema_util = None

    while True:
        try:
            t0 = time.time()
            for _ in range(iters_per_block):
                c = a @ b
                a = c * 0.0 + a
            torch.cuda.synchronize(gpu_id)
            busy = time.time() - t0

            if sleep_s > 0:
                time.sleep(sleep_s)

            u = _read_util(gpu_id) if mode == "closed" else None
            if u is not None:
                ema_util = u if ema_util is None else 0.7 * ema_util + 0.3 * u
                err = ema_util - target
                step = max(0.0005, busy * 0.15)
                if err > 2:
                    sleep_s = min(2.0, sleep_s + step)
                elif err < -2:
                    sleep_s = max(0.0, sleep_s - step)
            elif mode == "duty":
                duty = max(0.02, min(0.98, target / 100.0))
                sleep_s = busy * (1.0 / duty - 1.0)

            now = time.time()
            if now - last_log >= log_every_s:
                meas = f"{ema_util:.1f}%" if ema_util is not None else "n/a(open-loop)"
                print(f"[loadgen] gpu{gpu_id} SYNTHETIC LOAD: "
                      f"target={target:.0f}% measured={meas} sleep={sleep_s*1000:.1f}ms "
                      f"block_busy={busy*1000:.0f}ms", flush=True)
                last_log = now

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                torch.cuda.empty_cache()
                if n > 512:
                    n = n // 2
                    a = torch.randn(n, n, device=device, dtype=mm_dtype)
                    b = torch.randn(n, n, device=device, dtype=mm_dtype)
                    print(f"[loadgen] gpu{gpu_id} OOM -> matmul={n}x{n}", flush=True)
                    time.sleep(1.0)
                    continue
            print(f"[loadgen] gpu{gpu_id} warning: {e}", flush=True)
            time.sleep(2.0)


if __name__ == "__main__":
    procs = []
    for gid in _gpu_ids():
        p = mp.Process(target=worker, args=(gid,), daemon=True)
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
