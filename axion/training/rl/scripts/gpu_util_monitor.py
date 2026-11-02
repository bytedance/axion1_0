#!/usr/bin/env python3
"""Per-node GPU util monitor that separates the gpu_keeper (dummy) contribution
from the real RL training utilization -- via a BASELINE, since per-process pmon
attribution does not work here (container isolation hides the gpu_keeper PIDs
from nvidia-smi pmon).

How it works:
  Phase 1 (baseline): for the first --baseline-secs seconds the RL models have not
    loaded / are not computing yet, so whole-GPU util == gpu_keeper's contribution.
    We average it per-GPU -> keeper baseline. (Skipped if --keeper-baseline given.)
  Phase 2 (steady): every --interval seconds, log:
      total  = whole-GPU util now
      keeper = the baseline (gpu_keeper's ~constant floor)
      rl     = max(0, total - keeper)   == genuine RL training utilization.

Line format (parsed by gpu_util_report.sh):
  <ISO ts> node=<id> gpu=<g> total=<t> keeper=<k> rl=<r> mem_mb=<m>

Caveats: additive model (total ~= keeper + rl). Under heavy contention gpu_keeper
gets squeezed below its baseline, so rl is a slight under-estimate then -- fine as
a conservative "real RL util" number.
"""
import argparse, subprocess, time


def query_total():
    """Return {gpu_index: (util, mem_mb)}."""
    res = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 3:
                res[int(parts[0])] = (float(parts[1]), float(parts[2]))
    except Exception:
        pass
    return res


def measure_baseline(secs, node_id, log):
    """Average per-GPU util over `secs` seconds -> gpu_keeper baseline."""
    acc, n = {}, {}
    t_end = time.time() + secs
    while time.time() < t_end:
        for g, (util, _mem) in query_total().items():
            acc[g] = acc.get(g, 0.0) + util
            n[g] = n.get(g, 0) + 1
        time.sleep(3)
    base = {g: (acc[g] / n[g] if n.get(g) else 0.0) for g in acc}
    try:
        with open(log, "a") as f:
            f.write(f"# node={node_id} gpu_keeper baseline (per-gpu util%): "
                    + ", ".join(f"gpu{g}={v:.0f}" for g, v in sorted(base.items())) + "\n")
    except Exception:
        pass
    return base


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--node-id", default="0")
    ap.add_argument("--log", required=True)
    ap.add_argument("--interval", type=float, default=30)
    ap.add_argument("--baseline-secs", type=float, default=60,
                    help="measure the gpu_keeper baseline over this window at startup")
    ap.add_argument("--keeper-baseline", type=float, default=-1,
                    help="override: fixed keeper util%% for every GPU (skips measurement)")
    args = ap.parse_args()

    if args.keeper_baseline >= 0:
        base = {}  # filled lazily per gpu with the constant
        const = args.keeper_baseline
    else:
        base = measure_baseline(args.baseline_secs, args.node_id, args.log)
        const = None

    while True:
        ts = time.strftime("%Y-%m-%dT%H:%M:%S")
        totals = query_total()
        try:
            with open(args.log, "a") as f:
                for g in sorted(totals):
                    tot, mem = totals[g]
                    k = const if const is not None else base.get(g, 0.0)
                    k = min(k, tot)
                    rl = max(0.0, tot - k)
                    f.write(f"{ts} node={args.node_id} gpu={g} total={tot:.0f} "
                            f"keeper={k:.0f} rl={rl:.0f} mem_mb={mem:.0f}\n")
        except Exception:
            pass
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
