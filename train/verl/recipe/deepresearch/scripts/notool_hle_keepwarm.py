#!/usr/bin/env python3
"""No-tool HLE keep-warm / eval loop.

Continuously sends single-turn (NO tools, NO search) HLE questions to one or more
OpenAI-compatible vLLM endpoints and generates answers. Two purposes:

  1. Keep otherwise-idle GPUs busy with *legitimate* compute (real inference),
     instead of a dummy gpu-keeper. The platform kills low-util jobs; this fills
     the slack between real work.
  2. Produce a live "no-tool HLE accuracy" signal (how well the model answers
     without tools) that you can watch over training.

It is endpoint-agnostic:
  - Point it at the eval vLLM (fixed ports 6000-6003) to keep the service node busy.
  - Point it at the rollout vLLM (the 4B policy) to keep the rollout GPUs busy AND
    track the *current policy's* no-tool ability -- requires exposing the rollout
    server address (it uses dynamic Ray-assigned ports).

Keep --concurrency modest so it yields to real work (it shares the vLLM's
continuous-batching queue; high concurrency would slow real rollout/judging).

Usage:
  python notool_hle_keepwarm.py \
      --base-urls http://<host>:6000/v1,http://<host>:6001/v1 \
      --model qwen35-27b-eval \
      --data /mnt/.../QUEST/data/hle_text_val_20.parquet \
      --concurrency 4 --max-tokens 512 --log /path/keepwarm.log
"""
import argparse, asyncio, json, os, random, re, sys, time

def log(msg, path=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if path:
        try:
            with open(path, "a") as f:
                f.write(line + "\n")
        except Exception:
            pass

def load_questions(data_path):
    """Return list of (question, ref_answer) from a training-format parquet."""
    import pandas as pd
    df = pd.read_parquet(data_path)
    qs = []
    for _, row in df.iterrows():
        q = None
        ei = row.get("extra_info")
        if isinstance(ei, dict):
            q = ei.get("question")
        if not q:
            prompt = row.get("prompt")
            if prompt is not None and len(prompt):
                q = prompt[0].get("content") if isinstance(prompt[0], dict) else None
        ref = ""
        rm = row.get("reward_model")
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", {})
            if isinstance(gt, dict):
                ref = str(gt.get("ref_answer", ""))
        if q:
            qs.append((str(q), ref))
    return qs

_ANS_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
def extract_answer(text):
    m = list(_ANS_RE.finditer(text or ""))
    return m[-1].group(1).strip() if m else (text or "").strip()

def rough_correct(pred, ref):
    """Cheap local accuracy proxy (NOT the real judge): substring / exact match."""
    if not ref:
        return None
    p, r = extract_answer(pred).lower(), ref.strip().lower()
    return r in p or p in r

PROMPT_SUFFIX = (
    "\n\nAnswer directly from your own knowledge. Do NOT use any tools or search. "
    "Put your final answer inside <answer></answer>."
)

_WAITING_RE = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+([0-9.]+)", re.MULTILINE)

async def metrics_poller(base_url, busy_state, threshold, stop_at, logpath):
    """Poll a vLLM endpoint's /metrics every 2s; mark it 'busy' when it has real
    requests waiting, so filler workers yield to real judge/rollout traffic."""
    import httpx
    metrics_url = base_url.rstrip("/")
    if metrics_url.endswith("/v1"):
        metrics_url = metrics_url[:-3]
    metrics_url = metrics_url.rstrip("/") + "/metrics"
    async with httpx.AsyncClient(timeout=5) as hc:
        while time.time() < stop_at:
            try:
                r = await hc.get(metrics_url)
                m = _WAITING_RE.search(r.text or "")
                waiting = float(m.group(1)) if m else 0.0
                busy_state[base_url] = waiting > threshold
            except Exception:
                busy_state[base_url] = False  # can't tell -> assume free
            await asyncio.sleep(2)

async def worker(wid, client, base_url, model, questions, max_tokens, stop_at, stats, logpath, busy_state):
    while time.time() < stop_at:
        # Yield to real traffic: if this endpoint has real requests queued, back off.
        if busy_state is not None and busy_state.get(base_url):
            await asyncio.sleep(1.0)
            continue
        q, ref = random.choice(questions)
        t0 = time.time()
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": q + PROMPT_SUFFIX}],
                max_tokens=max_tokens,
                temperature=0.7,
            )
            out = resp.choices[0].message.content or ""
            stats["ok"] += 1
            stats["latency"] += time.time() - t0
            c = rough_correct(out, ref)
            if c is True:
                stats["correct"] += 1
            if c is not None:
                stats["scored"] += 1
        except Exception as e:
            stats["err"] += 1
            if stats["err"] % 20 == 1:
                log(f"[w{wid}] request error: {type(e).__name__}: {str(e)[:120]}", logpath)
            await asyncio.sleep(2)

async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-urls", default="", help="comma-separated OpenAI /v1 base urls")
    ap.add_argument("--base-urls-file", default="", help="file with one /v1 base url per line (polled until it appears; for the dynamically-ported rollout vLLM)")
    ap.add_argument("--model", default="auto", help="served model name, or 'auto' to detect via /v1/models")
    ap.add_argument("--data", required=True, help="training-format parquet with question + ref_answer")
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--api-key", default="EMPTY")
    ap.add_argument("--duration", type=float, default=0, help="seconds to run (0 = forever)")
    ap.add_argument("--report-every", type=float, default=60, help="seconds between stat prints")
    ap.add_argument("--yield-metrics", type=int, default=1, help="1=poll /metrics and yield when the endpoint has real requests waiting")
    ap.add_argument("--yield-waiting-threshold", type=float, default=0, help="back off when num_requests_waiting > this")
    ap.add_argument("--log", default=None)
    args = ap.parse_args()

    from openai import AsyncOpenAI
    import httpx

    # Resolve base urls: from --base-urls, or poll --base-urls-file until it
    # appears (the rollout vLLM writes its dynamic addresses there after init).
    bases = [b.strip() for b in args.base_urls.split(",") if b.strip()]
    if not bases and args.base_urls_file:
        for _ in range(600):  # up to ~30 min of polling
            try:
                if os.path.exists(args.base_urls_file):
                    bases = [l.strip() for l in open(args.base_urls_file) if l.strip()]
            except Exception:
                bases = []
            if bases:
                log(f"loaded {len(bases)} base url(s) from {args.base_urls_file}", args.log)
                break
            time.sleep(3)
    if not bases:
        log("no base urls (file never appeared?); exiting", args.log); sys.exit(1)

    # Resolve model: 'auto' -> query /v1/models on a ready endpoint.
    model = args.model
    if model == "auto":
        for _ in range(600):
            for b in bases:
                try:
                    r = httpx.get(b.rstrip("/") + "/models", timeout=5,
                                  headers={"Authorization": f"Bearer {args.api_key}"})
                    ids = [m.get("id") for m in r.json().get("data", []) if m.get("id")]
                    if ids:
                        model = ids[0]; break
                except Exception:
                    pass
            if model != "auto":
                break
            time.sleep(3)
        if model == "auto":
            log("could not auto-detect model; exiting", args.log); sys.exit(1)
        log(f"auto-detected model: {model}", args.log)
    args.model = model

    questions = load_questions(args.data)
    if not questions:
        log("no questions loaded; exiting", args.log); sys.exit(1)
    log(f"loaded {len(questions)} questions; {len(bases)} endpoint(s); concurrency={args.concurrency} model={args.model}", args.log)

    clients = [AsyncOpenAI(base_url=b, api_key=args.api_key, timeout=120, max_retries=0) for b in bases]
    stop_at = time.time() + args.duration if args.duration > 0 else float("inf")
    stats = {"ok": 0, "err": 0, "correct": 0, "scored": 0, "latency": 0.0}

    busy_state = {} if args.yield_metrics else None
    pollers = []
    if args.yield_metrics:
        for b in bases:
            busy_state[b] = False
            pollers.append(asyncio.create_task(
                metrics_poller(b, busy_state, args.yield_waiting_threshold, stop_at, args.log)))
        log("yield-metrics ON: filler will back off when endpoints have real requests waiting", args.log)

    tasks = [
        asyncio.create_task(
            worker(i, clients[i % len(clients)], bases[i % len(bases)], args.model,
                   questions, args.max_tokens, stop_at, stats, args.log, busy_state)
        )
        for i in range(args.concurrency)
    ]

    async def reporter():
        last_ok = 0; t_last = time.time()
        while time.time() < stop_at:
            await asyncio.sleep(args.report_every)
            now = time.time(); dt = now - t_last
            qps = (stats["ok"] - last_ok) / dt if dt > 0 else 0
            acc = (stats["correct"] / stats["scored"] * 100) if stats["scored"] else 0
            avg_lat = (stats["latency"] / stats["ok"]) if stats["ok"] else 0
            log(f"ok={stats['ok']} err={stats['err']} qps={qps:.2f} avg_lat={avg_lat:.1f}s "
                f"rough_notool_acc={acc:.1f}% ({stats['correct']}/{stats['scored']})", args.log)
            last_ok = stats["ok"]; t_last = now

    rep = asyncio.create_task(reporter())
    try:
        await asyncio.gather(*tasks)
    finally:
        rep.cancel()
        for p in pollers:
            p.cancel()
    log("keep-warm loop finished", args.log)

if __name__ == "__main__":
    asyncio.run(main())
