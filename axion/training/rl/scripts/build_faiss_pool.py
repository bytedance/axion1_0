#!/usr/bin/env python3
#
# Modified by the Axion authors for package paths, deployment settings and runtime fixes.
#
"""Rebuild the pool's FAISS index (the semantic layer) from the sqlite shards
of the read-only shared pool.

  python build_faiss_pool.py --pool <pool-dir> [--kinds search scholar]
                             [--model <path>] [--device cpu|cuda] [--batch-size 512]

Why this is separate: the stock build_faiss_from_search_cache() takes a single
sqlite file, while the pool is sharded. This script collects (query, result)
from every shard and builds one index.

Outputs, written into the --pool directory:
  {kind}_faiss.index        IndexFlatIP, vectors L2-normalised (inner product = cosine)
  {kind}_faiss_meta.db      faiss_search_meta(id, query, result), id = vector index

The teardown matches the merge scripts: journal_mode=DELETE on the meta db,
wal/shm removed, and the whole pool chmod a-w, so mode=ro readers never create
a -shm inside it. Re-run after every merge of new data into the pool.
"""
import argparse, glob, os, sqlite3, sys, time

KINDS = {"search": ("search_shard", "search_cache"),
         "scholar": ("scholar_shard", "scholar_cache")}


def read_rows(pool, prefix, table):
    seen, rows = set(), []
    for f in sorted(glob.glob(os.path.join(pool, prefix + "*.db"))):
        try:
            c = sqlite3.connect("file:%s?mode=ro" % f, uri=True, timeout=30)
            for q, r in c.execute("SELECT query, result FROM %s" % table):
                if q is None or r is None or q in seen:
                    continue
                seen.add(q); rows.append((q, r))
            c.close()
        except Exception as e:
            print("    %s read failed (skipped): %s" % (os.path.basename(f), e))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--kinds", nargs="+", default=["search", "scholar"], choices=list(KINDS))
    ap.add_argument("--model", default=os.getenv("DEEPRESEARCH_EMBEDDING_MODEL")
                    or "sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--max-entries", type=int, default=None)
    a = ap.parse_args()
    if not os.path.isdir(a.pool):
        sys.exit("pool directory does not exist: " + a.pool)

    import numpy as np, faiss
    from sentence_transformers import SentenceTransformer
    dev = a.device
    if dev is None:
        try:
            import torch; dev = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            dev = "cpu"
    print("pool  %s\nmodel %s (device=%s)\n" % (a.pool, a.model, dev))
    os.system("chmod -R u+w '%s' 2>/dev/null" % a.pool)
    enc = SentenceTransformer(a.model, device=dev)
    dim = enc.get_sentence_embedding_dimension()

    for kind in a.kinds:
        prefix, table = KINDS[kind]
        t0 = time.time()
        rows = read_rows(a.pool, prefix, table)
        if a.max_entries:
            rows = rows[: a.max_entries]
        if not rows:
            print("  [%s] no data, skipping" % kind); continue
        print("  [%s] %d deduplicated queries, encoding..." % (kind, len(rows)))
        index = faiss.IndexFlatIP(dim)
        ipath = os.path.join(a.pool, "%s_faiss.index" % kind)
        mpath = os.path.join(a.pool, "%s_faiss_meta.db" % kind)
        tmp_i, tmp_m = ipath + ".tmp", mpath + ".tmp"
        for p in (tmp_i, tmp_m):
            if os.path.exists(p): os.remove(p)
        mc = sqlite3.connect(tmp_m, timeout=60)
        mc.execute("PRAGMA journal_mode=DELETE"); mc.execute("PRAGMA synchronous=OFF")
        mc.execute("CREATE TABLE IF NOT EXISTS faiss_search_meta "
                   "(id INTEGER PRIMARY KEY, query TEXT NOT NULL, result TEXT NOT NULL)")
        done = 0
        for st in range(0, len(rows), a.batch_size):
            ch = rows[st: st + a.batch_size]
            emb = enc.encode([q for q, _ in ch], normalize_embeddings=True,
                             batch_size=a.batch_size, show_progress_bar=False)
            index.add(np.asarray(emb, dtype=np.float32))
            mc.executemany("INSERT OR REPLACE INTO faiss_search_meta (id, query, result) VALUES (?,?,?)",
                           [(st + i, ch[i][0], ch[i][1]) for i in range(len(ch))])
            done += len(ch)
            if (st // a.batch_size) % 40 == 0:
                el = time.time() - t0
                print("    %d/%d  %.0f rows/s" % (done, len(rows), done / max(el, 1e-6)), flush=True)
        mc.commit(); mc.close()
        faiss.write_index(index, tmp_i)
        os.replace(tmp_i, ipath); os.replace(tmp_m, mpath)
        print("  [%s] done: %d vectors, dim=%d, %.0fs" % (kind, index.ntotal, dim, time.time() - t0))

    for f in os.listdir(a.pool):
        if f.endswith(("-wal", "-shm")):
            try: os.remove(os.path.join(a.pool, f))
            except Exception: pass
    os.system("chmod -R a-w '%s' 2>/dev/null" % a.pool)
    print("\npool set back to read-only")


if __name__ == "__main__":
    main()
