#!/usr/bin/env python3
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
"""Merge each run's private search / scholar cache directories back into the
read-only shared pool. Run once after training finishes.

  python merge_serper_cache.py --pool <pool-dir> --sources <run-dir> [<run-dir>...] \
         [--kinds search scholar] [--dry-run]

Same design as merge_visit_cache.py:
- The pool is read-only (chmod a-w) with journal_mode=DELETE, so mode=ro
  readers never create -wal/-shm files inside it. A -shm shared across hosts is
  what caused page-level corruption previously.
- Shard by shard, row by row INSERT OR REPLACE, keeping the newer timestamp for
  a repeated query.
- A failing shard is skipped without affecting the others.
- Writes go to <pool>/.merge_tmp/ first and are swapped in atomically only when
  every shard succeeded, so a mid-way failure leaves nothing half-written.
- On a corrupt page, reading continues by jumping forward over rowids.
"""
import argparse, os, shutil, sqlite3, sys, time

KINDS = {
    "search":  ("search_shard",  "search_cache",
                "CREATE TABLE IF NOT EXISTS search_cache (query TEXT PRIMARY KEY, result TEXT NOT NULL, timestamp REAL NOT NULL)"),
    "scholar": ("scholar_shard", "scholar_cache",
                "CREATE TABLE IF NOT EXISTS scholar_cache (query TEXT PRIMARY KEY, result TEXT NOT NULL, timestamp REAL NOT NULL)"),
}
COLS = ["query", "result", "timestamp"]


def shard_files(d, prefix):
    try:
        return sorted(f for f in os.listdir(d) if f.startswith(prefix) and f.endswith(".db"))
    except Exception:
        return []


def rows_of(path, table):
    try:
        c = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=15)
    except Exception:
        return
    cl = ",".join(COLS); last = 0; CH = 300
    while True:
        try:
            rs = c.execute("SELECT rowid,%s FROM %s WHERE rowid>? ORDER BY rowid LIMIT %d" % (cl, table, CH), (last,)).fetchall()
        except Exception:
            step = 1; moved = False
            for _ in range(60):
                try:
                    r = c.execute("SELECT rowid,%s FROM %s WHERE rowid>? ORDER BY rowid LIMIT 1" % (cl, table), (last,)).fetchone()
                except Exception:
                    last += step; step *= 2; continue
                if r is None:
                    c.close(); return
                yield tuple(r[1:]); last = r[0]; moved = True; break
            if not moved:
                c.close(); return
            continue
        if not rs:
            c.close(); return
        for r in rs: yield tuple(r[1:])
        last = rs[-1][0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--kinds", nargs="+", default=["search", "scholar"], choices=list(KINDS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.pool):
        if a.dry_run: sys.exit("pool directory does not exist: " + a.pool)
        os.makedirs(a.pool); print("pool directory did not exist, created: %s" % a.pool)
    for s in a.sources:
        if not os.path.isdir(s): sys.exit("source directory does not exist: " + s)
    print("pool    %s\nsources %s\nkinds   %s\n" % (a.pool, ", ".join(a.sources), ", ".join(a.kinds)))
    tmp = os.path.join(a.pool, ".merge_tmp")
    if not a.dry_run:
        os.system("chmod -R u+w '%s' 2>/dev/null" % a.pool)
        shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    t0 = time.time(); g_before = g_added = 0; moved = 0
    for kind in a.kinds:
        prefix, table, ddl = KINDS[kind]
        names = set(shard_files(a.pool, prefix))
        for s in a.sources: names |= set(shard_files(s, prefix))
        names = sorted(names, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or 0))
        if not names:
            print("  [%s] no shards, skipping" % kind); continue
        print("  [%s] %d shards" % (kind, len(names)))
        for nm in names:
            pool_p = os.path.join(a.pool, nm); out = os.path.join(tmp, nm)
            try:
                if not a.dry_run:
                    dst = sqlite3.connect(out, timeout=30)
                    dst.execute("PRAGMA journal_mode=DELETE"); dst.execute("PRAGMA synchronous=OFF")
                    dst.execute(ddl)
                before = added = 0
                srcs = ([pool_p] if os.path.exists(pool_p) else []) + \
                       [os.path.join(s, nm) for s in a.sources if os.path.exists(os.path.join(s, nm))]
                for src in srcs:
                    cnt = 0
                    for row in rows_of(src, table):
                        cnt += 1
                        if not a.dry_run:
                            dst.execute("INSERT OR REPLACE INTO %s(%s) VALUES(?,?,?)" % (table, ",".join(COLS)), row)
                    if src == pool_p: before += cnt
                    else: added += cnt
                if not a.dry_run:
                    dst.commit()
                    q = dst.execute("PRAGMA quick_check").fetchone()[0]
                    total = dst.execute("SELECT count(*) FROM %s" % table).fetchone()[0]
                    dst.close()
                    if q != "ok":
                        print("    %-22s new db quick_check=%s, skipped (pool left unchanged)" % (nm, str(q)[:30]))
                        os.remove(out); continue
                    print("    %-22s pool %7d + source %7d -> merged %7d rows  qc=ok" % (nm, before, added, total))
                else:
                    print("    %-22s pool %7d + source %7d rows (dry-run)" % (nm, before, added))
                g_before += before; g_added += added
            except Exception as e:
                print("    %-22s error %s: %s (skipped)" % (nm, type(e).__name__, str(e)[:50]))
    if not a.dry_run:
        for nm in sorted(os.listdir(tmp)):
            shutil.move(os.path.join(tmp, nm), os.path.join(a.pool, nm)); moved += 1
        shutil.rmtree(tmp, ignore_errors=True)
        for f in os.listdir(a.pool):
            if f.endswith(("-wal", "-shm")):
                try: os.remove(os.path.join(a.pool, f))
                except Exception: pass
        os.system("chmod -R a-w '%s' 2>/dev/null" % a.pool)
        print("\nreplaced %d shards; pool had %d rows + %d added from sources; %.0fs; pool set back to read-only (DELETE mode)"
              % (moved, g_before, g_added, time.time() - t0))
    else:
        print("\n[dry-run] pool %d rows, sources %d rows" % (g_before, g_added))


if __name__ == "__main__":
    main()
