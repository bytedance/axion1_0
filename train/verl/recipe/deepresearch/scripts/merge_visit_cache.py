#!/usr/bin/env python3
#
# Modified by the Axion authors from the DeepResearch recipe: comments and
# log messages translated to English, hard-coded site paths replaced with
# environment variables or relative paths, and bug fixes. Unmodified files in
# this directory carry their original headers.
#
"""Merge the per-node visit cache directories back into the read-only shared
pool. Run once after training finishes.

  python merge_visit_cache.py --pool <pool-dir> --nodes <node-dir> [<node-dir>...] [--dry-run]

Design:
- The pool is normally read-only (chmod a-w); this script opens it for writing
  and restores the mode at the end.
- Shard by shard, row by row INSERT OR REPLACE, keeping the newer timestamp for
  a repeated url.
- A failing shard is skipped without affecting the others, the same idea as the
  circuit breaker in the tools.
- Writes go to <pool>/.merge_tmp/ first and are swapped in atomically only when
  every shard succeeded, so a mid-way failure leaves nothing half-written.
"""
import argparse, os, shutil, sqlite3, sys, time

TABLES = {
    "url_content":   (["url", "content", "timestamp"], "url"),
    "url_goal_info": (["url", "goal", "useful_information", "timestamp"], "url,goal"),
}
DDL = {
    "url_content":   "CREATE TABLE IF NOT EXISTS url_content (url TEXT PRIMARY KEY, content TEXT, timestamp REAL)",
    "url_goal_info": "CREATE TABLE IF NOT EXISTS url_goal_info (url TEXT, goal TEXT, useful_information TEXT, timestamp REAL, PRIMARY KEY(url,goal))",
}

def shard_files(d):
    return sorted(f for f in os.listdir(d) if f.startswith("visit_shard") and f.endswith(".db"))

def rows_of(path, table, cols):
    """Best-effort read: on a corrupt page, jump forward over rowids, matching the
    salvage logic in the tools."""
    try:
        c = sqlite3.connect("file:%s?mode=ro" % path, uri=True, timeout=15)
    except Exception:
        return
    cl = ",".join(cols); last = 0; CH = 300
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
    ap.add_argument("--nodes", nargs="+", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    if not os.path.isdir(a.pool): sys.exit("pool directory does not exist: " + a.pool)
    names = set(shard_files(a.pool))
    for n in a.nodes:
        if not os.path.isdir(n): sys.exit("node directory does not exist: " + n)
        names |= set(shard_files(n))
    names = sorted(names, key=lambda x: int("".join(ch for ch in x if ch.isdigit()) or 0))
    print("pool   %s\nnodes  %s\nshards %d\n" % (a.pool, ", ".join(a.nodes), len(names)))
    tmp = os.path.join(a.pool, ".merge_tmp")
    if not a.dry_run:
        os.system("chmod -R u+w '%s' 2>/dev/null" % a.pool)
        shutil.rmtree(tmp, ignore_errors=True); os.makedirs(tmp)
    t0 = time.time(); grand_before = grand_added = 0
    for nm in names:
        pool_p = os.path.join(a.pool, nm)
        out = os.path.join(tmp, nm)
        try:
            if not a.dry_run:
                dst = sqlite3.connect(out, timeout=30)
                dst.execute("PRAGMA journal_mode=DELETE"); dst.execute("PRAGMA synchronous=OFF")
                for t in DDL.values(): dst.execute(t)
            before = added = 0
            for src in ([pool_p] if os.path.exists(pool_p) else []) + [os.path.join(n, nm) for n in a.nodes if os.path.exists(os.path.join(n, nm))]:
                is_pool = (src == pool_p)
                for t, (cols, _) in TABLES.items():
                    cnt = 0
                    for row in rows_of(src, t, cols):
                        cnt += 1
                        if not a.dry_run:
                            dst.execute("INSERT OR REPLACE INTO %s(%s) VALUES(%s)" % (t, ",".join(cols), ",".join("?"*len(cols))), row)
                    if is_pool: before += cnt
                    else: added += cnt
            if not a.dry_run:
                dst.commit()
                q = dst.execute("PRAGMA quick_check").fetchone()[0]
                total = sum(dst.execute("SELECT count(*) FROM %s" % t).fetchone()[0] for t in TABLES)
                dst.close()
                if q != "ok":
                    print("  %-20s new db quick_check=%s, skipped (pool left unchanged)" % (nm, str(q)[:30])); os.remove(out); continue
                print("  %-20s pool %6d + node %6d -> merged %6d rows  qc=ok" % (nm, before, added, total))
            else:
                print("  %-20s pool %6d + node %6d rows (dry-run)" % (nm, before, added))
            grand_before += before; grand_added += added
        except Exception as e:
            print("  %-20s error %s: %s (skipped)" % (nm, type(e).__name__, str(e)[:50]))
    if not a.dry_run:
        moved = 0
        for nm in names:
            src = os.path.join(tmp, nm)
            if os.path.exists(src):
                shutil.move(src, os.path.join(a.pool, nm)); moved += 1
        shutil.rmtree(tmp, ignore_errors=True)
        os.system("chmod -R a-w '%s' 2>/dev/null" % a.pool)
        print("\nreplaced %d shards; pool had %d rows + %d added from nodes; %.0fs; pool set back to read-only"
              % (moved, grand_before, grand_added, time.time()-t0))
    else:
        print("\n[dry-run] pool %d rows, nodes %d rows" % (grand_before, grand_added))

if __name__ == "__main__":
    main()
