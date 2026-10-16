# Copyright (c) 2026 ByteDance Ltd. and/or its affiliates
# SPDX-License-Identifier: LicenseRef-Axion-NC-1.0
# Licensed under the Axion Non-Commercial Research and Responsible Use
# License, Version 1.0. See the LICENSE file at the repository root.

"""Check that every python command in a README still matches its script's CLI.

For each `python <script> --flag ...` line in a README, this resolves the
script (following any preceding `cd`), reads the script's own --help, and
reports flags the script does not accept and scripts that do not exist.

It catches the case where a generator is replaced but its documentation is
not, which otherwise fails only when a reader runs the first command.

    python train/check_readme_commands.py
"""
import argparse
import glob
import os
import re
import subprocess
import sys

SKIP = ("/LlamaFactory/", "/verl/docs/", "/verl/examples/")


def flags_of(script_path: str) -> set[str]:
    """Long options the script accepts, read from its own --help."""
    try:
        h = subprocess.run(
            [sys.executable, script_path, "--help"],
            capture_output=True, text=True, timeout=120,
            cwd=os.path.dirname(script_path) or ".",
        )
    except Exception:
        return set()
    # argparse wraps usage lines, so join them before matching.
    text = " ".join((h.stdout + h.stderr).split())
    return set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", text))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(__file__), ".."))
    a = ap.parse_args()
    root = os.path.abspath(a.root)

    problems, checked = [], 0
    for md in sorted(glob.glob(os.path.join(root, "**", "README.md"), recursive=True)):
        rel = os.path.relpath(md, root)
        if any(s in md for s in SKIP):
            continue
        base = os.path.dirname(md)
        cwd = base
        for i, line in enumerate(open(md, encoding="utf-8").read().split("\n"), 1):
            s = line.strip()
            cd = re.match(r"^cd\s+(\S+)", s)
            if cd:
                # A relative cd may be meant from the directory of the README,
                # from the current directory, or from the repository root; a
                # later `cd seal0` after an earlier `cd hle` means the first.
                for cand in (os.path.normpath(os.path.join(base, cd.group(1))),
                             os.path.normpath(os.path.join(cwd, cd.group(1))),
                             os.path.normpath(os.path.join(root, cd.group(1)))):
                    if os.path.isdir(cand):
                        cwd = cand
                        break
                continue
            m = re.match(r"^python3?\s+(\S+\.py)\s+(.*)$", s)
            if not m:
                continue
            script, args = m.group(1), m.group(2)
            path = next((c for c in (os.path.join(cwd, script),
                                     os.path.join(root, script),
                                     os.path.join(base, script)) if os.path.exists(c)), None)
            if path is None:
                problems.append(f"{rel}:{i}  script not found: {script}")
                continue
            accepted = flags_of(path)
            if not accepted:
                continue          # no argparse, or it could not be imported
            used = set(re.findall(r"--[a-zA-Z0-9][a-zA-Z0-9-]*", args.replace("\\", "")))
            unknown = sorted(used - accepted)
            if unknown:
                problems.append(f"{rel}:{i}  {script} does not accept {unknown}")
            checked += 1

    for p in problems:
        print(f"  {p}")
    print(f"{len(problems)} problems in {checked} documented commands")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
