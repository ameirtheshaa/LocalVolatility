#!/usr/bin/env python3
"""Re-key cached references after a change of the identity scheme (one-off maintenance tool).

Why: references generated before the reference-kind identity was narrowed carry the *full* code hash in
their identity; the cache is addressed by the identity hash, so they would be invisible to the narrowed
scheme and recomputed (expensive). This tool rewrites each sidecar whose ``identity["code"]`` equals
``--old`` (default: the current full-scope hash) to the current reference-scope hash, renames the
npz/json pair to the new identity hash, and updates results/references/index.json accordingly.

    python _migrate_cache_identity.py [--old HASH] [--dry-run]

Safe to run repeatedly; entries already on the new scheme are untouched.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import scc  # noqa: E402
from scc import identity  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", default=identity.code_identity("full"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    new_code = identity.code_identity("reference")
    cache_dir = os.path.join(scc.CACHE_ROOT, "reference")
    print(f"cache {cache_dir}: old code {args.old} -> new code {new_code}", flush=True)
    renamed = {}
    for side in sorted(glob.glob(os.path.join(cache_dir, "*.json"))):
        try:
            d = json.load(open(side, encoding="utf-8"))
        except Exception as e:
            print(f"  skip unreadable {side}: {e}")
            continue
        ident = d.get("identity") or {}
        if ident.get("kind") != "reference" or ident.get("code") != args.old:
            continue
        old_hash = os.path.basename(side)[:-5]
        ident = dict(ident)
        ident["code"] = new_code
        new_hash = identity.identity_hash(ident)
        npz_old = side[:-5] + ".npz"
        if not os.path.isfile(npz_old):
            print(f"  skip {old_hash}: npz missing")
            continue
        d["identity"] = ident
        d["identity_hash"] = new_hash
        d["migrated_from"] = old_hash
        if args.dry_run:
            print(f"  would rename {old_hash} -> {new_hash} ({ident.get('case_id')} n={ident.get('n')} sub={ident.get('sub')})")
            renamed[old_hash] = new_hash
            continue
        new_side = os.path.join(cache_dir, new_hash + ".json")
        new_npz = os.path.join(cache_dir, new_hash + ".npz")
        identity.write_json(new_side, d)
        identity.replace_retry(npz_old, new_npz)
        os.remove(side)
        renamed[old_hash] = new_hash
        print(f"  renamed {old_hash} -> {new_hash} ({ident.get('case_id')} n={ident.get('n')} sub={ident.get('sub')})", flush=True)
    index_path = os.path.join(scc.RESULTS, "references", "index.json")
    if renamed and os.path.isfile(index_path) and not args.dry_run:
        idx = json.load(open(index_path, encoding="utf-8"))
        n_upd = 0
        for cid, levels in idx.items():
            if not isinstance(levels, dict):
                continue
            for lv, e in levels.items():
                if isinstance(e, dict) and e.get("identity_hash") in renamed:
                    new_hash = renamed[e["identity_hash"]]
                    e["identity_hash"] = new_hash
                    if e.get("npz_path"):
                        e["npz_path"] = os.path.join(os.path.dirname(e["npz_path"]), new_hash + ".npz")
                    n_upd += 1
        identity.write_json(index_path, idx)
        print(f"index.json: {n_upd} entries updated", flush=True)
    print(f"done: {len(renamed)} references re-keyed", flush=True)


if __name__ == "__main__":
    main()
