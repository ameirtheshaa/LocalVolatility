#!/usr/bin/env python3
"""Launcher for running one pipeline stage on the GPU-laptop CPU lane (Windows, Dropbox-synced checkout).

    python _laptop_job.py <stage> [stage args...]        e.g. python _laptop_job.py s2_references --split train val

* keeps the regenerable reference cache OFF Dropbox (SCC_CACHE_ROOT=C:/mcp_jobs/scc_cache, override with
  the env var SCC_CACHE_ROOT or --cache-root as the first option);
* pins BLAS threads (OMP/MKL/OPENBLAS_NUM_THREADS=1) unless --threads N is given as the first option
  (the cost protocol pins them itself; reference generation with --parallel K wants 1 thread per worker);
* prints the interpreter and library versions, then runs scripts/<stage>.py as __main__ with the remaining args.
Results, manifests, models and verify files are written under the (Dropbox-synced) study directory as usual.
"""
from __future__ import annotations

import os
import runpy
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    args = sys.argv[1:]
    cache_root = os.environ.get("SCC_CACHE_ROOT", "C:/mcp_jobs/scc_cache" if os.name == "nt" else None)
    threads = "1"
    while args and args[0] in ("--cache-root", "--threads"):
        key = args.pop(0)
        val = args.pop(0)
        if key == "--cache-root":
            cache_root = val
        else:
            threads = val
    if not args:
        print(__doc__)
        sys.exit(2)
    stage = args.pop(0)
    if cache_root:
        os.makedirs(cache_root, exist_ok=True)
        os.environ["SCC_CACHE_ROOT"] = cache_root
    for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[k] = str(threads)
    import numpy, scipy  # noqa: E402  (after the thread pins)
    print(f"[laptop_job] stage={stage} args={args}", flush=True)
    print(f"[laptop_job] python {sys.version.split()[0]} numpy {numpy.__version__} scipy {scipy.__version__} "
          f"platform {sys.platform} cpus {os.cpu_count()} threads {threads} cache_root {cache_root}", flush=True)
    script = os.path.join(HERE, stage + ".py")
    if not os.path.isfile(script):
        print(f"[laptop_job] no such stage script: {script}", flush=True)
        sys.exit(2)
    sys.argv = [script] + args
    t0 = time.time()
    try:
        runpy.run_path(script, run_name="__main__")
        code = 0
    except SystemExit as e:
        code = int(e.code or 0) if isinstance(e.code, int) or e.code is None else 1
    print(f"[laptop_job] stage {stage} finished with exit code {code} in {time.time() - t0:.0f}s", flush=True)
    sys.exit(code)


if __name__ == "__main__":
    main()
