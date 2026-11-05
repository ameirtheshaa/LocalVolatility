"""Timing protocol helpers (PLAN.md section 9)."""
from __future__ import annotations

import contextlib
import os
import platform
import sys
import time
from typing import Callable, Dict

import numpy as np

_THREAD_VARS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")


@contextlib.contextmanager
def pin_threads():
    """Set OMP/MKL/OPENBLAS_NUM_THREADS=1 (and threadpoolctl limit 1 when importable); restore on exit."""
    saved = {k: os.environ.get(k) for k in _THREAD_VARS}
    for k in _THREAD_VARS:
        os.environ[k] = "1"
    limiter = None
    try:
        from threadpoolctl import threadpool_limits
        limiter = threadpool_limits(limits=1)
    except Exception:
        limiter = None
    try:
        yield
    finally:
        if limiter is not None:
            try:
                limiter.unregister()
            except Exception:
                pass
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def timing_env() -> Dict:
    """Hardware / software fingerprint for the timing tables."""
    import scipy
    blas = None
    try:
        blas = np.show_config(mode="dicts")
        blas = blas.get("Build Dependencies", blas) if isinstance(blas, dict) else blas
    except Exception:
        try:
            blas = str(np.__config__.show())
        except Exception:
            blas = "unavailable"
    try:
        from threadpoolctl import threadpool_info
        tpi = list(threadpool_info())
    except Exception:
        tpi = []
    return {
        "platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
        "cpu_count": os.cpu_count(), "python": sys.version.split()[0], "numpy": np.__version__,
        "scipy": scipy.__version__, "blas": blas, "threadpoolctl_info": tpi,
        "threads_env": {k: os.environ.get(k) for k in _THREAD_VARS},
    }


def time_callable(fn: Callable[[], object], repeats: int = 7, warmup: int = 2) -> Dict:
    """cold = first call; then ``warmup`` untimed calls; then ``repeats`` timed calls (perf_counter)."""
    t0 = time.perf_counter()
    fn()
    cold = time.perf_counter() - t0
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    s = np.asarray(samples)
    q75, q25 = np.percentile(s, [75, 25])
    return {"median": float(np.median(s)), "iqr": float(q75 - q25), "min": float(s.min()), "max": float(s.max()),
            "samples": [float(x) for x in samples], "cold": float(cold)}
