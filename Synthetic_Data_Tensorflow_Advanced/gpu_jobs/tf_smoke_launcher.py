#!/usr/bin/env python3
"""
Launcher: runs mz_pinn_step4_smoke.py using the tf216 conda env.
Needed because the gpu-laptop MCP looks for tf216 at C:/Users/ameir/anaconda3/envs/tf216
but the actual path is C:/ProgramData/anaconda3/envs/tf216.
"""
import subprocess, sys, os, json

TF216_PY = r"C:\ProgramData\anaconda3\envs\tf216\python.exe"
SCRIPT   = r"C:\mcp_jobs\scripts\mz_pinn_step4_smoke.py"

if not os.path.exists(TF216_PY):
    print(f"ERROR: {TF216_PY} not found")
    sys.exit(1)

print(f"Launching: {TF216_PY} {SCRIPT}")
sys.stdout.flush()

env = os.environ.copy()
# Fix Windows cp1252 encoding breaking on Unicode chars (e.g. ✓ in dupire_pipeline)
env["PYTHONIOENCODING"] = "utf-8"
env["PYTHONUTF8"] = "1"

result = subprocess.run(
    [TF216_PY, "-X", "utf8", "-u", SCRIPT],
    stdout=sys.stdout, stderr=sys.stderr,
    env=env,
)
sys.exit(result.returncode)
