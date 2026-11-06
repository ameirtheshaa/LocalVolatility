import os
import sys
import tempfile

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY not in sys.path:
    sys.path.insert(0, STUDY)

# Must be set before scc is imported: redirects results/, models/ and the cache to a scratch dir.
if "scc" not in sys.modules:
    _tmp = tempfile.mkdtemp(prefix="scc_test_out_")
    os.environ["SCC_OUTPUT_ROOT"] = _tmp
    os.environ.pop("SCC_CACHE_ROOT", None)
    os.environ.pop("SCC_PHASE", None)
OUTPUT_ROOT = os.environ["SCC_OUTPUT_ROOT"]
