import os
import sys
import tempfile

STUDY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if STUDY not in sys.path:
    sys.path.insert(0, STUDY)

# Must be set before dsm.checkpoint / dsm.reference are imported.
if "dsm.checkpoint" not in sys.modules:
    _tmp = tempfile.mkdtemp(prefix="dsm_test_out_")
    os.environ["DSM_OUTPUT_ROOT"] = _tmp
OUTPUT_ROOT = os.environ["DSM_OUTPUT_ROOT"]
