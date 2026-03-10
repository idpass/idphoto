"""Configure test paths so examples/eval_utils.py is importable."""

import sys
from pathlib import Path

# Add the examples directory to sys.path so tests can import eval_utils
_examples_dir = str(Path(__file__).resolve().parent.parent / "examples")
if _examples_dir not in sys.path:
    sys.path.insert(0, _examples_dir)
