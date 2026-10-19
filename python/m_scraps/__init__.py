"""Multi-sublattice SCRAPS: public Python API and compatible legacy tooling."""
from pathlib import Path
import sys

__version__ = "0.5.1"
_PACKAGE = Path(__file__).resolve().parent
# Source checkout and built wheel share the existing tools without copying
# their implementation into a second maintained Python API.
_TOOLS = _PACKAGE / "tools"
if not _TOOLS.is_dir():
    _TOOLS = _PACKAGE.parents[1] / "tools"
if str(_TOOLS) not in sys.path:
    sys.path.insert(0, str(_TOOLS))

__all__ = ["analyze_structure", "run_sqs_from_atoms", "read_scraps_output", "make_scraps_inputs", "write_scraps_poscar"]

def __getattr__(name):
    if name == "analyze_structure":
        from structure_analysis import analyze_structure
        return analyze_structure
    if name in __all__:
        from . import _api
        return getattr(_api, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
