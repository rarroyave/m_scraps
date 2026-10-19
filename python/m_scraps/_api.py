"""Load the shared implementation under its package-qualified name."""
import importlib.util
import sys
from . import _TOOLS

_name = "m_scraps._implementation"
if _name not in sys.modules:
    _spec = importlib.util.spec_from_file_location(_name, _TOOLS / "scraps_ase.py")
    _module = importlib.util.module_from_spec(_spec)
    sys.modules[_name] = _module
    _spec.loader.exec_module(_module)
_module = sys.modules[_name]
run_sqs_from_atoms = _module.run_sqs_from_atoms
read_scraps_output = _module.read_scraps_output
make_scraps_inputs = _module.make_scraps_inputs
write_scraps_poscar = _module.write_scraps_poscar
