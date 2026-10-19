"""Compatibility import for applications written before the m_scraps rename."""
from m_scraps import run_sqs_from_atoms, read_scraps_output, make_scraps_inputs, write_scraps_poscar


def __getattr__(name):
    # Preserve legacy helpers used by callers that import this module first.
    from m_scraps import _api
    return getattr(_api._module, name)
