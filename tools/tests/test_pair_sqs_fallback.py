import numpy as np
from pymatgen.core import Lattice, Structure

from pair_sqs_fallback import anneal_random_pairs, random_pair_targets
from periodic_multigraph import periodic_shells


def test_random_pair_targets_reject_nonintegral_cell():
    try:
        random_pair_targets((28, 28, 28), 8)
    except ValueError as error:
        assert "nonintegral" in str(error)
    else:
        raise AssertionError("84-site equiatomic ternary must not have exact pair targets")


def test_fallback_finds_exact_random_first_shell_for_cubic_bcc():
    # 3x3x3 conventional BCC: 54 atoms, 18 per species, exact NN targets.
    structure = Structure(Lattice.cubic(3.2), ["Mo", "Mo"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    structure.make_supercell((3, 3, 3))
    result = anneal_random_pairs(structure, ("Mo", "Nb", "Ti"), (18, 18, 18), seed=17, steps=80_000)
    assert result.objective == 0.0
    assert np.allclose(result.alpha, 0.0)
    assert periodic_shells(result.structure, 1)[0].coordination == 8
