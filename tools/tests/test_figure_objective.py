"""Native figure objectives: exhaustive swap audit, parser and MPI integration."""
from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from test_input_parser import GOOD_INPUT, MPIRUN, SCRAPS_BIN, run_scraps

REPO = Path(__file__).resolve().parents[2]
# Includes repeated periodic site IDs and a figure touching both swap sites.
TERMS = [
    [(0.5, [(0, 0), (1, 0)]), (0.5, [(1, 0), (2, 0)])],
    [(0.5, [(0, 0), (0, 0), (1, 0)]), (0.5, [(1, 0), (2, 0), (3, 0)])],
    [(1.0, [(0, 0), (1, 0), (2, 0), (3, 0)])],
]


def write_figures(path, n=4):
    positions = ([[0, 0, 0]] * n if n == 4 else
                 [[(x + b) / 2, (y + b) / 2, (z + b) / 2]
                  for b in [0, 0.5] for z in range(2)
                  for y in range(2) for x in range(2)])
    cell = np.eye(3) * (1 if n == 4 else 2)
    lines = [f"SCRAPS_FIGURES 1\n{n} 2 1 3", " ".join(map(str, cell.ravel()))]
    lines.extend(" ".join(map(str, p)) for p in positions)
    lines.append("-1 1")
    for order, terms in zip([2, 3, 4], TERMS):
        lines.append(f"{order} {order - 1} 0 {len(terms)}")
        for coefficient, factors in terms:
            lines.append(" ".join(map(str, [coefficient] + [v for p in factors for v in p])))
    path.write_text("\n".join(lines) + "\n")


def reference(occupations):
    spins = 2 * np.array(occupations) - 1
    return np.array([sum(w * np.prod([spins[site] for site, _ in factors])
                         for w, factors in terms) for terms in TERMS])


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    compiler = shutil.which("c++")
    if not compiler:
        pytest.skip("C++ compiler unavailable")
    work = tmp_path_factory.mktemp("figure-probe")
    source = work / "probe.cpp"
    source.write_text(r'''
#include "FigureObjective.hpp"
#include <iostream>
#include <iomanip>
int main(int argc, char** argv) {
    FigureObjective f;
    f.load(argv[1], 4, 2);
    std::cout << std::setprecision(17);
    for (int mask = 0; mask < 16; ++mask) {
        std::vector<int> occ(4);
        for (int i = 0; i < 4; ++i) occ[i] = (mask >> i) & 1;
        auto base = f.correlations(occ);
        for (int a = 0; a < 4; ++a) for (int b = 0; b < 4; ++b) {
            auto proposal = f.swapped(occ, base, a, b);
            std::cout << mask << ' ' << a << ' ' << b;
            for (double x : proposal) std::cout << ' ' << x;
            std::cout << '\n';
        }
    }
    // Mixed-order weights normalize globally, retaining the order penalty.
    f.wd = 0.7; f.wn = 0.5;
    std::cout << "score " << f.score({0.2, -0.4, 0.6}) << '\n';
    f.mode = "atat";
    std::cout << "atat " << f.score({0.2, -0.4, 0.6}) << '\n';
    std::cout << "range " << f.score({0.0, -0.4, 0.6}) << '\n';
    std::cout << "matched " << f.score({0.0, 0.0, 0.0}) << '\n';
    f.wd = 20.0;
    std::cout << "decay " << f.score({0.2, -0.4, 0.6}) << '\n';
}
''')
    exe = work / "probe"
    subprocess.run([compiler, "-std=c++17", "-I", str(REPO / "SCRAPs"),
                    str(source), "-o", str(exe)], check=True, capture_output=True)
    return exe


def test_all_binary_swaps_repeated_images_and_normalization(probe, tmp_path):
    figures = tmp_path / "figures.txt"
    write_figures(figures)
    lines = subprocess.check_output([str(probe), str(figures)], text=True).splitlines()
    assert len(lines) == 261
    for line in lines[:256]:
        mask, a, b, *actual = map(float, line.split())
        occ = [(int(mask) >> i) & 1 for i in range(4)]
        a, b = int(a), int(b)
        occ[a], occ[b] = occ[b], occ[a]
        assert np.allclose(actual, reference(occ), atol=1e-14)
    weights = np.exp(-0.7 * np.array([1, 2, 3])) * np.array([1, 0.5, 0.25])
    expected = np.dot(weights, [0.2, 0.4, 0.6]) / weights.sum()
    assert float(lines[256].split()[1]) == pytest.approx(expected, abs=1e-14)
    # Golden values from the unmodified ATAT 3.52 calc_objective_func,
    # zero_tolerance=0.001; the last case is NOT an exact correlation match.
    expected_atat = [-1.4932660126550767, -3.0602188394654668,
                     -3.4028234663852886e38, -3.4028234663852886e38]
    for line, expected in zip(lines[257:], expected_atat):
        assert float(line.split()[1]) == pytest.approx(expected, rel=1e-14)



requires_scraps = pytest.mark.skipif(not SCRAPS_BIN.exists() or MPIRUN is None,
                                    reason="scraps binary not built or mpirun missing")


@requires_scraps
@pytest.mark.parametrize("objective", ["weighted_l1", "atat"])
@pytest.mark.parametrize("kernel,ranks,fast", [("cuckoo", 1, 1), ("cuckoo", 2, 0),
                                               ("sa", 1, 1), ("pt", 2, 1)])
def test_native_figures_mpi_and_output(tmp_path, kernel, ranks, fast, objective):
    write_figures(tmp_path / "figures.txt", n=16)
    # Seed 47 fixes initialization/proposals; check observables, not a trajectory.
    inp = GOOD_INPUT + (f"RNG_SEED 47\nFIGURES_FILE figures.txt\n"
                        f"MC_KERNEL {kernel}\nFAST_SWAPS {fast}\nFIGURE_OBJECTIVE {objective}\n"
                        "CLUSTER_SWAP_K 2\nFIGURE_WD 0.7\nFIGURE_WN 0.5\n")
    rc, log = run_scraps(tmp_path, inp, ranks=ranks)
    assert rc == 0, log[-3000:]
    assert "unrecognized INPUT keyword" not in log
    occupations = np.loadtxt(tmp_path / "FIGURE_OCCUPATIONS.txt", dtype=int)
    assert np.bincount(occupations).tolist() == [8, 8]
    with (tmp_path / "FIGURE_CORRELATIONS.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert np.allclose([float(r["correlation"]) for r in rows], reference(occupations), atol=1e-14)
    assert "all components within FIGURE_TOL" in log


@requires_scraps
@pytest.mark.parametrize("extra,diagnostic", [
    ("FIGURE_WN 0\n", "FIGURE parameters"),
    ("FIGURE_WD 1\n", "require FIGURES_FILE"),
    ("FIGURE_OBJECTIVE nonsense\n", "FIGURE_OBJECTIVE"),
    ("FIGURES_FILE\n", "FIGURES_FILE requires"),
    ("FIGURES_FILE missing.txt\n", "FIGURES_FILE"),
    ("FIGURES_FILE figures.txt\nTARGETED_SWAPS 1\n", "TARGETED_SWAPS"),
    ("FIGURES_FILE figures.txt\nSRO 0 1 0 0\n", "replaces SRO"),
])
def test_figures_input_rejected(tmp_path, extra, diagnostic):
    rc, log = run_scraps(tmp_path, GOOD_INPUT + extra)
    assert rc != 0
    assert diagnostic in log


@requires_scraps
def test_figure_geometry_mismatch_rejected(tmp_path):
    write_figures(tmp_path / "figures.txt", n=16)
    p = tmp_path / "figures.txt"
    p.write_text(p.read_text().replace("2.0 0.0 0.0", "3.0 0.0 0.0", 1))
    rc, log = run_scraps(tmp_path, GOOD_INPUT + "FIGURES_FILE figures.txt\n")
    assert rc != 0
    assert "cell differs" in log


@requires_scraps
def test_sa_archive_preserves_best_recorded_structure(tmp_path):
    write_figures(tmp_path / "figures.txt", n=16)
    inp = GOOD_INPUT + (
        "FIGURES_FILE figures.txt\nRNG_SEED 103\nMC_KERNEL sa\n"
        "SA_T_INITIAL 10\nSA_T_FINAL 10\nCONVERGENCE_LOG progress.csv\n")
    rc, log = run_scraps(tmp_path, inp, ranks=2)
    assert rc == 0, log[-3000:]
    occupations = np.loadtxt(tmp_path / "FIGURE_OCCUPATIONS.txt", dtype=int)
    final_score = np.abs(reference(occupations)).mean()
    with (tmp_path / "progress.csv").open() as stream:
        records = list(csv.reader(stream))
    logged_scores = [float(row[1]) for row in records[1:]]
    assert final_score <= min(logged_scores) + 1e-6


def write_shared_figures(path, counts=((6,2),(2,6)), groups=None):
    """Version 3 shares physical binary species but records distinct budgets."""
    write_figures(path,n=16)
    lines = path.read_text().splitlines()
    lines[0] = 'SCRAPS_FIGURES 3'
    # Existing helper packs each cell into one row: header, shape, cell, sites, basis.
    insert = 3+16+1
    allowed = ['2 0 1']*16
    identities = [i//8 for i in range(16)] if groups is None else groups
    budgets = [f'{identities[i]} {counts[i//8][0]} {counts[i//8][1]}' for i in range(16)]
    lines[insert:insert] = allowed+budgets
    path.write_text('\n'.join(lines)+'\n')


SHARED_PARTITION = ('SUBLATTICE 7 BASIS 0 SPECIES 0 1 COUNTS 6 2\n'
                    'SUBLATTICE 9 BASIS 1 SPECIES 0 1 COUNTS 2 6\n')


@requires_scraps
@pytest.mark.parametrize('kernel,fast', [('cuckoo',0),('cuckoo',1),('sa',1),('pt',1)])
def test_shared_species_local_budgets(tmp_path,kernel,fast):
    write_shared_figures(tmp_path/'figures.txt')
    rc,log = run_scraps(tmp_path,GOOD_INPUT+SHARED_PARTITION+
                        f'FIGURES_FILE figures.txt\nMC_KERNEL {kernel}\nFAST_SWAPS {fast}\n'
                        'RNG_SEED 47\nCLUSTER_SWAP_K 3\n',ranks=2)
    assert rc == 0, log[-4000:]
    occ = np.loadtxt(tmp_path/'FIGURE_OCCUPATIONS.txt',dtype=int)
    assert np.bincount(occ[:8],minlength=2).tolist() == [6,2]
    assert np.bincount(occ[8:],minlength=2).tolist() == [2,6]
    rows = np.genfromtxt(tmp_path/'FIGURE_CORRELATIONS.csv',delimiter=',',names=True)
    np.testing.assert_allclose(rows['correlation'],reference(occ),atol=1e-12)


@requires_scraps
@pytest.mark.parametrize('partition,extra,diagnostic', [
    (SHARED_PARTITION,'','COUNTS currently requires FIGURES_FILE'),
    (SHARED_PARTITION.replace('COUNTS 2 6',''),'FIGURES_FILE figures.txt\n','every group'),
    (SHARED_PARTITION.replace('COUNTS 2 6','COUNTS -1 9'),'FIGURES_FILE figures.txt\n','nonnegative'),
    (SHARED_PARTITION.replace('COUNTS 2 6','COUNTS 1 6'),'FIGURES_FILE figures.txt\n','capacity'),
    (SHARED_PARTITION.replace('COUNTS 2 6','COUNTS 3 5'),'FIGURES_FILE figures.txt\n','plus fixed sites must equal ELEMENT'),
])
def test_shared_species_invalid_counts(tmp_path,partition,extra,diagnostic):
    write_shared_figures(tmp_path/'figures.txt')
    rc,log = run_scraps(tmp_path,GOOD_INPUT+partition+extra)
    assert rc != 0
    assert diagnostic in log


@requires_scraps
@pytest.mark.parametrize('damage', ['counts','groups','version','warm_start'])
def test_shared_species_stale_targets_and_warm_start(tmp_path,damage):
    write_shared_figures(tmp_path/'figures.txt',counts=((5,3),(3,5)) if damage == 'counts' else ((6,2),(2,6)),
                         groups=[0]*16 if damage == 'groups' else None)
    extra = ''
    if damage == 'version':
        p = tmp_path/'figures.txt'
        lines = p.read_text().splitlines()
        lines[0] = 'SCRAPS_FIGURES 2'
        del lines[36:52]
        p.write_text('\n'.join(lines)+'\n')
    if damage == 'warm_start':
        (tmp_path/'seed.txt').write_text(' '.join(map(str,[0]*4+[1]*4+[0]*4+[1]*4)))
        extra = 'INIT_NEST seed.txt\n'
    rc,log = run_scraps(tmp_path,GOOD_INPUT+SHARED_PARTITION+'FIGURES_FILE figures.txt\n'+extra)
    assert rc != 0
    diagnostic = ('requires FIGURES_FILE format version 3' if damage == 'version' else
                  'INIT_NEST violates SUBLATTICE species counts' if damage == 'warm_start' else
                  'group/count metadata disagree')
    assert diagnostic in log


@requires_scraps
@pytest.mark.parametrize('kind', ['partial_overlap','fixed_reuse'])
def test_shared_species_partial_and_fixed_reuse(tmp_path,kind):
    p = tmp_path/'figures.txt'
    write_figures(p,n=16)
    lines = p.read_text().splitlines()
    lines[0] = 'SCRAPS_FIGURES 3'
    if kind == 'partial_overlap':
        lines[1] = '16 3 1 3'
        lines[19] = '-1 1 0'
        allowed = ['2 0 1']*8+['2 1 2']*8
        budgets = ['0 6 2 0']*8+['1 0 2 6']*8
        import re
        inp = re.sub(r'TYP\s+2', 'TYP 3', GOOD_INPUT).replace('ELEMENT 8 8','ELEMENT 6 4 6')
        partition = ('SUBLATTICE 0 BASIS 0 SPECIES 0 1 COUNTS 6 2\n'
                     'SUBLATTICE 1 BASIS 1 SPECIES 1 2 COUNTS 2 6\n')
        expected = [[6,2,0],[0,2,6]]
        values = np.array([-1,1,0])
    else:
        allowed = ['2 0 1']*8+['1 1']*8
        budgets = ['0 2 6']*8+['-1 0 1']*8
        inp = GOOD_INPUT.replace('ELEMENT 8 8','ELEMENT 2 14')
        partition = ('SUBLATTICE 0 BASIS 0 SPECIES 0 1 COUNTS 2 6\n'
                     'SPECTATOR_BASIS_SPECIES 1 1\n')
        expected = [[2,6],[0,8]]
        values = np.array([-1,1])
    lines[20:20] = allowed+budgets
    p.write_text('\n'.join(lines)+'\n')
    rc,log = run_scraps(tmp_path,inp+partition+'FIGURES_FILE figures.txt\nRNG_SEED 47\n',ranks=2)
    assert rc == 0,log[-4000:]
    occ = np.loadtxt(tmp_path/'FIGURE_OCCUPATIONS.txt',dtype=int)
    for group in (0,1):
        assert np.bincount(occ[group*8:(group+1)*8],minlength=len(values)).tolist() == expected[group]
    reference_values = [sum(w*np.prod([values[occ[site]] for site,_ in factors]) for w,factors in terms)
                        for terms in TERMS]
    actual = np.genfromtxt(tmp_path/'FIGURE_CORRELATIONS.csv',delimiter=',',names=True)['correlation']
    np.testing.assert_allclose(actual,reference_values,atol=1e-12)
