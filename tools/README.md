# tools/ — verification utilities

## `verify_sro.py`

Independent Warren-Cowley α verifier for SCRAPS supercells. Computes
α_ij per neighbor shell directly from atomic positions using
pymatgen — no dependence on SCRAPS's internal accounting.

### Why a separate verifier

The upstream SCRAPS `printtoscreen()` used a buggy α formula
(off by a factor of ~N_i, asymmetric). This fork patched it, but
having an independent computation is still useful for:

- regression-testing after any code change
- cross-validating SCRAPS output against textbook reference structures
- driving CI-style sweeps that compare achieved-vs-target across
  many configurations

### Install

```bash
pip install pymatgen
```

Numpy is a transitive dep.

### Usage

```bash
# Basic — print alpha matrices for the first 3 shells
verify_sro.py SCRAPS.vasp --species Ni,Co,Cr

# Just the nearest-neighbor shell
verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --shells 1

# Compare against the same INPUT file you fed SCRAPS
verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --target INPUT \
    --tolerance 0.02

# Machine-readable output for CI / regression sweeps
verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --target INPUT --json
```

The `--species` flag is required for SCRAPS-generated POSCARs because
SCRAPS writes count blocks without symbol labels. For standard
VASP 5+ POSCARs that already have a species line, `--species` is
optional.

### CLI flags

| Flag | Default | Meaning |
| --- | --- | --- |
| `poscar` (positional) | — | Path to the POSCAR file |
| `--species` | (read from POSCAR) | Comma-separated symbols in POSCAR-block order |
| `--shells N` | 3 | Number of neighbor shells to compute |
| `--cutoff R` | smallest lattice vector / 2 | Max neighbor distance (Å) |
| `--rel-gap T` | 0.05 | Shell-clustering relative-gap threshold |
| `--target FILE` | — | Target α file (SCRAPS INPUT-style) |
| `--tolerance T` | 0.02 | `|achieved − target|` pass/fail threshold |
| `--json` | (off) | Emit JSON instead of human table |
| `--spectator-species LIST` | — | Comma-separated species to treat as spectator (e.g. `B` for HEB-type AlB₂ diborides) — α is computed M-M only, with per-sublattice reference c_j^M |

### Target file format

The same `SRO` lines used by the SCRAPS INPUT file:

```
SRO  <species_i>  <species_j>  <shell_index>  <alpha_target>
```

Other tokens are ignored, so you can pass a SCRAPS INPUT file directly
(`--target INPUT`) without preprocessing. Comments after `#` are
stripped per line.

### Exit codes

| Code | Meaning |
| --- | --- |
| 0 | All target pairs within tolerance (or no `--target` given) |
| 1 | At least one target pair failed |
| 2 | Argument error / file not found / pymatgen missing |

### Example output

```
$ verify_sro.py SCRAPS.vasp --species Ni,Co,Cr --target INPUT --shells 1
Lattice: (3.0, 3.0, 3.0) (Angstrom), angles (90.0, 90.0, 90.0)
Atoms: 108 total, species ['Ni', 'Co', 'Cr']
Detected shells: 1
  shell 0: r ∈ [0.7071, 0.7071], z = 12

=== Shell 0 (z = 12) ===
                Ni       Co       Cr
    Ni  +0.0972 +0.0000 -0.0972
    Co  +0.0000 +0.0556 -0.0556
    Cr  -0.0972 -0.0556 +0.1528

Target comparison (tolerance = 0.02):
  pair          target  achieved    |err|   status
  Ni-Cr        -0.1000   -0.0972   0.0028   PASS
  Co-Cr        -0.0500   -0.0556   0.0056   PASS
```

### Tests

```bash
pytest tools/tests/test_verify_sro.py
```

The test suite covers:

- **B2 NiAl** (`make_b2_NiAl`): perfect ordering, α(Ni-Al) = −1 exactly
- **L1₂ Ni₃Al** (`make_L12_Ni3Al`): partial ordering with textbook α matrix
  (Ni-Ni = +1/9, Ni-Al = Al-Ni = −1/3, Al-Al = +1)
- Alpha-matrix symmetry across multiple shells
- Multi-shell detection on BCC (z = 8, 6, 12 at distances a√3/2, a, a√2)
- Target file parser (well-formed, comments, malformed lines)
- SCRAPS round-trip: random NiCoCr should give max|α| < 0.05; CSRO-targeted
  NiCoCr should match its INPUT within tolerance 0.03
- **HEB round-trip (spectator-aware)**: random (Hf,Zr,Ti,Ta,Nb)B₂ should
  give max|α| < 0.01 on the M-M shell when B is treated as spectator;
  CSRO-targeted version should match its target within 0.05

Tests requiring SCRAPS output gracefully skip if `SCRAPS.vasp` hasn't
been generated yet.
