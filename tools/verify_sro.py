#!/usr/bin/env python3
"""
Independent Warren-Cowley α verification from a SCRAPS.vasp supercell.
Reads VASP-format POSCAR, finds first-shell neighbors by direct distance,
computes alpha_ij = 1 - P_ij/c_j for every species pair.
"""
import sys
import numpy as np
from pathlib import Path

VASP = Path(sys.argv[1])
SPECIES = sys.argv[2:] if len(sys.argv) > 2 else None

with VASP.open() as f:
    lines = [l.rstrip() for l in f]

scale = float(lines[1])
lat = np.array([list(map(float, lines[i].split()[:3])) for i in (2, 3, 4)]) * scale
counts = [int(x) for x in lines[5].split()]
n_total = sum(counts)
mode = lines[6].strip().lower()[0]
coord_lines = lines[7:7 + n_total]
coords = np.array([list(map(float, l.split()[:3])) for l in coord_lines])
if mode == 'd':
    coords = coords @ lat
elif mode == 'c':
    pass
else:
    raise SystemExit(f"unknown coord mode {mode}")

species_of = np.repeat(np.arange(len(counts)), counts)
n_typ = len(counts)
names = SPECIES if SPECIES and len(SPECIES) == n_typ else [f"sp{i}" for i in range(n_typ)]

inv_lat = np.linalg.inv(lat)
def min_image_dist2(a, b):
    d = a - b
    df = d @ inv_lat
    df -= np.round(df)
    dc = df @ lat
    return float(dc @ dc)

# Find shell-1 cutoff from histogram of all pairwise distances.
dists2 = []
for i in range(n_total):
    for j in range(i + 1, n_total):
        dists2.append(min_image_dist2(coords[i], coords[j]))
dists2 = np.array(sorted(dists2))
unique_d = np.unique(np.round(np.sqrt(dists2), 4))
print(f"Unique distance shells (first 6): {unique_d[:6]}")
shell1_max = unique_d[0] * 1.05  # 5% tolerance

# Build first-shell neighbor lists.
neighbors = [[] for _ in range(n_total)]
for i in range(n_total):
    for j in range(n_total):
        if i == j:
            continue
        if min_image_dist2(coords[i], coords[j]) <= shell1_max ** 2:
            neighbors[i].append(j)

z = len(neighbors[0])
print(f"Coordination number (shell 1): {z}")
assert all(len(nl) == z for nl in neighbors), "non-uniform coordination — check shell cutoff"

# count_ij = number of (μ=i, ν=j) ordered nearest-neighbor pairs
count = np.zeros((n_typ, n_typ), dtype=int)
for i in range(n_total):
    si = species_of[i]
    for j in neighbors[i]:
        sj = species_of[j]
        count[si][sj] += 1

# Warren-Cowley alpha: α_ij = 1 - P(j|i)/c_j
# P(j|i) = count[i][j] / (z * N_i)
c = np.array(counts) / n_total
alpha = np.zeros((n_typ, n_typ))
for i in range(n_typ):
    for j in range(n_typ):
        Pji = count[i][j] / (z * counts[i])
        alpha[i][j] = 1 - Pji / c[j]

print(f"\nAlpha matrix (Warren-Cowley, shell 1):")
print(f"             {'  '.join(f'{n:>8}' for n in names)}")
for i, n in enumerate(names):
    row = '  '.join(f'{alpha[i][j]:+.4f}' for j in range(n_typ))
    print(f"  {n:>3}      {row}")

# Asymmetry check: α_ij and α_ji should match in a finite cell only if
# c_i * α_ij = c_j * α_ji (i.e., count is symmetric).
print(f"\nSymmetrized alpha (average of α_ij and α_ji):")
print(f"             {'  '.join(f'{n:>8}' for n in names)}")
sym = 0.5 * (alpha + alpha.T)
for i, n in enumerate(names):
    row = '  '.join(f'{sym[i][j]:+.4f}' for j in range(n_typ))
    print(f"  {n:>3}      {row}")
