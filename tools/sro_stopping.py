"""Exact, independently checked stopping certificates for fixed-cell SRO search."""
from collections import Counter
from copy import deepcopy
from fractions import Fraction
import hashlib

import numpy as np
from ase.neighborlist import neighbor_list

from sro_bounds import KEYS, sro_bounds
from structure_analysis import analyze_structure


class SROBoundStop:
    """Call with a candidate ASE Atoms; True means a proven fixed-cell minimum.

    Candidate sites must retain the input geometry and order. Only occupations
    may change, preserving each sublattice's composition. RMS exact values are
    squared errors. No floating-point score or tolerance can trigger stopping.
    """

    def __init__(self, atoms, *, cutoff, targets, shells=2, sublattices=None,
                 metric='max_error'):
        self.atoms = atoms.copy()
        self.metric = metric
        self.bound = sro_bounds(atoms, cutoff=cutoff, targets=targets, shells=shells,
                                sublattices=sublattices, metric=metric)
        self.certificate = None
        self.witness = None
        self.labels = np.array(sublattices['site_labels'] if sublattices else ['all'] * len(atoms))
        symbols = np.array(atoms.get_chemical_symbols())
        self.counts = {g: Counter(symbols[self.labels == g]) for g in set(self.labels)}
        analysis = analyze_structure(atoms, cutoff=cutoff, shells=shells,
                                     sublattices=sublattices, sro_targets=targets)
        rows = {tuple(r[k] for k in KEYS): r for r in analysis['sro']}
        ii, jj, distances = neighbor_list('ijd', atoms, cutoff)
        self.prepared = []
        for entry, target in zip(self.bound['entries'], targets):
            key = tuple(target[k] for k in KEYS)
            row = rows[key]
            mask = ((self.labels[ii] == key[0]) & (self.labels[jj] == key[1]) &
                    (distances >= row['distance_min']) & (distances <= row['distance_max']))
            self.prepared.append((ii[mask], jj[mask], key[3], key[4],
                                  Fraction(entry['denominator_exact']), Fraction(str(target['target']))))

    def __call__(self, candidate):
        if (len(candidate) != len(self.atoms) or
                not np.array_equal(candidate.pbc, self.atoms.pbc) or
                not np.array_equal(candidate.cell, self.atoms.cell) or
                not np.array_equal(candidate.positions, self.atoms.positions)):
            raise ValueError('Stopping certificate requires identical fixed geometry and site order')
        symbols = np.array(candidate.get_chemical_symbols())
        if any(Counter(symbols[self.labels == g]) != counts for g, counts in self.counts.items()):
            raise ValueError('Stopping certificate requires the original sublattice compositions')
        counts = [int(np.count_nonzero((symbols[i] == a) & (symbols[j] == b)))
                  for i, j, a, b, d, t in self.prepared]
        errors = [abs(1 - Fraction(n) / d - t) for n, (_, _, _, _, d, t) in zip(counts, self.prepared)]
        actual = (max(errors) if self.metric == 'max_error' else
                  sum(e * e for e in errors) / len(errors) if self.metric == 'rms_error' else
                  sum(errors) / len(errors))
        lower = Fraction(self.bound['result']['lower_bound_exact'])
        if actual < lower:
            raise RuntimeError('Verified witness violates the lower bound')
        if actual != lower:
            return False
        self.witness = candidate.copy()
        self.certificate = dict(status='certified_minimum', basis='bound_attained',
            objective='directed_pair_sro', metric=self.metric,
            lower_bound_exact=str(lower), achieved_error_exact=str(actual),
            exact_value_convention=self.bound['result']['exact_value_convention'],
            pair_counts=counts, bound_report_for_initial_structure=deepcopy(self.bound),
            occupations_sha256=hashlib.sha256(','.join(symbols).encode()).hexdigest())
        return True
