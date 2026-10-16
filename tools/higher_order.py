"""Export icet 4 cluster functions for SCRAPs' native figure objective.

icet is optional and imported only when exporting. The exported basis uses
SCRAPs species indices, irrespective of icet's atomic-number ordering.
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

import numpy as np
from ase import Atoms
from ase.data import atomic_numbers


def scraps_supercell(atoms, cell_dim):
    """Construct basis-major, z/y/x-minor sites exactly as Lattice::initialize."""
    dims = np.asarray(cell_dim)
    if dims.shape != (3,) or np.any(dims < 1) or np.any(dims != dims.astype(int)):
        raise ValueError("cell_dim must contain three positive integers")
    cell = np.asarray(atoms.cell)
    if not np.all(atoms.pbc) or abs(np.linalg.det(cell)) < 1e-12:
        raise ValueError("figure export requires a nonsingular, fully periodic cell")
    positions = [r + i*cell[0] + j*cell[1] + k*cell[2]
                 for r in atoms.positions
                 for k in range(int(dims[2])) for j in range(int(dims[1]))
                 for i in range(int(dims[0]))]
    return Atoms([atoms[0].symbol]*len(positions), positions=positions,
                 cell=cell*dims[:, None], pbc=True)


def evaluate_figures(model, occupations):
    """Independent full recomputation, occupations in SCRAPs site/species order."""
    basis = np.asarray(model['basis'])
    occ = np.asarray(occupations, dtype=int)
    return np.array([sum(term['coefficient'] * np.prod(
        [basis[b, occ[s]] for s, b in term['factors']]) for term in f['terms'])
        for f in model['features']])


def _site_alphabets(nbasis, counts, ncell, sublattices, spectators):
    """Validate local capacities and preserve independently constrained groups."""
    allowed = [None] * nbasis
    basis_groups = [None] * nbasis
    fixed = {}
    for bi, sp in spectators or []:
        if not 0 <= bi < nbasis or not 0 <= sp < len(counts) or bi in fixed:
            raise ValueError("invalid or duplicate spectator basis/species index")
        fixed[bi] = sp
        allowed[bi] = [sp]
    fixed_counts = Counter(fixed.values())
    declarations = list(sublattices or [])
    if any(len(g) not in (3, 4) for g in declarations):
        raise ValueError("sublattice tuples require id, basis, species and optional counts")
    explicit = any(len(g) == 4 for g in declarations)
    if explicit and not all(len(g) == 4 for g in declarations):
        raise ValueError("explicit counts are required on every declared sublattice")
    if not explicit:
        declared_usage = Counter(s for g in declarations for s in g[2])
        if any(n > 1 or s in fixed_counts for s,n in declared_usage.items()):
            raise ValueError("overlapping species require explicit counts; legacy sets must be disjoint")
        if any(counts[sp] != n*ncell for sp,n in fixed_counts.items()):
            raise ValueError("spectator ELEMENT count must exactly match its fixed sites")
    if not declarations:
        declarations = [(0, [b for b in range(nbasis) if b not in fixed],
                         [s for s in range(len(counts)) if s not in fixed_counts])]
    groups, ids, used_species = [], set(), set(fixed_counts)
    allocated = np.array([fixed_counts[s]*ncell for s in range(len(counts))], dtype=int)
    for declaration in declarations:
        sid, sites, local_species = declaration[:3]
        sites, local_species = list(sites), list(local_species)
        if sid in ids or sid < 0 or not sites or not local_species:
            raise ValueError("sublattice IDs must be distinct and groups nonempty")
        ids.add(sid)
        if len(set(sites)) != len(sites) or len(set(local_species)) != len(local_species):
            raise ValueError("duplicate sublattice basis or species index")
        if any(s < 0 or s >= len(counts) for s in local_species):
            raise ValueError("invalid sublattice species index")
        if not explicit and any(s in used_species for s in local_species):
            raise ValueError("overlapping species require explicit counts; legacy sets must be disjoint")
        if any(b < 0 or b >= nbasis or allowed[b] is not None for b in sites):
            raise ValueError("sublattice basis sets must be valid and disjoint")
        local_counts = np.asarray(declaration[3] if explicit else counts[local_species])
        if (local_counts.shape != (len(local_species),) or
                not np.all(np.isfinite(local_counts)) or np.any(local_counts < 0) or
                np.any(local_counts != local_counts.astype(int))):
            raise ValueError("sublattice counts must be nonnegative integers aligned with species")
        if sum(local_counts) != len(sites)*ncell:
            raise ValueError("sublattice counts do not match site capacity")
        gid = len(groups)
        groups.append(dict(id=int(sid), basis=sites, species=local_species,
                           counts=local_counts.astype(int).tolist(), fixed=False))
        allocated[local_species] += local_counts.astype(int)
        used_species.update(local_species)
        for b in sites:
            allowed[b] = sorted(local_species)
            basis_groups[b] = gid
    for sp in sorted(fixed_counts):
        sites = [b for b, s in fixed.items() if s == sp]
        gid = len(groups)
        groups.append(dict(id=None, basis=sites, species=[sp],
                           counts=[len(sites)*ncell], fixed=True))
        for b in sites:
            basis_groups[b] = gid
    if any(a is None for a in allowed) or used_species != set(range(len(counts))):
        raise ValueError("sublattices must cover every non-spectator site and species")
    if not np.array_equal(allocated, counts):
        raise ValueError("sublattice counts plus spectator allocation must equal global ELEMENT counts")
    return allowed, basis_groups, groups


def export_figures(path, atoms, species, element_counts, cell_dim, cutoffs,
                   sublattices=None, spectator_basis_species=None):
    """Write FIGURES plus JSON provenance; return the complete figure model.

    Cutoffs are maximum intersite distances (Angstrom), for pairs, triplets,
    and optionally quadruplets. Targets describe independent random-site
    occupations, not sampling without replacement at fixed finite composition.
    Periodically repeated site IDs are retained as repeated factors.
    Independently constrained sublattices retain distinct symmetry labels even
    when their physical species alphabets overlap. Four-member sublattice
    tuples supply local counts aligned with their species indices; all groups
    must supply counts when any does. Fixed sites have no nonconstant point functions, so icet
    eliminates their redundant clusters; their occupations remain fixed.
    """
    try:
        import icet
        from icet import ClusterSpace
    except ImportError as exc:
        raise ImportError("Higher-order export requires optional dependency icet==4.0") from exc
    if icet.__version__.split('.')[0] != '4':
        raise ValueError(f"Figure exporter requires icet 4.x; found {icet.__version__}")
    species = list(species)
    counts = np.asarray(element_counts)
    if len(species) < 2 or len(set(species)) != len(species):
        raise ValueError("figure_species must contain at least two distinct chemical symbols")
    if any(s not in atomic_numbers for s in species):
        raise ValueError("figure_species must be valid chemical symbols")
    cell = scraps_supercell(atoms, cell_dim)
    if (counts.shape != (len(species),) or np.any(counts <= 0)
            or np.any(counts != counts.astype(int)) or sum(counts) != len(cell)):
        raise ValueError("element_counts must be positive integers matching species and supercell")
    if not 1 <= len(cutoffs) <= 3 or not np.all(np.isfinite(cutoffs)) or min(cutoffs) <= 0:
        raise ValueError("figure_cutoffs requires one to three positive finite distances")
    ncell = int(np.prod(cell_dim))
    basis_allowed, basis_groups, groups = _site_alphabets(
        len(atoms), counts, ncell, sublattices, spectator_basis_species)
    site_allowed = [a for a in basis_allowed for _ in range(ncell)]
    site_groups = [g for g in basis_groups for _ in range(ncell)]
    usage = Counter(s for g in groups for s in g['species'])
    use_auxiliary = any(n > 1 for n in usage.values())
    # Distinct auxiliary alphabets prevent icet from merging groups with the
    # same physical species but independently specified compositions.
    from ase.data import chemical_symbols
    label_maps, next_label = [], 1
    for group in groups:
        ordered = sorted(group['species'], key=lambda s: atomic_numbers[species[s]])
        if use_auxiliary:
            if next_label + len(ordered) > len(chemical_symbols):
                raise ValueError("too many sublattice-local species for auxiliary symmetry labels")
            labels = chemical_symbols[next_label:next_label+len(ordered)]
            next_label += len(ordered)
        else:
            labels = [species[s] for s in ordered]
        label_maps.append(dict(zip(ordered, labels)))
        group['symmetry_labels'] = [label_maps[-1][s] for s in group['species']]
    primitive = atoms.copy()
    primitive.set_chemical_symbols([label_maps[g][a[0]] for g,a in zip(basis_groups,basis_allowed)])
    cell.set_chemical_symbols([label_maps[g][a[0]] for g,a in zip(site_groups,site_allowed)])
    cs = ClusterSpace(primitive, list(cutoffs),
                      [[label_maps[g][s] for s in a] for g,a in zip(basis_groups,basis_allowed)])
    basis, moments, row_lookup = [], [], {}
    for gid, group in enumerate(groups):
        alphabet = group['species']
        species_order = sorted(alphabet, key=lambda s: atomic_numbers[species[s]])
        concentration = np.asarray(group['counts']) / sum(group['counts'])
        for k in range(len(alphabet)-1):
            values = np.zeros(len(species))
            for s in alphabet:
                values[s] = cs.evaluate_cluster_function(len(alphabet), k, species_order.index(s))
            row_lookup[gid, k] = len(basis)
            basis.append(values)
            moments.append(float(values[alphabet] @ concentration))
    basis = np.asarray(basis)
    moments = np.asarray(moments)
    orbits = cs.orbit_list.get_supercell_orbit_list(cell, 1e-6)
    features = []
    for row in cs.as_list:
        if row['order'] < 2:
            continue
        orbit = orbits[row['orbit_index']]
        vector = row['multicomponent_vector']
        decoration = next(x for x in orbit.cluster_vector_elements
                          if list(x['multicomponent_vector']) == list(vector))
        terms = Counter()
        for cluster in orbit.clusters:
            sites = [s.index for s in cluster.lattice_sites]
            for permutation in decoration['site_permutations']:
                # Keep factors repeated when two periodic images map to one site.
                factors = tuple(sorted((sites[k], row_lookup[site_groups[sites[k]],
                                                              vector[permutation[k]]])
                                       for k in range(len(vector))))
                terms[factors] += 1
        nterms = sum(terms.values())
        coords = np.asarray(cs.get_coordinates_of_representative_cluster(row['orbit_index']))
        diameter = float(np.linalg.norm(coords[:, None] - coords[None, :], axis=2).max())
        features.append(dict(index=row['index'], orbit_index=row['orbit_index'],
                             order=row['order'], diameter=diameter,
                             multicomponent_vector=list(vector), multiplicity=row['multiplicity'],
                             target=float(sum(n/nterms * np.prod([moments[b] for _, b in factors])
                                              for factors, n in terms.items())),
                             terms=[dict(coefficient=n/nterms, factors=list(factors))
                                    for factors, n in sorted(terms.items())]))
    if not features or not any(f['order'] == 2 for f in features):
        raise ValueError("figure_cutoffs must include at least one pair orbit")
    version = (3 if any(len(g) == 4 for g in sublattices or []) else
               2 if sublattices or spectator_basis_species else 1)
    model = dict(format='SCRAPS_FIGURES', version=version, icet_version=icet.__version__,
                 species=species, element_counts=counts.astype(int).tolist(),
                 cutoffs=list(cutoffs), cell=cell.cell.tolist(),
                 fractional_sites=cell.get_scaled_positions(wrap=False).tolist(),
                 basis=basis.tolist(), features=features,
                 basis_allowed_species=basis_allowed, site_allowed_species=site_allowed,
                 basis_moments=moments.tolist(), sublattice_groups=groups,
                 site_groups=site_groups, uses_auxiliary_labels=use_auxiliary,
                 target_convention='independent random-site occupations',
                 self_interacting=bool(cs.is_supercell_self_interacting(cell)))
    # Check several asymmetric occupations against icet before accepting an export.
    rng = np.random.default_rng(86421)
    for _ in range(3):
        occ = np.array([rng.choice(a) for a in site_allowed])
        sample = cell.copy()
        sample.set_chemical_symbols([label_maps[g][int(i)] for g,i in zip(site_groups,occ)])
        reference = cs.get_cluster_vector(sample)[[f['index'] for f in features]]
        np.testing.assert_allclose(evaluate_figures(model, occ), reference, atol=2e-10, rtol=0)
    lines = [f'SCRAPS_FIGURES {version}', f'{len(cell)} {len(species)} {len(basis)} {len(features)}']
    for rows in (model['cell'], model['fractional_sites'], model['basis']):
        lines.extend(' '.join(format(x, '.17g') for x in r) for r in rows)
    if version >= 2:
        lines.extend(str(len(a)) + ' ' + ' '.join(map(str, a)) for a in site_allowed)
    if version == 3:
        for gid in site_groups:
            group = groups[gid]
            local = [0]*len(species)
            for sp, n in zip(group['species'], group['counts']):
                local[sp] = 1 if group['fixed'] else n
            lines.append(str(-1 if group['fixed'] else gid) + ' ' + ' '.join(map(str, local)))
    for f in features:
        lines.append(f"{f['order']} {f['diameter']:.17g} {f['target']:.17g} {len(f['terms'])}")
        for t in f['terms']:
            lines.append(f"{t['coefficient']:.17g} " + ' '.join(f'{s} {b}' for s,b in t['factors']))
    path = Path(path)
    path.write_text('\n'.join(lines)+'\n')
    path.with_suffix(path.suffix+'.json').write_text(json.dumps(model, indent=2)+'\n')
    return model
