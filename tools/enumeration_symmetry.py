"""Conservative site symmetries and fixed-composition orbit representatives."""
from itertools import combinations

import numpy as np
import spglib


def permutation_group(generators, sites, limit=4096):
    """Close verified generators under composition; include identity/inverses."""
    identity = tuple(range(sites))
    group, selected = {identity}, []
    for generator in generators:
        generator = tuple(generator)
        if generator in group:
            continue
        selected.append(generator)
        pending = list(group)
        while pending:
            element = pending.pop()
            for g in selected:
                composed = tuple(g[i] for i in element)
                if composed not in group:
                    if len(group) >= limit:
                        raise ValueError('Symmetry group exceeds 4096 site permutations; use --symmetry none')
                    group.add(composed)
                    pending.append(composed)
    return np.array(sorted(group), dtype=int)


def site_permutations(geometry, labels, prepared, symprec):
    """Find parent-cell symmetries, then verify exact target bond multiplicities.

    Labels, not current chemical occupations, color the parent lattice. Every
    accepted permutation preserves each sublattice label and each requested
    directed-shell adjacency matrix individually. Thus no species, target,
    weight or composition interchange is assumed. Integer graph checks protect
    the objective even when crystallographic symmetry uses a loose tolerance.
    """
    sites = len(geometry)
    label_numbers = {label: i + 1 for i, label in enumerate(sorted(set(labels)))}
    cell = (geometry.cell.array, geometry.get_scaled_positions(),
            [label_numbers[label] for label in labels])
    symmetry = spglib.get_symmetry(cell, symprec=symprec)
    if symmetry is None:
        raise ValueError('Could not determine cell symmetry; use --symmetry none')
    graphs = []
    for centers, neighbors, *_ in prepared:
        graph = np.zeros((sites, sites), dtype=np.int64)
        np.add.at(graph, (centers, neighbors), 1)
        if not any(np.array_equal(graph, other) for other in graphs):
            graphs.append(graph)
    positions = cell[1]
    accepted, checked = [], set()
    for rotation, translation in zip(symmetry['rotations'], symmetry['translations']):
        transformed = positions @ rotation.T + translation
        delta = transformed[:, None, :] - positions[None, :, :]
        delta -= np.rint(delta)
        distance = np.linalg.norm(delta @ geometry.cell.array, axis=2)
        matches = (distance <= symprec) & (labels[:, None] == labels[None, :])
        if not np.all(matches.sum(axis=1) == 1) or not np.all(matches.sum(axis=0) == 1):
            continue
        permutation = matches.argmax(axis=1)
        key = tuple(permutation)
        if key in checked:
            continue
        checked.add(key)
        if all(np.array_equal(graph[np.ix_(permutation, permutation)], graph) for graph in graphs):
            accepted.append(key)
    return permutation_group(accepted, sites)


def symmetry_occupations(labels, composition, permutations):
    """Generate one occupation per group orbit by successive subset orbits.

    At each species assignment, choose one site subset per orbit of the group
    fixing previous assignments, then recurse with its setwise stabilizer.
    The last species occupies the complement. Leaf multiplicity is the full
    group order divided by the stabilizer order (orbit-stabilizer theorem).
    No list of all complete labeled-site decorations is generated or stored.
    """
    groups = [(np.flatnonzero(labels == name).tolist(),
               [(s, n) for s, n in sorted(composition[name]['counts'].items()) if n])
              for name in sorted(composition)]
    current = [''] * len(labels)
    order = len(permutations)

    def assign_group(index, stabilizer):
        if index == len(groups):
            yield tuple(current), order // len(stabilizer)
            return
        positions, species = groups[index]

        def assign_species(remaining, species_index, subgroup):
            symbol, count = species[species_index]
            if species_index == len(species) - 1:
                for site in remaining:
                    current[site] = symbol
                yield from assign_group(index + 1, subgroup)
                return
            seen = set()
            for selected in combinations(remaining, count):
                if selected in seen:
                    continue
                # A subset and all its images share one representative.
                images = np.sort(subgroup[:, selected], axis=1)
                seen.update(map(tuple, images))
                fixed = subgroup[np.all(images == np.array(selected), axis=1)]
                selected_set = set(selected)
                for site in selected:
                    current[site] = symbol
                yield from assign_species([site for site in remaining if site not in selected_set],
                                          species_index + 1, fixed)

        yield from assign_species(positions, 0, stabilizer)

    yield from assign_group(0, permutations)
