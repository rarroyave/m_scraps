"""Exact bounded enumeration of necessary pair-count tables, not occupations."""
from fractions import Fraction


def joint_count_bounds(blocks, entries, composition, sites, max_states):
    from sro_bounds import pair_count_bound, graph_count_bound
    totals = dict(max_error=Fraction(0), mean_absolute_error=Fraction(0), rms_error=Fraction(0))
    diagnostics = []
    for block in blocks.values():
        indices = block['indices']
        independent = [Fraction(entries[i]['component_lower_bound_exact']) for i in indices]
        fallback = dict(max_error=max(independent), mean_absolute_error=sum(independent),
                        rms_error=sum(e * e for e in independent))
        minima = fallback
        states, tables, complete = 0, 0, True
        if len(indices) > 1:
            source, destination = block['source'], block['destination']
            src, dst = composition[source], composition[destination]
            aa = sorted(a for a, n in src['counts'].items() if n)
            bb = sorted(b for b, n in dst['counts'].items() if n)
            symmetric = source == destination
            row_totals = [src['counts'][a] * block['z_src'] for a in aa]
            col_totals = [dst['counts'][b] * block['z_dst'] for b in bb]
            variables = []
            for i, a in enumerate(aa):
                for j, b in enumerate(bb):
                    if symmetric and j < i:
                        continue
                    denominator = Fraction(row_totals[i] * dst['counts'][b], dst['sites'])
                    low, high, _, _, _, _ = pair_count_bound(row_totals[i], col_totals[j],
                        len(block['ii']), denominator, Fraction(0), same_species=symmetric and i == j)
                    low, high, step, _, _, _ = graph_count_bound(block['ii'], block['jj'], sites,
                        composition, source, destination, a, b, low, high, denominator, Fraction(0))
                    variables.append((i, j, low, high, step))
            table = {}
            best = {k: None for k in totals}

            def visit(k):
                nonlocal states, tables, complete
                if states >= max_states:
                    complete = False
                    return
                states += 1
                if k == len(variables):
                    if any(row_totals) or (not symmetric and any(col_totals)):
                        return
                    tables += 1
                    residuals = []
                    for index in indices:
                        entry = entries[index]
                        a, b = entry['center_species'], entry['neighbor_species']
                        if entry['source'] != source:
                            a, b = b, a
                        n = table[aa.index(a), bb.index(b)]
                        residuals.append(abs((Fraction(entry['target_pair_count_exact']) - n) /
                                             Fraction(entry['denominator_exact'])))
                    values = dict(max_error=max(residuals), mean_absolute_error=sum(residuals),
                                  rms_error=sum(e * e for e in residuals))
                    for name, value in values.items():
                        if best[name] is None or value < best[name]:
                            best[name] = value
                    return
                i, j, low, high, step = variables[k]
                limit = min(high, row_totals[i], row_totals[j] if symmetric and i != j else
                            high if symmetric else col_totals[j])
                # Remaining domain capacities provide cheap necessary pruning.
                for r in range(len(aa)):
                    remaining = sum(v[3] for v in variables[k:] if v[0] == r or (symmetric and v[1] == r))
                    if row_totals[r] < 0 or row_totals[r] > remaining:
                        return
                for n in range(low, limit + 1, step or 1):
                    table[i, j] = n
                    row_totals[i] -= n
                    if symmetric:
                        table[j, i] = n
                        if i != j:
                            row_totals[j] -= n
                    else:
                        col_totals[j] -= n
                    visit(k + 1)
                    row_totals[i] += n
                    if symmetric and i != j:
                        row_totals[j] += n
                    if not symmetric:
                        col_totals[j] += n
                    if not complete:
                        return
            visit(0)
            if complete:
                if not tables:
                    raise RuntimeError('Joint count relaxation excludes the supplied valid witness')
                minima = best
        strengthened = any(minima[k] > fallback[k] for k in totals)
        diagnostics.append(dict(target_indices=indices, states=states, feasible_count_tables=tables,
            status='complete' if len(indices) > 1 and complete else 'state_cap' if not complete else 'single_component',
            strengthened=strengthened,
            explanation=('Requested targets compete for the same reciprocal pair counts or fixed neighbor totals.'
                         if strengthened else 'No strengthening established; a zero bound does not prove lattice feasibility.'),
            max_error_lower_bound_exact=str(minima['max_error']),
            sum_absolute_error_lower_bound_exact=str(minima['mean_absolute_error']),
            sum_squared_error_lower_bound_exact=str(minima['rms_error'])))
        totals['max_error'] = max(totals['max_error'], minima['max_error'])
        totals['mean_absolute_error'] += minima['mean_absolute_error']
        totals['rms_error'] += minima['rms_error']
    for name in ('mean_absolute_error', 'rms_error'):
        totals[name] /= len(entries)
    return totals, diagnostics
