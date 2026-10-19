"""Read immutable ATAT reference data without importing or executing ATAT/icet."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

FIXTURES = Path(__file__).resolve().parent / 'fixtures' / 'atat'


def manifest(root=FIXTURES):
    root = Path(root)
    data = json.loads((root / 'manifest.json').read_text())
    for name, digest in data['sha256'].items():
        if hashlib.sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'ATAT fixture checksum mismatch: {name}')
    return data


def figure_model(path):
    """Independent Python evaluation of the documented native FIGURES format."""
    path = Path(path)
    text = gzip.decompress(path.read_bytes()).decode() if path.suffix == '.gz' else path.read_text()
    t = iter(text.split())
    if next(t) != 'SCRAPS_FIGURES':
        raise ValueError('Invalid figure header')
    version, sites, species, rows, count = [int(next(t)) for _ in range(5)]
    if version not in (1, 2, 3):
        raise ValueError('Unsupported figure version')
    cell = np.array([float(next(t)) for _ in range(9)]).reshape(3, 3)
    positions = np.array([float(next(t)) for _ in range(sites*3)]).reshape(sites, 3)
    basis = np.array([float(next(t)) for _ in range(rows*species)]).reshape(rows, species)
    allowed = [list(range(species)) for _ in range(sites)]
    if version >= 2:
        allowed = [[int(next(t)) for _ in range(int(next(t)))] for _ in range(sites)]
    groups, local_counts = [], []
    if version == 3:
        for _ in range(sites):
            groups.append(int(next(t)))
            local_counts.append([int(next(t)) for _ in range(species)])
    features = []
    for _ in range(count):
        order, diameter, target, n = int(next(t)), float(next(t)), float(next(t)), int(next(t))
        terms = []
        for _ in range(n):
            coefficient = float(next(t))
            factors = [(int(next(t)), int(next(t))) for _ in range(order)]
            terms.append((coefficient, factors))
        features.append(dict(order=order, diameter=diameter, target=target, terms=terms))
    if next(t, None) is not None:
        raise ValueError('Trailing figure data')
    return dict(sites=sites, species=species, cell=cell, positions=positions,
                basis=basis, allowed=allowed, groups=groups, local_counts=local_counts,
                features=features)


def correlations(model, occupations):
    occ = np.asarray(occupations, dtype=int)
    if occ.shape != (model['sites'],) or any(s not in a for s,a in zip(occ, model['allowed'])):
        raise ValueError('Invalid occupations or site alphabet')
    basis = model['basis']
    return np.array([sum(c*np.prod([basis[b, occ[s]] for s,b in terms])
                        for c,terms in f['terms']) for f in model['features']])


def metrics(model, vector, wd=1.0, wn=0.5):
    features = model['features']
    errors = np.abs(np.asarray(vector)-[f['target'] for f in features])
    diameter = np.array([f['diameter'] for f in features])
    order = np.array([f['order'] for f in features])
    logw = -wd*diameter/min(diameter)+(order-2)*np.log(wn)
    weights = np.exp(logw-max(logw))
    return dict(max_error=float(max(errors)), weighted_l1=float(weights@errors/sum(weights)),
                by_order={str(n):dict(max_error=float(max(errors[order==n])),
                                      rms=float(np.sqrt(np.mean(errors[order==n]**2))))
                          for n in sorted(set(order))})
