"""Assemble a completed campaign into the portable, checksummed test corpus."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    status = json.loads((args.run/'status.json').read_text())
    if status != dict(state='complete', completed=40, total=40):
        parser.error('Expected a completed 40-trial campaign')
    source = args.run/'regression-fixtures'
    manifest = json.loads((source/'manifest.json').read_text())
    for name, digest in manifest['sha256'].items():
        if hashlib.sha256((source/name).read_bytes()).hexdigest() != digest:
            parser.error(f'Campaign fixture checksum mismatch: {name}')
    shutil.copytree(source, args.output)
    for name in manifest['cases']:
        path = args.output/f'{name}.json.gz'
        data = json.loads(gzip.decompress(path.read_bytes()))
        if data['case']['target_mode'] == 'reference':
            data['target_reference_occupations'] = [int(v) for v in
                (args.run/'inputs'/f'{name}.reference-occupations.txt').read_text().split()]
            path.write_bytes(gzip.compress(json.dumps(data).encode(), mtime=0))
        shutil.copy2(args.run/'inputs'/f'{name}.poscar', args.output/f'{name}.poscar')
    shutil.copy2(args.run/'summary.json', args.output/'baseline.json')
    shutil.copy2(args.run/'protocol.json', args.output/'protocol.json')
    manifest.update(schema_version=1, atat_version='3.52', icet_version='4.0',
        note='Measured 40-trial screening campaign; no cross-host timing equivalence implied.',
        source_manifest_sha256=hashlib.sha256((args.run/'manifest.json').read_bytes()).hexdigest())
    manifest['sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(args.output.iterdir()) if p.name != 'manifest.json'}
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')


if __name__ == '__main__':
    main()
