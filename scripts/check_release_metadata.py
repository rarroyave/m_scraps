"""Check release identity across packaging, CFF and Zenodo metadata.

Run cffconvert --validate separately for the complete CFF schema. This check
validates the Zenodo fields used by this project, not the entire Zenodo API.
"""
import argparse
import ast
import json
from pathlib import Path
import re
import tomllib

import yaml


def orcid_id(value):
    """Return the canonical identifier after ISO 7064 MOD 11-2 validation."""
    identifier = value.removeprefix('https://orcid.org/')
    if not re.fullmatch(r'\d{4}-\d{4}-\d{4}-\d{3}[\dX]', identifier):
        raise ValueError(f'Invalid ORCID format: {value}')
    digits = identifier.replace('-', '')
    total = 0
    for digit in digits[:-1]:
        total = (total + int(digit)) * 2
    check = (12 - total % 11) % 11
    if digits[-1] != ('X' if check == 10 else str(check)):
        raise ValueError(f'Invalid ORCID checksum: {value}')
    return identifier


def validate(root, tag=None):
    cff = yaml.safe_load((root / 'CITATION.cff').read_text())
    zenodo = json.loads((root / '.zenodo.json').read_text())
    package = tomllib.loads((root / 'pyproject.toml').read_text())['project']
    cmake = re.search(r'project\(m_scraps\s+VERSION\s+(\S+)',
                      (root / 'CMakeLists.txt').read_text())
    module = ast.parse((root / 'python/m_scraps/__init__.py').read_text())
    versions = {'package': package['version'], 'CFF': cff['version'],
                'Zenodo': zenodo['version'], 'CMake': cmake.group(1) if cmake else None}
    versions['Python'] = next((ast.literal_eval(node.value) for node in module.body
                              if isinstance(node, ast.Assign) and any(
                                  isinstance(t, ast.Name) and t.id == '__version__'
                                  for t in node.targets)), None)
    if tag is not None:
        versions['tag'] = tag.removeprefix('v')
    if len(set(versions.values())) != 1 or not versions['package']:
        raise ValueError(f'Inconsistent release versions: {versions}')
    if zenodo['title'] != cff['title']:
        raise ValueError('CFF and Zenodo titles differ')
    if zenodo['license'].lower() != cff['license'].lower():
        raise ValueError('CFF and Zenodo licenses differ')
    if zenodo['upload_type'] != 'software' or not zenodo['description'].strip():
        raise ValueError('Zenodo needs a software resource type and description')
    if not zenodo['creators']:
        raise ValueError('Zenodo needs at least one creator')
    cff_people = [(f"{a['family-names']}, {a['given-names']}",
                   orcid_id(a['orcid']) if 'orcid' in a else None)
                  for a in cff['authors']]
    zenodo_people = [(a['name'], orcid_id(a['orcid']) if 'orcid' in a else None)
                     for a in zenodo['creators']]
    if cff_people != zenodo_people:
        raise ValueError('CFF and Zenodo creators/ORCIDs differ')
    for person in cff.get('references', []):
        for author in person.get('authors', []):
            if 'orcid' in author:
                orcid_id(author['orcid'])
    for identifier in zenodo.get('related_identifiers', []):
        if not all(identifier.get(k) for k in ('identifier', 'relation', 'resource_type')):
            raise ValueError('Incomplete Zenodo related identifier')
    return versions['package']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--tag', help='Expected release tag, e.g. v0.5.1')
    args = parser.parse_args()
    try:
        version = validate(args.root, args.tag)
    except (ValueError, KeyError, TypeError, OSError) as error:
        parser.exit(1, f'Release metadata error: {error}\n')
    print(f'Release metadata valid for {version}')
