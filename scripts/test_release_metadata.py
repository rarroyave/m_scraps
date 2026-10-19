"""Regression checks for failures that would otherwise break archival."""
import shutil
import tempfile
import unittest
from pathlib import Path

from check_release_metadata import orcid_id, validate

ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_orcid_checksums(self):
        self.assertEqual(orcid_id('https://orcid.org/0000-0001-7548-8686'),
                         '0000-0001-7548-8686')
        self.assertEqual(orcid_id('0000-0002-1694-233X'), '0000-0002-1694-233X')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            orcid_id('0000-0002-0773-9939')
        with self.assertRaisesRegex(ValueError, 'format'):
            orcid_id('not-an-orcid')

    def test_current_metadata_and_wrong_tag(self):
        version = validate(ROOT)
        self.assertEqual(validate(ROOT, f'v{version}'), version)
        with self.assertRaisesRegex(ValueError, 'versions'):
            validate(ROOT, 'v999.0.0')

    def test_invalid_archive_orcid_and_version_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('CITATION.cff', '.zenodo.json', 'pyproject.toml',
                         'CMakeLists.txt', 'python/m_scraps/__init__.py'):
                dest = root / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(ROOT / name, dest)
            path = root / '.zenodo.json'
            original = path.read_text()
            path.write_text(original.replace('0000-0001-7548-8686', '0000-0002-0773-9939'))
            with self.assertRaisesRegex(ValueError, 'checksum'):
                validate(root)
            path.write_text(original)
            path = root / 'python/m_scraps/__init__.py'
            path.write_text('__version__ = "999.0.0"\n')
            with self.assertRaisesRegex(ValueError, 'versions'):
                validate(root)


if __name__ == '__main__':
    unittest.main()
