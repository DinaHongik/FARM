"""Check the publication gate after merging the existing repository."""
import contextlib
import importlib.util
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('release_verifier', ROOT/'scripts/verify_release.py')
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


class ReleaseGateTests(unittest.TestCase):
    def verify_paths(self, paths):
        stdout, stderr = io.StringIO(), io.StringIO()
        with patch.object(verifier.subprocess, 'run', return_value=SimpleNamespace(stdout='\0'.join(paths))):
            with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                result = verifier.main()
        return result, stderr.getvalue()

    def test_existing_data_documentation_and_scripts_pass(self):
        result, errors = self.verify_paths([
            'data/README.md', 'data/category_analyze.py', 'data/split_data.py',
            'data/test/README.md', 'data/ifttt_examples.json', 'data/manifest.json',
        ])
        self.assertEqual(result, 0, errors)

    def test_private_data_credentials_and_weights_still_fail(self):
        paths = ['data/full_dataset.json', 'data/test/examples.json', 'data/test/raw.csv',
                 'data/.env', 'private/README.md', 'models/encoder.safetensors', 'secret.pem']
        result, errors = self.verify_paths(paths)
        self.assertEqual(result, 1)
        for name in paths:
            self.assertIn('Private or unexpected tracked artifact: '+name, errors)

    def test_manuscript_files_are_blocked_even_when_force_added(self):
        paths = ['paper/source.txt', 'docs/manuscript.tex', 'docs/appendix.PDF',
                 'main.docx', 'references.bib', 'manuscript/README.md']
        result, errors = self.verify_paths(paths)
        self.assertEqual(result, 1)
        for name in paths:
            self.assertIn('Manuscript artifact is not allowed in this release: '+name, errors)


if __name__ == '__main__':
    unittest.main()
