"""Exercise history integration against a temporary local bare Git repository."""
import contextlib
import importlib.util
import io
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('farm_publish', ROOT/'scripts/publish.py')
publish = importlib.util.module_from_spec(spec)
spec.loader.exec_module(publish)


def git(root, *args):
    return subprocess.run(['git', *args], cwd=root, check=True, capture_output=True, text=True).stdout.strip()


def initialize(root):
    root.mkdir()
    git(root, 'init', '-b', 'main')
    git(root, 'config', 'user.name', 'FARM release test')
    git(root, 'config', 'user.email', 'test@example.invalid')


class PublishingTests(unittest.TestCase):
    def test_integrates_existing_history_and_preserves_remote_only_files(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            bare = base/'origin.git'
            git(base, 'init', '--bare', '--initial-branch=main', str(bare))
            old = base/'old'; initialize(old)
            (old/'README.md').write_text('Previously published README\n')
            (old/'existing.txt').write_text('Keep this file\n')
            git(old, 'add', '.'); git(old, 'commit', '-m', 'Existing publication')
            original = git(old, 'rev-parse', 'HEAD')
            git(old, 'remote', 'add', 'origin', str(bare)); git(old, 'push', 'origin', 'main')
            release = base/'release'; initialize(release)
            (release/'README.md').write_text('Updated FARM code release\n')
            (release/'scripts').mkdir(); (release/'tests').mkdir()
            (release/'scripts/verify_release.py').write_text('raise SystemExit(0)\n')
            (release/'tests/test_integrated.py').write_text(
                'import unittest\nfrom pathlib import Path\n'
                'class IntegratedFiles(unittest.TestCase):\n'
                '    def test_existing_file_preserved(self):\n'
                '        self.assertTrue(Path("existing.txt").is_file())\n')
            git(release, 'add', '.'); git(release, 'commit', '-m', 'New code release')
            git(release, 'remote', 'add', 'origin', str(bare))
            with patch.object(publish, 'ROOT', release), patch.object(publish, 'ALLOWED', {str(bare)}):
                with contextlib.redirect_stdout(io.StringIO()):
                    publish.main()
            self.assertEqual((release/'README.md').read_text(), 'Updated FARM code release\n')
            self.assertEqual((release/'existing.txt').read_text(), 'Keep this file\n')
            git(release, 'merge-base', '--is-ancestor', original, 'HEAD')
            self.assertEqual(git(bare, 'rev-parse', 'main'), git(release, 'rev-parse', 'HEAD'))

    def test_wrong_destination_fails_before_fetch_or_push(self):
        with tempfile.TemporaryDirectory() as directory:
            release=Path(directory)/'release'; initialize(release)
            (release/'README.md').write_text('FARM\n')
            git(release,'add','.');git(release,'commit','-m','Release')
            git(release,'remote','add','origin','https://example.invalid/wrong.git')
            with patch.object(publish,'ROOT',release):
                with self.assertRaisesRegex(RuntimeError,'origin must point only'):
                    publish.main()


if __name__=='__main__':
    unittest.main()
