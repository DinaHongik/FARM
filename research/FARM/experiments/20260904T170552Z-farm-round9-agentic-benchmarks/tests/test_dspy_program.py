from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.artifact_io import sha256_file
from farm_r9.dspy_program import InvalidDSPyArtifact, dspy, validate_optimizer_manifest


def _manifest() -> dict:
    return {
        "schema_version": "round9-dspy-optimizer-v1", "status": "optimized",
        "optimizer": "BootstrapFewShot", "optimizer_executed": True,
        "training_source": "recipegen", "training_classification": "public",
        "train_count": 20, "training_case_ids_sha256": "a" * 64,
        "heldout_overlap_count": 0, "program_state_path": "program_state.json",
        "program_state_sha256": "b" * 64,
    }


class DSPyProgramTests(unittest.TestCase):
    def test_manifest_rejects_private_training_and_fake_compilation(self) -> None:
        private = _manifest()
        private["training_source"] = "farm_v2"
        private["training_classification"] = "confidential"
        with self.assertRaises(InvalidDSPyArtifact):
            validate_optimizer_manifest(private)
        fake = _manifest()
        fake["optimizer_executed"] = False
        with self.assertRaises(InvalidDSPyArtifact):
            validate_optimizer_manifest(fake)

    def test_manifest_verifies_state_hash_and_containment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            state = root / "program_state.json"
            state.write_text(json.dumps({"state": 1}), encoding="utf-8")
            manifest = _manifest()
            manifest["program_state_sha256"] = sha256_file(state)
            path = root / "optimizer_manifest.json"
            path.write_text(json.dumps(manifest), encoding="utf-8")
            self.assertEqual(validate_optimizer_manifest(manifest, manifest_path=path)["status"], "optimized")
            manifest["program_state_path"] = "../escape.json"
            with self.assertRaises(InvalidDSPyArtifact):
                validate_optimizer_manifest(manifest, manifest_path=path)

    def test_real_dspy_types_when_dependency_is_available(self) -> None:
        if dspy is None:
            self.skipTest("DSPy is only pinned on DGX")
        from farm_r9.dspy_program import AppletConfigurationProgram, AppletConfigurationSignature
        self.assertTrue(issubclass(AppletConfigurationProgram, dspy.Module))
        self.assertTrue(issubclass(AppletConfigurationSignature, dspy.Signature))


if __name__ == "__main__":
    unittest.main()
