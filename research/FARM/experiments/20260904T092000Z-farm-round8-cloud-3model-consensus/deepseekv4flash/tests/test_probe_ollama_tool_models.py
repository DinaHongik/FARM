from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from probe_ollama_tool_models import load_env, synthetic_request, write_json_new  # noqa: E402


class ProbeTests(unittest.TestCase):
    def test_synthetic_request_has_complete_opaque_candidate_contract(self) -> None:
        request = synthetic_request()
        self.assertEqual(10, len(request["candidates"]["trigger"]))
        self.assertEqual(10, len(request["candidates"]["action"]))
        serialized = json.dumps(request)
        self.assertNotIn("https://", serialized)
        self.assertNotIn("valid_pairs", serialized)
        self.assertNotIn("gold", serialized.casefold())

    def test_env_parser_handles_quotes_without_logging_values(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / ".env"
            path.write_text("A='one'\nB=\"two\"\n# C=three\n", encoding="utf-8")
            self.assertEqual({"A": "one", "B": "two"}, load_env(path))

    def test_new_writer_rejects_secret_and_overwrite(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "probe.json"
            with self.assertRaisesRegex(RuntimeError, "credential"):
                write_json_new(path, {"bad": "secret"}, "secret")
            write_json_new(path, {"ok": True}, "secret")
            with self.assertRaises(FileExistsError):
                write_json_new(path, {"ok": False}, "secret")


if __name__ == "__main__":
    unittest.main()

