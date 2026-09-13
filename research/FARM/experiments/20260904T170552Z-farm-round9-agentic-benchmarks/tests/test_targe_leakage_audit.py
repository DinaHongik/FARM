from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.targe_leakage_audit import (  # noqa: E402
    CLASSES,
    classify_frozen_sample,
    normalize_exact_text,
)


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def _case(
    index: int, query: str, output: tuple[str, str, str, str], prior: bool
) -> dict[str, object]:
    tc, tf, ac, af = output
    return {
        "benchmark": "targe_gold",
        "case_id": f"case-{index}",
        "input": {"query": query},
        "private_gold": {
            "trigger_channel": tc,
            "trigger_function": tf,
            "action_channel": ac,
            "action_function": af,
        },
        "audit": {"source_index": index, "train_pair_overlap": prior},
    }


class TargeLeakageAuditTests(unittest.TestCase):
    def test_normalization_changes_only_unicode_form_and_whitespace(self) -> None:
        self.assertEqual(normalize_exact_text("  Café\r\n  THEN  X "), "Café THEN X")
        self.assertNotEqual(normalize_exact_text("ABC!"), normalize_exact_text("abc!"))
        self.assertNotEqual(normalize_exact_text("A-B"), normalize_exact_text("A B"))

    def test_classifies_all_three_outcomes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train = tmp_path / "train_recipe.json"
            test = tmp_path / "test_recipe.json"
            selected = tmp_path / "cases.jsonl"
            _json(
                train,
                [
                    {
                        "input": "same input\r\n",
                        "output": "IF Alpha Start THEN Beta Send",
                    },
                    {"input": "input only", "output": "IF Alpha Start THEN Beta Old"},
                ],
            )
            _json(
                test,
                [
                    {
                        "input": "same   input",
                        "output": "IF Alpha Start THEN Beta Send",
                    },
                    {"input": "input only", "output": "IF Alpha Start THEN Beta New"},
                    {"input": "never seen", "output": "IF Gamma Tick THEN Delta Store"},
                ],
            )
            _jsonl(
                selected,
                [
                    _case(0, "same input", ("Alpha", "Start", "Beta", "Send"), True),
                    _case(1, "input only", ("Alpha", "Start", "Beta", "New"), True),
                    _case(2, "never seen", ("Gamma", "Tick", "Delta", "Store"), False),
                ],
            )

            records, aggregate = classify_frozen_sample(
                selected_cases_path=selected,
                train_recipe_path=train,
                test_recipe_path=test,
                benchmark="targe_gold",
                expected_n=3,
            )
            self.assertEqual([row["leakage_class"] for row in records], list(CLASSES))
            self.assertEqual(aggregate["class_counts"], {name: 1 for name in CLASSES})
            self.assertEqual(aggregate["n"], 3)
            self.assertEqual(
                set(aggregate["source_hashes"]),
                {"selected_cases_sha256", "train_recipe_sha256", "test_recipe_sha256"},
            )
            self.assertTrue(
                all(
                    "query" not in row and "output_program" not in row
                    for row in records
                )
            )

    def test_rejects_case_that_does_not_match_official_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            train = tmp_path / "train_recipe.json"
            test = tmp_path / "test_recipe.json"
            selected = tmp_path / "cases.jsonl"
            _json(train, [{"input": "x", "output": "IF A B THEN C D"}])
            _json(test, [{"input": "official", "output": "IF A B THEN C D"}])
            _jsonl(selected, [_case(0, "changed", ("A", "B", "C", "D"), False)])
            with self.assertRaisesRegex(ValueError, "frozen query differs"):
                classify_frozen_sample(
                    selected_cases_path=selected,
                    train_recipe_path=train,
                    test_recipe_path=test,
                    benchmark="targe_gold",
                    expected_n=1,
                )

    def test_cli_refuses_to_overwrite_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            output = tmp_path / "existing"
            output.mkdir()
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_targe_program_leakage.py"),
                    "--output-root",
                    str(output),
                    "--repo-root",
                    str(tmp_path),
                    "--splits",
                    "gold",
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("refusing to overwrite", process.stdout + process.stderr)
            self.assertEqual(list(output.iterdir()), [])

    def test_source_guards_private_non_destructive_artifacts(self) -> None:
        prepared = ROOT / "prepared" / "targe_gold" / "cases.jsonl"
        if not prepared.is_file():
            self.skipTest("requires the access-controlled prepared TARGE artifact")
        script = (ROOT / "scripts" / "audit_targe_program_leakage.py").read_text(
            encoding="utf-8"
        )
        self.assertIn("if output_root.exists()", script)
        self.assertIn("os.chmod(records_path, 0o600)", script)
        self.assertIn("os.chmod(aggregate_path, 0o600)", script)


if __name__ == "__main__":
    unittest.main()
