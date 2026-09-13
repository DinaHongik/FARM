from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.artifact_io import sha256_file  # noqa: E402
from farm_r9.recipegen_farm_overlap import (  # noqa: E402
    LEXICAL_THRESHOLD,
    RecipeGenFarmOverlapError,
    audit_cases_against_training,
    build_training_index,
    canonical_input_key,
    content_tokens,
    load_pinned_farm_training_index,
    make_public_aggregate,
    normalize_input,
)


def _corpus(
    kind: str, url: str, channel: str, display: str, function: str
) -> dict[str, object]:
    return {
        "url": url,
        "kind": kind,
        "channel": channel,
        "channel_display": display,
        "function_name": function,
        "function_slug": canonical_input_key(function).replace(" ", "_"),
    }


def _row(
    partition: str,
    group_id: str,
    query: str,
    trigger_url: str,
    action_url: str,
) -> dict[str, object]:
    return {
        "split": partition,
        "group_id": group_id,
        "query": query,
        "query_norm": canonical_input_key(query),
        "valid_pairs": [{"trigger_url": trigger_url, "action_url": action_url}],
    }


def _case(
    index: int,
    query: str,
    trigger: tuple[str, str],
    action: tuple[str, str],
) -> dict[str, object]:
    return {
        "benchmark": "recipegen_gold",
        "case_id": f"public-case-{index}",
        "input": {"query": query},
        "private_gold": {
            "trigger_channel": trigger[0],
            "trigger_function": f"{trigger[0]}.{trigger[1]}",
            "action_channel": action[0],
            "action_function": f"{action[0]}.{action[1]}",
        },
    }


def _fixture() -> tuple[list[dict[str, object]], object]:
    triggers = [
        _corpus("trigger", "t1", "alpha_service", "Alpha Service", "New Item"),
        _corpus("trigger", "t2", "gamma", "Gamma", "Timer Fires"),
        _corpus("trigger", "t3", "epsilon", "Epsilon", "Door Opens"),
    ]
    actions = [
        _corpus("action", "a1", "beta", "Beta", "Save Item"),
        _corpus("action", "a2", "delta", "Delta", "Send Alert"),
        _corpus("action", "a3", "zeta", "Zeta", "Turn Light On"),
    ]
    encoder_query = "Save Red Blue Green Yellow Orange Purple White Black Gray Brown"
    reranker_query = "alpha beta gamma delta epsilon zeta eta theta iota kappa extra"
    index = build_training_index(
        split_rows={
            "encoder_train": [_row("encoder_train", "g1", encoder_query, "t1", "a1")],
            "reranker_train": [
                _row("reranker_train", "g2", reranker_query, "t2", "a2")
            ],
        },
        trigger_corpus=triggers,
        action_corpus=actions,
        source_artifact_sha256="1" * 64,
        manifest_sha256="2" * 64,
    )
    cases = [
        _case(
            0,
            encoder_query,
            ("Alpha_Service", "New_Item"),
            ("Beta", "Save_Item"),
        ),
        _case(
            1,
            "alpha beta gamma delta epsilon zeta eta theta iota kappa",
            ("Gamma", "Timer_Fires"),
            ("Delta", "Send_Alert"),
        ),
        _case(
            2,
            "unrelated words another workflow",
            ("Epsilon", "Door_Opens"),
            ("Zeta", "Turn_Light_On"),
        ),
    ]
    return cases, index


class RecipeGenFarmOverlapTests(unittest.TestCase):
    def test_strict_exact_and_lossy_lexical_normalizers_remain_distinct(self) -> None:
        self.assertEqual(normalize_input("  Café\r\n  X! "), "Café X!")
        self.assertNotEqual(normalize_input("ABC!"), normalize_input("abc"))
        self.assertEqual(canonical_input_key("ABC!"), canonical_input_key("abc"))
        self.assertEqual(
            content_tokens("When the Red Button fires"), {"red", "button", "fires"}
        )

    def test_predeclared_near_threshold_is_high_precision(self) -> None:
        left = frozenset(
            "alpha beta gamma delta epsilon zeta eta theta iota kappa".split()
        )
        right = left | {"extra"}
        self.assertGreaterEqual(
            len(left & right) / len(left | right), LEXICAL_THRESHOLD
        )

    def test_exact_program_and_lexical_outcomes_are_separate(self) -> None:
        cases, index = _fixture()
        records, counts, failures = audit_cases_against_training(
            cases=cases,
            index=index,
            benchmark="recipegen_gold",
        )
        self.assertEqual(counts["all_training_input_exact_overlap"], 1)
        self.assertEqual(counts["all_training_input_canonical_equivalent_overlap"], 1)
        self.assertEqual(counts["all_training_program_exact_overlap"], 2)
        self.assertEqual(counts["all_training_input_program_pair_exact_overlap"], 1)
        self.assertEqual(counts["all_training_lexical_near_input_overlap"], 1)
        self.assertEqual(
            counts["all_training_lexical_near_input_program_pair_overlap"], 1
        )
        self.assertEqual(counts["all_training_exact_or_lexical_near_input_overlap"], 2)
        self.assertEqual(counts["catalog_joint_endpoint_mapped"], 3)
        self.assertEqual(sum(failures.values()), 0)
        serialized = json.dumps(records)
        for private_value in ("g1", "g2", "t1", "a1", "Save Red Blue"):
            self.assertNotIn(private_value, serialized)
        self.assertNotIn("sha256", serialized.casefold())

    def test_loader_binds_manifest_hashes_and_recomputes_query_key(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            data_root = Path(directory)
            (data_root / "corpus").mkdir()
            (data_root / "splits").mkdir()
            values = {
                "corpus/triggers.json": [
                    _corpus("trigger", "t1", "alpha", "Alpha", "Starts")
                ],
                "corpus/actions.json": [
                    _corpus("action", "a1", "beta", "Beta", "Stops")
                ],
                "splits/encoder_train.json": [
                    _row("encoder_train", "g1", "One input", "t1", "a1")
                ],
                "splits/reranker_train.json": [
                    _row("reranker_train", "g2", "Other input", "t1", "a1")
                ],
            }
            artifacts = {}
            expected_rows = {}
            for relative, value in values.items():
                path = data_root / relative
                path.write_text(json.dumps(value), encoding="utf-8")
                artifacts[relative] = {
                    "sha256": sha256_file(path),
                    "rows": len(value),
                }
                expected_rows[relative] = len(value)
            manifest = {
                "dataset_id": "fixture-v2",
                "config": {
                    "normalizer": "unicode-nfkc-casefold-alnum-v1",
                    "paraphrase": {
                        "threshold": 0.9,
                        "minimum_content_tokens": 3,
                    },
                },
                "artifacts": artifacts,
            }
            manifest_path = data_root / "manifest.json"
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            index = load_pinned_farm_training_index(
                data_root,
                expected_dataset_id="fixture-v2",
                expected_manifest_sha256=sha256_file(manifest_path),
                expected_artifacts={
                    relative: entry["sha256"] for relative, entry in artifacts.items()
                },
                expected_rows=expected_rows,
            )
            self.assertEqual(len(index.examples["encoder_train"]), 1)

            values["splits/encoder_train.json"][0]["query_norm"] = "wrong"
            path = data_root / "splits/encoder_train.json"
            path.write_text(
                json.dumps(values["splits/encoder_train.json"]), encoding="utf-8"
            )
            digest = sha256_file(path)
            manifest["artifacts"]["splits/encoder_train.json"]["sha256"] = digest
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            expected_artifacts = {
                relative: entry["sha256"]
                for relative, entry in manifest["artifacts"].items()
            }
            with self.assertRaisesRegex(
                RecipeGenFarmOverlapError, "does not reproduce"
            ):
                load_pinned_farm_training_index(
                    data_root,
                    expected_dataset_id="fixture-v2",
                    expected_manifest_sha256=sha256_file(manifest_path),
                    expected_artifacts=expected_artifacts,
                    expected_rows=expected_rows,
                )

    def test_confidential_public_aggregate_is_denominator_aligned(self) -> None:
        cases, index = _fixture()
        records, counts, failures = audit_cases_against_training(
            cases=cases,
            index=index,
            benchmark="recipegen_gold",
        )
        aggregate = make_public_aggregate(
            summary={
                "n": len(records),
                "counts": counts,
                "failures": failures,
                "sample_artifact_sha256": "3" * 64,
                "source_artifact_sha256": "4" * 64,
            },
            case_audit_sha256="5" * 64,
            code_artifact_sha256="6" * 64,
        )
        self.assertNotIn("benchmark_label", aggregate)
        self.assertEqual(
            set(aggregate["raw_numerators"]),
            set(aggregate["raw_denominators"]),
        )
        self.assertTrue(
            all(
                value == len(records)
                for value in aggregate["raw_denominators"].values()
            )
        )

    def test_cli_refuses_to_overwrite_before_reading_sources(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            process = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "audit_recipegen_farm_training_overlap.py"),
                    "--farm-root",
                    str(Path(directory) / "missing"),
                    "--output-root",
                    str(output),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(process.returncode, 0)
            self.assertIn("refusing to overwrite", process.stdout + process.stderr)

    def test_real_outputs_are_private_and_case_ledger_has_no_farm_records(self) -> None:
        audit_root = ROOT / "audits" / "recipegen_farm_training_overlap_v2"
        if not audit_root.is_dir():
            self.skipTest("requires the locally generated overlap audit")
        self.assertEqual(audit_root.stat().st_mode & 0o777, 0o700)
        forbidden_keys = {
            "farm_id",
            "group_id",
            "query",
            "program",
            "url",
            "input_sha256",
            "program_sha256",
            "source_index",
        }

        def keys(value: object) -> set[str]:
            if isinstance(value, dict):
                return set(value) | {
                    child_key for child in value.values() for child_key in keys(child)
                }
            if isinstance(value, list):
                return {child_key for child in value for child_key in keys(child)}
            return set()

        for split in ("gold", "noisy"):
            directory = audit_root / split
            aggregate_path = directory / "aggregate_public.json"
            ledger_path = directory / "case_audit.jsonl"
            self.assertEqual(directory.stat().st_mode & 0o777, 0o700)
            self.assertEqual(aggregate_path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(ledger_path.stat().st_mode & 0o777, 0o600)
            aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
            self.assertEqual(
                set(aggregate),
                {
                    "n",
                    "raw_numerators",
                    "raw_denominators",
                    "percentages",
                    "confidence_intervals",
                    "failure_counts",
                    "hashes",
                },
            )
            with ledger_path.open(encoding="utf-8") as handle:
                records = [json.loads(line) for line in handle]
            self.assertEqual(len(records), 150)
            self.assertTrue(keys(records).isdisjoint(forbidden_keys))


if __name__ == "__main__":
    unittest.main()
