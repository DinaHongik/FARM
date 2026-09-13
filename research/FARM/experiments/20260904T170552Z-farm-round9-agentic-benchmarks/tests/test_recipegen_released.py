from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from farm_r9.adapters.recipegen import prepare_recipegen
from farm_r9.artifact_io import (
    ordered_ids_sha256,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from farm_r9 import recipegen_released as released


CHANNEL_TARGET = "GitHub <sep> Twitter"
FUNCTION_TARGET = (
    "GitHub <sep> GitHub.New_repository <sep> Twitter <sep> Twitter.Post_a_tweet"
)
FIELD_TARGET = (
    "GitHub <sep> GitHub.New_repository <sep> Owner <sep> Twitter <sep> "
    "Twitter.Post_a_tweet <sep> Tweet text"
)
EMPTY_FIELD_TARGET = (
    "GitHub <sep> GitHub.New_repository <sep> ### <sep> Twitter <sep> "
    "Twitter.Post_a_tweet <sep> ###"
)


class RecipeGenReleasedTests(unittest.TestCase):
    def _fixture(
        self, root: Path, *, wrong_field_endpoint: bool = False,
        field_target: str = FIELD_TARGET,
    ) -> dict[str, Path]:
        processed = root / "processed.csv"
        processed.write_text(
            "source,split,target,granularity\n"
            f"do the public task,gold,{CHANNEL_TARGET},channel\n"
            f"do the public task,gold,{FUNCTION_TARGET},function\n"
            f"do the public task,gold,{field_target},field\n",
            encoding="utf-8",
        )
        cases, sample = prepare_recipegen(
            processed, split="gold", size=1, seed=released.EXPECTED_SEED
        )
        cases_path = root / "cases.jsonl"
        manifest_path = root / "sample.json"
        write_jsonl_atomic(cases_path, cases)
        sample.update(
            {
                "schema_version": "round9-sample-manifest-v1",
                "case_file": "prepared/recipegen_gold/cases.jsonl",
                "ordered_case_ids_sha256": ordered_ids_sha256(cases),
                "case_payload_sha256": sha256_file(cases_path),
            }
        )
        write_json_atomic(manifest_path, sample)

        archive_path = root / "results.zip"
        field_prediction = (
            field_target.replace("GitHub.New_repository", "GitHub.Wrong_endpoint")
            if wrong_field_endpoint
            else field_target
        )
        track_predictions = {
            "channel": CHANNEL_TARGET,
            "function": FUNCTION_TARGET,
            "field": field_prediction,
        }
        prompts = {
            "channel": "GENERATE CHANNEL ONLY WITHOUT FUNCTION",
            "function": "GENERATE CHANNEL AND FUNCTION FOR BOTH TRIGGER AND ACTION",
            "field": "GENERATE ON THE FIELD-LEVEL GRANULARITY",
        }
        targets = {
            "channel": CHANNEL_TARGET,
            "function": FUNCTION_TARGET,
            "field": field_target,
        }
        with zipfile.ZipFile(
            archive_path, "w", compression=zipfile.ZIP_STORED
        ) as archive:
            for track in ("channel", "function", "field"):
                base = f"results/oneshot_model/oneshot_gold_{track}/id0"
                archive.writestr(
                    f"{base}.src", f"{prompts[track]} <pf> do the public task\n"
                )
                archive.writestr(f"{base}.gold", targets[track] + "\n")
                predictions = [
                    track_predictions[track],
                    *[f"wrong beam {index}" for index in range(2, 11)],
                ]
                archive.writestr(f"{base}.pred", "\n".join(predictions) + "\n")
        return {
            "archive": archive_path,
            "processed": processed,
            "cases": cases_path,
            "manifest": manifest_path,
        }

    def _evaluate(self, paths: dict[str, Path]) -> dict:
        with patch.multiple(
            released,
            EXPECTED_SAMPLE_N=1,
            EXPECTED_POPULATIONS={"gold": 1, "noisy": 1},
            RESULTS_ZIP_SIZE_BYTES=paths["archive"].stat().st_size,
            RESULTS_ZIP_SHA256=sha256_file(paths["archive"]),
            PROCESSED_CSV_SHA256=sha256_file(paths["processed"]),
            BOOTSTRAP_RESAMPLES=25,
        ):
            return released.evaluate_released_predictions(
                archive_path=paths["archive"],
                processed_csv=paths["processed"],
                cases_path=paths["cases"],
                manifest_path=paths["manifest"],
                split="gold",
            )

    def test_scores_verified_archive_without_case_level_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._evaluate(self._fixture(Path(directory)))
        self.assertEqual(aggregate["n"], 1)
        self.assertEqual(
            aggregate["raw_numerators"]["function_track_endpoint_joint_exact"], 1
        )
        self.assertEqual(
            aggregate["raw_numerators"]["field_joint_ordered_exact_endpoint_gated"],
            1,
        )
        conditional_label = (
            "field_track_ordered_label_lists_joint_exact_given_"
            "function_track_endpoint_joint_exact"
        )
        self.assertEqual(aggregate["raw_numerators"][conditional_label], 1)
        self.assertEqual(aggregate["raw_denominators"][conditional_label], 1)
        self.assertEqual(aggregate["percentages"][conditional_label], 100.0)
        self.assertEqual(
            aggregate["confidence_intervals"][conditional_label],
            released._wilson_95(1, 1),
        )
        self.assertEqual(
            set(aggregate["raw_numerators"]),
            set(aggregate["raw_denominators"]),
        )
        serialized = json.dumps(aggregate, sort_keys=True)
        for forbidden in ("case_id", "query", "do the public task", "Tweet text"):
            self.assertNotIn(forbidden, serialized)
        self.assertNotIn("records", aggregate)
        self.assertNotIn("endpoint_conditioned", serialized)

    def test_field_labels_receive_no_credit_when_endpoint_is_wrong(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._evaluate(
                self._fixture(Path(directory), wrong_field_endpoint=True)
            )
        self.assertEqual(
            aggregate["raw_numerators"]["field_trigger_ordered_exact_endpoint_gated"],
            0,
        )
        self.assertEqual(
            aggregate["percentages"]["field_label_micro_recall_endpoint_gated"],
            50.0,
        )
        conditional_label = (
            "field_track_ordered_label_lists_joint_exact_given_"
            "function_track_endpoint_joint_exact"
        )
        self.assertEqual(aggregate["raw_numerators"][conditional_label], 1)
        self.assertEqual(aggregate["raw_denominators"][conditional_label], 1)

    def test_empty_field_lists_cannot_credit_wrong_endpoint_in_macro_rates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._evaluate(self._fixture(
                Path(directory), wrong_field_endpoint=True,
                field_target=EMPTY_FIELD_TARGET,
            ))
        for metric in ("precision", "recall", "f1"):
            self.assertEqual(aggregate["percentages"][
                f"field_label_macro_{metric}_endpoint_gated"], 0.0)

    def test_correct_empty_field_lists_get_full_macro_credit(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            aggregate = self._evaluate(self._fixture(
                Path(directory), field_target=EMPTY_FIELD_TARGET,
            ))
        self.assertEqual(aggregate["percentages"][
            "field_label_macro_f1_endpoint_gated"], 100.0)

    def test_tampered_archive_fails_before_scoring(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            with paths["archive"].open("ab") as handle:
                handle.write(b"tamper")
            with patch.multiple(
                released,
                EXPECTED_SAMPLE_N=1,
                EXPECTED_POPULATIONS={"gold": 1, "noisy": 1},
                RESULTS_ZIP_SIZE_BYTES=paths["archive"].stat().st_size,
                RESULTS_ZIP_SHA256="0" * 64,
            ):
                with self.assertRaisesRegex(released.RecipeGenReleasedError, "SHA-256"):
                    released.evaluate_released_predictions(
                        archive_path=paths["archive"],
                        processed_csv=paths["processed"],
                        cases_path=paths["cases"],
                        manifest_path=paths["manifest"],
                        split="gold",
                    )

    def test_archive_gold_mapping_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._fixture(Path(directory))
            replacement = Path(directory) / "mismatched.zip"
            with (
                zipfile.ZipFile(paths["archive"], "r") as source,
                zipfile.ZipFile(
                    replacement, "w", compression=zipfile.ZIP_STORED
                ) as target,
            ):
                for info in source.infolist():
                    payload = source.read(info.filename)
                    if info.filename.endswith("oneshot_gold_function/id0.gold"):
                        payload = b"Wrong <sep> Gold\n"
                    target.writestr(info.filename, payload)
            paths["archive"] = replacement
            with self.assertRaisesRegex(
                released.RecipeGenReleasedError, "released gold"
            ):
                self._evaluate(paths)

    def test_exclusive_writer_preserves_existing_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "aggregate.json"
            released.write_aggregate_exclusive(output, {"benchmark_label": "x", "n": 1})
            original = output.read_bytes()
            with self.assertRaisesRegex(released.RecipeGenReleasedError, "overwrite"):
                released.write_aggregate_exclusive(
                    output, {"benchmark_label": "y", "n": 2}
                )
            self.assertEqual(output.read_bytes(), original)
            self.assertEqual(os.stat(output).st_mode & 0o777, 0o600)


if __name__ == "__main__":
    unittest.main()
