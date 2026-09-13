from __future__ import annotations

import json
import hashlib
import pickle
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from farm_r9.adapters.yao_interactive import (
    YAO_CONVERTED_SCHEMA,
    YAO_PICKLE_SHA256,
    YAO_REPOSITORY_COMMIT,
    convert_loaded_yao_data,
    freeze_simulator_answer,
    prepare_yao_interactive,
)
from convert_yao_official import load_verified_pickle


class _EncoderFixture:
    def __init__(self, classes: list[bytes]) -> None:
        self.classes_ = classes


class YaoAdapterTests(unittest.TestCase):
    def test_converter_checks_hash_then_rejects_non_whitelisted_pickle_globals(self) -> None:
        payload = pickle.dumps(Path("a-path-global"), protocol=2)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.pkl"
            source.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                load_verified_pickle(source, expected_sha256="0" * 64)
            with self.assertRaisesRegex(pickle.UnpicklingError, "forbidden global"):
                load_verified_pickle(
                    source,
                    expected_sha256=hashlib.sha256(payload).hexdigest(),
                )

    def test_experiment_adapter_refuses_pickle_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pickle_path = Path(directory) / "official.pkl"
            pickle_path.write_bytes(b"not loaded")

            with self.assertRaisesRegex(ValueError, "converted JSON"):
                prepare_yao_interactive(pickle_path)

    def test_simulator_answer_is_frozen_by_the_registered_hash_rule(self) -> None:
        frozen = freeze_simulator_answer(
            recipe_id="160256",
            component_index=3,
            answer_options=["first", "second", "third"],
            ask_ordinal=0,
        )
        self.assertEqual(frozen, {"answer_index": 2, "answer": "third"})

    def test_converted_population_uses_fixed_ambiguity_quotas(self) -> None:
        tag_counts = {"CI": 727, "VI-1": 453, "VI-2": 818, "VI-3": 272, "VI-4": 1600}
        converted_rows = []
        source_index = 0
        for tag, count in tag_counts.items():
            for _ in range(count):
                recipe_id = str(100000 + source_index)
                converted_rows.append({
                    "case_id": f"yao:{recipe_id}",
                    "recipe_id": recipe_id,
                    "request": f"request {recipe_id}",
                    "words": ["request", recipe_id],
                    "gold": {
                        "trigger_channel": "Trigger Service",
                        "trigger_function": "Trigger Function",
                        "action_channel": "Action Service",
                        "action_function": "Action Function",
                    },
                    "ambiguity_tag": tag,
                    "pseudo_ask_labels": [0, 1, 0, 1],
                    "simulator_answers": {
                        "trigger_channel": ["trigger service"],
                        "trigger_function": ["trigger answer one", "trigger answer two"],
                        "action_channel": ["action service"],
                        "action_function": ["action answer one", "action answer two"],
                    },
                    "valid_endpoint_constraints": {
                        "trigger_functions_for_service": ["Trigger Function"],
                        "trigger_services_for_function": ["Trigger Service"],
                        "action_functions_for_service": ["Action Function"],
                        "action_services_for_function": ["Action Service"],
                    },
                    "source_index": source_index,
                })
                source_index += 1
        converted = {
            "schema_version": YAO_CONVERTED_SCHEMA,
            "source": {
                "repository_commit": YAO_REPOSITORY_COMMIT,
                "pickle_sha256": YAO_PICKLE_SHA256,
            },
            "test": converted_rows,
        }

        with tempfile.TemporaryDirectory() as directory:
            converted_path = Path(directory) / "yao.json"
            converted_path.write_text(json.dumps(converted), encoding="utf-8")
            first, manifest = prepare_yao_interactive(converted_path)
            second, _ = prepare_yao_interactive(converted_path)

        self.assertEqual(len(first), 150)
        self.assertEqual([row["case_id"] for row in first], [row["case_id"] for row in second])
        self.assertEqual(
            manifest["sample_strata"],
            {"CI": 28, "VI-1": 18, "VI-2": 32, "VI-3": 10, "VI-4": 62},
        )
        self.assertEqual(manifest["sampling"], "yao-ambiguity-stratified-sha256-v1")
        self.assertTrue(all(row["benchmark"] == "interactive_ifttt" for row in first))
        self.assertTrue(all("pickle" not in json.dumps(row).casefold() for row in first))
        self.assertTrue(all(row["simulator"]["frozen_answers"] for row in first))

    def test_loaded_official_objects_convert_to_inert_case_specific_json(self) -> None:
        raw = {
            "label_types": ("trigger_chans", "trigger_funcs", "action_chans", "action_funcs"),
            "num_labels": [1, 1, 1, 1],
            "labelers": {
                "trigger_chans": _EncoderFixture([b"Weather"]),
                "trigger_funcs": _EncoderFixture([b"Today's report"]),
                "action_chans": _EncoderFixture([b"Notification"]),
                "action_funcs": _EncoderFixture([b"Send notification"]),
            },
            "word_ids": {"weather": 0, "report": 1, "notify": 2},
            "train": [{"recipe_id": "train-1"}],
            "dev": [{"recipe_id": "dev-1"}],
            "sample_dev": [],
            "test": [{
                "recipe_id": "160256",
                "words": ["weather", "report"],
                "ids": [0, 1],
                "labels": [0, 0, 0, 0],
                "label_names": [b"Weather", b"Today's report", b"Notification", b"Send notification"],
                "pseudo_ask_labels": [0, 1, 0, 1],
                "tags": ["VI-2"],
            }],
            "user_answers": [
                {"Weather": [[0]]},
                {"weather.today's report": [[0, 1], [1]]},
                {"Notification": [[2]]},
                {"notification.send notification": [[2], [1, 2]]},
            ],
            "chnl_fn_constraints": [
                {0: [0]},
                {0: [0]},
                {0: [0]},
                {0: [0]},
            ],
        }

        converted = convert_loaded_yao_data(
            raw,
            source_sha256=YAO_PICKLE_SHA256,
            require_official_shape=False,
        )

        self.assertEqual(converted["schema_version"], YAO_CONVERTED_SCHEMA)
        self.assertEqual(converted["test"][0]["request"], "weather report")
        self.assertEqual(
            converted["test"][0]["simulator_answers"]["trigger_function"],
            ["weather report", "report"],
        )
        self.assertEqual(
            converted["test"][0]["valid_endpoint_constraints"],
            {
                "trigger_functions_for_service": ["Today's report"],
                "trigger_services_for_function": ["Weather"],
                "action_functions_for_service": ["Send notification"],
                "action_services_for_function": ["Notification"],
            },
        )
        self.assertNotIn("train", converted)
        json.dumps(converted)


if __name__ == "__main__":
    unittest.main()
