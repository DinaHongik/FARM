from __future__ import annotations

import hashlib
import importlib.util
import fcntl
import json
import os
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "launch_bfcl_h21_after_recipegen",
    ROOT / "scripts" / "launch_bfcl_h21_after_recipegen.py",
)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_sha256(identifiers: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, values: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(value, sort_keys=True) + "\n" for value in values),
        encoding="utf-8",
    )


class RecipeGenFixture:
    def __init__(self, root: Path, *, n: int = 150) -> None:
        self.n = n
        self.output = root / "recipegen-output"
        self.output.mkdir(parents=True)
        self.candidates = root / "candidates.jsonl"
        self.candidate_manifest = root / "candidates.manifest.json"
        self.sample_manifest = root / "sample-manifest.json"
        self.pid_file = root / "recipegen.pid"
        self.record_lock = self.output / ".records.lock"
        self.record_lock.touch()
        self.case_ids = [f"public-case-{index:03d}" for index in range(n)]
        _write_jsonl(
            self.candidates,
            [
                {
                    "case_id": case_id,
                    "benchmark": "recipegen_noisy",
                    "data_classification": "public",
                }
                for case_id in self.case_ids
            ],
        )
        self.candidate_sha256 = _sha256(self.candidates)
        _write_json(
            self.candidate_manifest,
            {
                "schema_version": "round9-candidates-manifest-v1",
                "benchmark": "recipegen_noisy",
                "data_classification": "public",
                "rows": n,
                "retrieve_k": 50,
                "top_k": 10,
                "gold_injection": False,
                "rank_hidden_from_models": True,
                "output_sha256": self.candidate_sha256,
            },
        )
        self.candidate_manifest_sha256 = _sha256(self.candidate_manifest)
        self.ordered_ids_sha256 = _ordered_ids_sha256(self.case_ids)
        _write_json(
            self.sample_manifest,
            {
                "schema_version": "round9-sample-manifest-v1",
                "benchmark": "recipegen_noisy",
                "sample_size": n,
                "seed": 9052026,
                "case_payload_sha256": "c" * 64,
                "ordered_case_ids": self.case_ids,
                "ordered_case_ids_sha256": self.ordered_ids_sha256,
            },
        )
        self.run_manifest = self.output / "manifest.json"
        _write_json(
            self.run_manifest,
            {
                "schema_version": "round9-endpoint-run-manifest-v1",
                "benchmark": "recipegen_noisy",
                "arm": "same_model_one_shot",
                "data_classification": "public",
                "data_source": "recipegen",
                "candidate_artifact_sha256": self.candidate_sha256,
                "ordered_case_ids_sha256": self.ordered_ids_sha256,
                "model_metadata": {
                    "name": "deepseek-v4-flash:0731",
                    "provider": "ollama_cloud",
                },
                "protocol_metadata": {
                    "arm": "same_model_one_shot",
                    "semantic_calls_max": 1,
                    "repairs_max": 0,
                    "temperature": 0,
                    "seed": 42,
                },
                "records_file": "records.jsonl",
                "aggregate_file": "aggregate_private.json",
                "public_aggregate_file": "aggregate_public.json",
            },
        )
        self.contract = MODULE.RecipeGenContract(
            expected_n=n,
            candidate_sha256=self.candidate_sha256,
            candidate_manifest_sha256=self.candidate_manifest_sha256,
            ordered_case_ids_sha256=self.ordered_ids_sha256,
            sample_payload_sha256="c" * 64,
        )

    def write_records_and_aggregates(self, *, records_n: int) -> None:
        records = [
            {"case_id": case_id, "attempt": 1, "terminal": True}
            for case_id in self.case_ids[:records_n]
        ]
        records_path = self.output / "records.jsonl"
        _write_jsonl(records_path, records)
        records_sha256 = _sha256(records_path)
        manifest_sha256 = _sha256(self.run_manifest)
        _write_json(
            self.output / "aggregate_private.json",
            {
                "schema_version": "round9-endpoint-aggregate-private-v1",
                "benchmark": "recipegen_noisy",
                "arm": "same_model_one_shot",
                "data_classification": "public",
                "intended_n": self.n,
                "terminal_n": self.n,
                "pending_n": 0,
                "metrics": {"function_joint": {"n": self.n}},
                "records_sha256": records_sha256,
                "manifest_sha256": manifest_sha256,
            },
        )
        _write_json(
            self.output / "aggregate_public.json",
            {
                "n": self.n,
                "raw_numerators": {"function_joint": 100},
                "raw_denominators": {"function_joint": self.n},
                "percentages": {"function_joint": 100 * 100 / self.n},
                "confidence_intervals": {
                    "function_joint": {"low": 0.0, "high": 100.0}
                },
                "failure_counts": {},
                "hashes": {
                    "artifact_sha256": records_sha256,
                    "source_artifact_sha256": self.candidate_sha256,
                    "manifest_sha256": manifest_sha256,
                },
                "benchmark_label": "recipegen_noisy.same_model_one_shot",
                "model_metadata": {
                    "name": "deepseek-v4-flash:0731",
                    "provider": "ollama_cloud",
                },
                "protocol_metadata": {
                    "arm": "same_model_one_shot",
                    "semantic_calls_max": 1,
                    "repairs_max": 0,
                    "temperature": 0,
                    "seed": 42,
                },
                "operational_metrics": {},
            },
        )
        self.pid_file.write_text("99999999\n", encoding="ascii")

    def paths(self) -> object:
        return MODULE.RecipeGenPaths(
            output_directory=self.output,
            candidate_artifact=self.candidates,
            candidate_manifest=self.candidate_manifest,
            sample_manifest=self.sample_manifest,
            pid_file=self.pid_file,
        )


class RecipeGenCompletionTests(unittest.TestCase):
    def test_waits_for_all_150_unique_terminal_records_and_bound_aggregates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecipeGenFixture(Path(directory))
            fixture.write_records_and_aggregates(records_n=149)

            self.assertEqual(
                MODULE.recipegen_completion_status(
                    fixture.paths(),
                    fixture.contract,
                    pid_is_running=lambda _pid: False,
                ),
                (False, "recipegen_records_incomplete"),
            )

            fixture.write_records_and_aggregates(records_n=150)
            self.assertEqual(
                MODULE.recipegen_completion_status(
                    fixture.paths(),
                    fixture.contract,
                    pid_is_running=lambda _pid: False,
                ),
                (True, "recipegen_complete_and_idle"),
            )

    def test_rejects_complete_looking_artifacts_with_wrong_model_or_sample(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecipeGenFixture(Path(directory))
            fixture.write_records_and_aggregates(records_n=150)
            run_manifest = json.loads(fixture.run_manifest.read_text(encoding="utf-8"))
            run_manifest["model_metadata"]["name"] = "wrong-model"
            _write_json(fixture.run_manifest, run_manifest)

            with self.assertRaisesRegex(RuntimeError, "run manifest"):
                MODULE.recipegen_completion_status(
                    fixture.paths(),
                    fixture.contract,
                    pid_is_running=lambda _pid: False,
                )

        with tempfile.TemporaryDirectory() as directory:
            fixture = RecipeGenFixture(Path(directory))
            fixture.write_records_and_aggregates(records_n=150)
            sample_manifest = json.loads(
                fixture.sample_manifest.read_text(encoding="utf-8")
            )
            sample_manifest["ordered_case_ids_sha256"] = "0" * 64
            _write_json(fixture.sample_manifest, sample_manifest)

            with self.assertRaisesRegex(RuntimeError, "sample"):
                MODULE.recipegen_completion_status(
                    fixture.paths(),
                    fixture.contract,
                    pid_is_running=lambda _pid: False,
                )

    def test_waits_for_stopped_process_and_idle_record_lock(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            fixture = RecipeGenFixture(Path(directory))
            fixture.write_records_and_aggregates(records_n=150)
            self.assertEqual(
                MODULE.recipegen_completion_status(
                    fixture.paths(),
                    fixture.contract,
                    pid_is_running=lambda _pid: True,
                ),
                (False, "recipegen_process_still_running"),
            )

            descriptor = os.open(fixture.record_lock, os.O_RDWR)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                self.assertEqual(
                    MODULE.recipegen_completion_status(
                        fixture.paths(),
                        fixture.contract,
                        pid_is_running=lambda _pid: False,
                    ),
                    (False, "recipegen_record_lock_busy"),
                )
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)


class CloudCapacityTests(unittest.TestCase):
    def test_reserves_exactly_two_slots_atomically_and_releases_them(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease_directory = Path(directory) / "leases"
            lease_directory.mkdir(mode=0o700)
            held_path = lease_directory / "ollama-cloud-0.slot"
            held_descriptor = os.open(held_path, os.O_RDWR | os.O_CREAT, 0o600)
            fcntl.flock(held_descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            try:
                with MODULE.reserve_cloud_capacity(
                    lease_directory, required_slots=2
                ) as available:
                    self.assertTrue(available)
                    locked = 0
                    for slot in range(3):
                        descriptor = os.open(
                            lease_directory / f"ollama-cloud-{slot}.slot",
                            os.O_RDWR,
                        )
                        try:
                            try:
                                fcntl.flock(
                                    descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB
                                )
                            except BlockingIOError:
                                locked += 1
                            else:
                                fcntl.flock(descriptor, fcntl.LOCK_UN)
                        finally:
                            os.close(descriptor)
                    self.assertEqual(locked, 3)
            finally:
                fcntl.flock(held_descriptor, fcntl.LOCK_UN)
                os.close(held_descriptor)

            with MODULE.reserve_cloud_capacity(
                lease_directory, required_slots=3
            ) as available:
                self.assertTrue(available)
            self.assertEqual(os.stat(lease_directory).st_mode & 0o777, 0o700)
            self.assertTrue(
                all(
                    os.stat(path).st_mode & 0o777 == 0o600
                    for path in lease_directory.glob("*.slot")
                )
            )

    def test_does_not_partially_reserve_when_two_slots_are_not_free(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            lease_directory = Path(directory) / "leases"
            lease_directory.mkdir(mode=0o700)
            held_descriptors: list[int] = []
            try:
                for slot in (0, 1):
                    path = lease_directory / f"ollama-cloud-{slot}.slot"
                    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    held_descriptors.append(descriptor)
                with MODULE.reserve_cloud_capacity(
                    lease_directory, required_slots=2
                ) as available:
                    self.assertFalse(available)

                contender = os.open(
                    lease_directory / "ollama-cloud-2.slot", os.O_RDWR
                )
                try:
                    fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    fcntl.flock(contender, fcntl.LOCK_UN)
                finally:
                    os.close(contender)
            finally:
                for descriptor in held_descriptors:
                    fcntl.flock(descriptor, fcntl.LOCK_UN)
                    os.close(descriptor)


class BFCLProtocolAndCacheTests(unittest.TestCase):
    def test_accepts_only_the_frozen_h21_protocol_manifest(self) -> None:
        contract = MODULE.validate_bfcl_protocol_manifest(
            ROOT / "manifests" / "bfcl_native_official_horizon_v3.json"
        )
        self.assertEqual(contract.registry_name, "farm-r9-bfcl-dsv4-agent-v3h21")
        self.assertEqual(contract.source_registry, "farm-r9-bfcl-dsv4-agent-v2")
        self.assertEqual(contract.workers, 2)
        self.assertEqual(contract.cache_entry_count, 1299)

        with tempfile.TemporaryDirectory() as directory:
            changed = Path(directory) / "changed.json"
            manifest = json.loads(
                (
                    ROOT / "manifests" / "bfcl_native_official_horizon_v3.json"
                ).read_text(encoding="utf-8")
            )
            manifest["model"] = "wrong-model"
            _write_json(changed, manifest)
            with self.assertRaisesRegex(RuntimeError, "protocol manifest"):
                MODULE.validate_bfcl_protocol_manifest(changed)

    def test_requires_a_distinct_complete_cache_copy_with_bound_inventory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            destination = root / "destination"
            source.mkdir(mode=0o700)
            destination.mkdir(mode=0o700)
            for index, payload in enumerate((b"one\n", b"two\n")):
                source_file = source / f"{index:02d}.json"
                destination_file = destination / f"{index:02d}.json"
                source_file.write_bytes(payload)
                destination_file.write_bytes(payload)
                os.chmod(source_file, 0o600)
                os.chmod(destination_file, 0o600)
            count, digest = MODULE.cache_inventory(source)
            contract = MODULE.BFCLContract(
                registry_name="fresh-registry",
                source_registry="source-registry",
                workers=2,
                cache_entry_count=count,
                cache_inventory_sha256=digest,
            )
            self.assertEqual(
                MODULE.bfcl_cache_copy_status(source, destination, contract),
                (True, "bfcl_cache_copy_verified"),
            )

            (destination / "01.json").write_bytes(b"changed\n")
            with self.assertRaisesRegex(RuntimeError, "inventory"):
                MODULE.bfcl_cache_copy_status(source, destination, contract)

    def test_waits_for_missing_destination_but_rejects_a_missing_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir(mode=0o700)
            source_file = source / "entry.json"
            source_file.write_text("{}\n", encoding="utf-8")
            os.chmod(source_file, 0o600)
            count, digest = MODULE.cache_inventory(source)
            contract = MODULE.BFCLContract(
                registry_name="fresh-registry",
                source_registry="source-registry",
                workers=2,
                cache_entry_count=count,
                cache_inventory_sha256=digest,
            )
            self.assertEqual(
                MODULE.bfcl_cache_copy_status(
                    source, root / "destination", contract
                ),
                (False, "bfcl_cache_copy_pending"),
            )
            with self.assertRaisesRegex(RuntimeError, "source cache"):
                MODULE.bfcl_cache_copy_status(
                    root / "missing-source", root / "destination", contract
                )


if __name__ == "__main__":
    unittest.main()
