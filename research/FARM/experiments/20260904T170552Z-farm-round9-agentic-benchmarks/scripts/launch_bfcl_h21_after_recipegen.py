#!/usr/bin/env python3
"""Read-only preparation checks retained from the former BFCL queue design.

Production launch uses launch_bfcl_official_horizon_v3_dgx.sh. This utility
audits the frozen cache/protocol and does not start or stop any experiment.
"""

from __future__ import annotations

import errno
import fcntl
import hashlib
import json
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Iterator, NamedTuple


class RecipeGenContract(NamedTuple):
    expected_n: int
    candidate_sha256: str
    candidate_manifest_sha256: str
    ordered_case_ids_sha256: str
    sample_payload_sha256: str


class RecipeGenPaths(NamedTuple):
    output_directory: Path
    candidate_artifact: Path
    candidate_manifest: Path
    sample_manifest: Path
    pid_file: Path


class BFCLContract(NamedTuple):
    registry_name: str
    source_registry: str
    workers: int
    cache_entry_count: int
    cache_inventory_sha256: str


def validate_bfcl_protocol_manifest(path: Path) -> BFCLContract:
    value = _read_json(path)
    expected = {
        "arm": "native_tool_agent", "benchmark": "bfcl_v4_multi_turn_miss_param",
        "fresh_registry_name": "farm-r9-bfcl-dsv4-agent-v3h21", "n": 150,
        "model": "deepseek-v4-flash:0731", "workers": 2, "temperature": 0,
        "seed": 9052026, "thinking_mode": "low", "max_physical_calls_per_turn": 21,
        "official_execution_step_limit": 20,
        "official_repository_commit": "f7cf7359b7ac615a0b294831c5ba2bc95ee4a000",
        "sample_sha256": "e2848d00c5f4b9781e162a60cc66b1f0c795763de3491b26fce3abbb22ab264e",
    }
    cache = value.get("cache_reuse", {})
    expected_cache = {
        "expected_entry_count_before_launch": 1299,
        "inventory_sha256": "6838ac903a985907c6685b4244427ecb4c6a604f664fdfb9fad9a3de9d1d3df9",
        "mode": "copy_into_fresh_v3_cache_before_launch",
        "source_registry": "farm-r9-bfcl-dsv4-agent-v2",
    }
    if any(value.get(key) != item for key, item in expected.items()) or cache != expected_cache:
        raise RuntimeError("BFCL protocol manifest differs from the frozen correction")
    return BFCLContract(expected["fresh_registry_name"], cache["source_registry"], 2,
                        cache["expected_entry_count_before_launch"], cache["inventory_sha256"])


def cache_inventory(path: Path) -> tuple[int, str]:
    if not path.is_dir() or path.is_symlink():
        raise RuntimeError("cache inventory directory is absent or unsafe")
    files = sorted(path.glob("*.json"), key=lambda item: item.name)
    digest = hashlib.sha256()
    for item in files:
        if item.is_symlink() or not item.is_file() or "\n" in item.name:
            raise RuntimeError("unsafe cache inventory entry")
        digest.update(f"{_sha256_file(item)}  {item.name}\n".encode("utf-8"))
    return len(files), digest.hexdigest()


def bfcl_cache_copy_status(source: Path, destination: Path, contract: BFCLContract) -> tuple[bool, str]:
    if not source.is_dir() or source.is_symlink():
        raise RuntimeError("BFCL source cache is missing or unsafe")
    expected = (contract.cache_entry_count, contract.cache_inventory_sha256)
    if cache_inventory(source) != expected:
        raise RuntimeError("BFCL source cache inventory changed")
    if not destination.exists():
        return False, "bfcl_cache_copy_pending"
    if source.resolve() == destination.resolve():
        raise RuntimeError("BFCL cache copy must use a distinct directory")
    if cache_inventory(destination) != expected:
        raise RuntimeError("BFCL copied cache inventory differs")
    return True, "bfcl_cache_copy_verified"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ordered_ids_sha256(identifiers: list[str]) -> str:
    return hashlib.sha256(
        ("\n".join(identifiers) + "\n").encode("utf-8")
    ).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError("a required RecipeGen artifact is not a JSON object")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError("a RecipeGen JSONL artifact contains a non-object")
            rows.append(value)
    return rows


def _pid_is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _record_lock_is_idle(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        raise RuntimeError("RecipeGen record lock is missing or unsafe")
    flags = os.O_RDWR
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return False
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        return True
    finally:
        os.close(descriptor)


@contextmanager
def reserve_cloud_capacity(
    lease_directory: Path,
    *,
    required_slots: int = 2,
) -> Iterator[bool]:
    """Hold exactly ``required_slots`` limiter locks, or hold none.

    This is only a launch gate. BFCL's request-level limiter remains the
    authoritative concurrency control after these descriptors are released.
    """

    if isinstance(required_slots, bool) or not 1 <= required_slots <= 3:
        raise ValueError("required Cloud slots must be an integer in 1..3")
    if lease_directory.is_symlink():
        raise RuntimeError("Cloud lease directory must not be a symlink")
    lease_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(lease_directory, 0o700)
    held: list[int] = []
    try:
        for slot in range(3):
            path = lease_directory / f"ollama-cloud-{slot}.slot"
            flags = os.O_RDWR | os.O_CREAT
            if hasattr(os, "O_CLOEXEC"):
                flags |= os.O_CLOEXEC
            if hasattr(os, "O_NOFOLLOW"):
                flags |= os.O_NOFOLLOW
            descriptor = os.open(path, flags, 0o600)
            os.fchmod(descriptor, 0o600)
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(descriptor)
                continue
            held.append(descriptor)
            if len(held) == required_slots:
                break
        if len(held) != required_slots:
            for descriptor in held:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
                os.close(descriptor)
            held.clear()
            yield False
            return
        yield True
    finally:
        for descriptor in held:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def _validate_recipegen_bindings(
    paths: RecipeGenPaths,
    contract: RecipeGenContract,
    *,
    candidate_rows: list[dict[str, Any]],
    records: list[dict[str, Any]],
    run_manifest: dict[str, Any],
    private_aggregate: dict[str, Any],
    public_aggregate: dict[str, Any],
) -> None:
    candidate_sha256 = _sha256_file(paths.candidate_artifact)
    if candidate_sha256 != contract.candidate_sha256:
        raise RuntimeError("RecipeGen candidate artifact binding changed")
    if _sha256_file(paths.candidate_manifest) != contract.candidate_manifest_sha256:
        raise RuntimeError("RecipeGen candidate manifest binding changed")

    candidate_manifest = _read_json(paths.candidate_manifest)
    expected_candidate_manifest = {
        "schema_version": "round9-candidates-manifest-v1",
        "benchmark": "recipegen_noisy",
        "data_classification": "public",
        "rows": contract.expected_n,
        "retrieve_k": 50,
        "top_k": 10,
        "gold_injection": False,
        "rank_hidden_from_models": True,
        "output_sha256": candidate_sha256,
    }
    for key, expected in expected_candidate_manifest.items():
        if candidate_manifest.get(key) != expected:
            raise RuntimeError("RecipeGen candidate manifest contract changed")

    sample_manifest = _read_json(paths.sample_manifest)
    expected_sample = {
        "schema_version": "round9-sample-manifest-v1",
        "benchmark": "recipegen_noisy",
        "sample_size": contract.expected_n,
        "seed": 9052026,
        "case_payload_sha256": contract.sample_payload_sha256,
        "ordered_case_ids_sha256": contract.ordered_case_ids_sha256,
    }
    for key, expected in expected_sample.items():
        if sample_manifest.get(key) != expected:
            raise RuntimeError("RecipeGen frozen sample binding changed")
    sample_ids = sample_manifest.get("ordered_case_ids")
    if (
        not isinstance(sample_ids, list)
        or len(sample_ids) != contract.expected_n
        or not all(isinstance(case_id, str) and case_id for case_id in sample_ids)
        or len(set(sample_ids)) != contract.expected_n
        or _ordered_ids_sha256(sample_ids) != contract.ordered_case_ids_sha256
    ):
        raise RuntimeError("RecipeGen frozen sample IDs are invalid")

    candidate_ids = [row.get("case_id") for row in candidate_rows]
    if (
        len(candidate_rows) != contract.expected_n
        or candidate_ids != sample_ids
        or any(row.get("benchmark") != "recipegen_noisy" for row in candidate_rows)
        or any(row.get("data_classification") != "public" for row in candidate_rows)
    ):
        raise RuntimeError("RecipeGen candidate-to-sample binding changed")

    expected_run = {
        "schema_version": "round9-endpoint-run-manifest-v1",
        "benchmark": "recipegen_noisy",
        "arm": "same_model_one_shot",
        "data_classification": "public",
        "data_source": "recipegen",
        "candidate_artifact_sha256": candidate_sha256,
        "ordered_case_ids_sha256": contract.ordered_case_ids_sha256,
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
    }
    for key, expected in expected_run.items():
        if run_manifest.get(key) != expected:
            raise RuntimeError("RecipeGen run manifest contract changed")

    record_ids = [row.get("case_id") for row in records]
    if any(not isinstance(case_id, str) or not case_id for case_id in record_ids):
        raise RuntimeError("RecipeGen records contain an invalid case ID")
    if len(set(record_ids)) != contract.expected_n:
        raise RuntimeError("RecipeGen completed record IDs do not match n=150")
    if set(record_ids) != set(sample_ids):
        raise RuntimeError("RecipeGen record-to-sample binding changed")
    latest: dict[str, dict[str, Any]] = {}
    for row in records:
        attempt = row.get("attempt")
        if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 1:
            raise RuntimeError("RecipeGen record attempt is invalid")
        case_id = str(row["case_id"])
        previous = latest.get(case_id)
        if previous is None or attempt > previous["attempt"]:
            latest[case_id] = row
        elif attempt == previous["attempt"] and row != previous:
            raise RuntimeError("RecipeGen records contain an ambiguous attempt")
    if any(row.get("terminal") is not True for row in latest.values()):
        raise RuntimeError("RecipeGen latest records are not terminal")

    records_sha256 = _sha256_file(paths.output_directory / "records.jsonl")
    manifest_sha256 = _sha256_file(paths.output_directory / "manifest.json")
    expected_private = {
        "benchmark": "recipegen_noisy",
        "arm": "same_model_one_shot",
        "data_classification": "public",
        "intended_n": contract.expected_n,
        "terminal_n": contract.expected_n,
        "pending_n": 0,
        "records_sha256": records_sha256,
        "manifest_sha256": manifest_sha256,
    }
    for key, expected in expected_private.items():
        if private_aggregate.get(key) != expected:
            raise RuntimeError("RecipeGen private aggregate binding changed")
    private_metrics = private_aggregate.get("metrics")
    if not isinstance(private_metrics, dict) or not private_metrics:
        raise RuntimeError("RecipeGen private aggregate metrics are incomplete")
    if any(
        not isinstance(metric, dict) or metric.get("n") != contract.expected_n
        for metric in private_metrics.values()
    ):
        raise RuntimeError("RecipeGen private metric denominator changed")

    if public_aggregate.get("n") != contract.expected_n:
        raise RuntimeError("RecipeGen public aggregate denominator changed")
    expected_public = {
        "benchmark_label": "recipegen_noisy.same_model_one_shot",
        "model_metadata": expected_run["model_metadata"],
        "protocol_metadata": expected_run["protocol_metadata"],
    }
    for key, expected in expected_public.items():
        if public_aggregate.get(key) != expected:
            raise RuntimeError("RecipeGen public aggregate contract changed")
    denominators = public_aggregate.get("raw_denominators")
    if not isinstance(denominators, dict) or not denominators:
        raise RuntimeError("RecipeGen public aggregate metrics are incomplete")
    if any(value != contract.expected_n for value in denominators.values()):
        raise RuntimeError("RecipeGen public metric denominator changed")
    hashes = public_aggregate.get("hashes")
    if not isinstance(hashes, dict) or hashes != {
        "artifact_sha256": records_sha256,
        "source_artifact_sha256": candidate_sha256,
        "manifest_sha256": manifest_sha256,
    }:
        raise RuntimeError("RecipeGen public aggregate hash binding changed")


def recipegen_completion_status(
    paths: RecipeGenPaths,
    contract: RecipeGenContract,
    *,
    pid_is_running: Callable[[int], bool] = _pid_is_running,
) -> tuple[bool, str]:
    required = (
        paths.candidate_artifact,
        paths.candidate_manifest,
        paths.sample_manifest,
        paths.output_directory / "manifest.json",
        paths.output_directory / "records.jsonl",
        paths.output_directory / "aggregate_private.json",
        paths.output_directory / "aggregate_public.json",
    )
    if any(not path.is_file() or path.is_symlink() for path in required):
        return False, "recipegen_artifacts_pending"
    try:
        candidate_rows = _read_jsonl(paths.candidate_artifact)
        records = _read_jsonl(paths.output_directory / "records.jsonl")
        run_manifest = _read_json(paths.output_directory / "manifest.json")
        private_aggregate = _read_json(
            paths.output_directory / "aggregate_private.json"
        )
        public_aggregate = _read_json(
            paths.output_directory / "aggregate_public.json"
        )
    except (OSError, json.JSONDecodeError):
        return False, "recipegen_artifacts_not_stable"
    record_ids = {
        row.get("case_id")
        for row in records
        if isinstance(row.get("case_id"), str) and row.get("case_id")
    }
    if len(record_ids) < contract.expected_n:
        return False, "recipegen_records_incomplete"
    _validate_recipegen_bindings(
        paths,
        contract,
        candidate_rows=candidate_rows,
        records=records,
        run_manifest=run_manifest,
        private_aggregate=private_aggregate,
        public_aggregate=public_aggregate,
    )
    if not paths.pid_file.is_file() or paths.pid_file.is_symlink():
        raise RuntimeError("RecipeGen producer PID file is missing or unsafe")
    try:
        pid = int(paths.pid_file.read_text(encoding="ascii").strip())
    except (OSError, UnicodeError, ValueError) as error:
        raise RuntimeError("RecipeGen producer PID file is invalid") from error
    if pid <= 1:
        raise RuntimeError("RecipeGen producer PID is invalid")
    if pid_is_running(pid):
        return False, "recipegen_process_still_running"
    if not _record_lock_is_idle(paths.output_directory / ".records.lock"):
        return False, "recipegen_record_lock_busy"
    return True, "recipegen_complete_and_idle"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--source-cache", type=Path, required=True)
    parser.add_argument("--destination-cache", type=Path, required=True)
    args = parser.parse_args()
    contract = validate_bfcl_protocol_manifest(args.protocol_manifest)
    ready, status = bfcl_cache_copy_status(args.source_cache, args.destination_cache, contract)
    print(json.dumps({"ready": ready, "status": status, "launch_performed": False}))
