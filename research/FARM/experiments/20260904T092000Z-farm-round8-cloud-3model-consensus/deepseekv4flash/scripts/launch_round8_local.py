#!/usr/bin/env python3
"""Prepare five GPU-isolated local Ollama daemons and launch Round8 arms.

This is a strictly loopback path.  It reuses the verified server on port 11434,
starts only absent servers on ports 11435--11438, verifies the exact Granite
model digest on every port, and only then starts five detached experiment tmux
sessions.  It never pulls a model, stops a daemon, or kills an experiment.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ID = "20260904T074517Z-farm-round8-executable-agent"
DATASET_ID = "f12eb4abab5cce04947c6d8f57ebc807f5eb7bb5bec402d6e1b078cf7e1aec8d"
MODEL = "granite4:small-h"
MODEL_DIGEST = "2f8a7367d4416b508336f6170c295e8465ff004731ae563631196781299f750d"
DEFAULT_PYTHON = Path("/raid/session/aicontents/farm/.venv/bin/python")
DEFAULT_MODELS = Path("/raid/session/aicontents/farm/.ollama_models")
DEFAULT_OLLAMA = Path("/raid/session/aicontents/.local/ollama/bin/ollama")
ARMS = (
    "typed_direct",
    "typed_schema_plan",
    "typed_execute_repair",
    "dspy_factorized",
    "trigger_consensus_executable",
)
PROVENANCE_FILES = (
    "EXPERIMENT_MATRIX.json",
    "scripts/run_round8.py",
    "scripts/dspy_executable_agent.py",
    "scripts/executable_applet.py",
    "scripts/frozen_router.py",
)
FULL_SCREEN_ROWS = 231
FULL_SCREEN_IDS_SHA256 = (
    "087c4f5d87c357952a3a94c02abe359710d06be6927859b1644da092cbf87f14"
)
SMOKE_ROWS = 2
SMOKE_IDS_SHA256 = (
    "cb3cf029b4873c2b04d01d68d87ba750810106255369a4c59afe1ef5c027ebd2"
)
CONSENSUS_SMOKE_ROWS = 5
CONSENSUS_SMOKE_IDS_SHA256 = (
    "6b0df973b07158ef9233256531ab86532556f4b06cdeb7779787a55ed382f121"
)


@dataclass(frozen=True)
class Slot:
    arm: str
    gpu: int
    port: int

    @property
    def host(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def experiment_session(self) -> str:
        return f"farm-r8-local-{self.arm.replace('_', '-')}"

    def experiment_session_for(self, smoke: int | None) -> str:
        if smoke is None:
            return self.experiment_session
        return f"farm-r8-local-smoke-{smoke}-{self.arm.replace('_', '-')}"

    @property
    def server_session(self) -> str:
        return f"farm-r8-ollama-{self.port}"


SLOTS = tuple(Slot(arm=arm, gpu=index, port=11434 + index) for index, arm in enumerate(ARMS))


def local_model_root(run_root: Path) -> Path:
    return run_root / "local_granite" / MODEL_DIGEST


def local_artifact_root(run_root: Path, smoke: int | None = None) -> Path:
    if smoke is not None and smoke not in {SMOKE_ROWS, CONSENSUS_SMOKE_ROWS}:
        raise ValueError("the frozen local smoke sizes are 2 and 5")
    return local_model_root(run_root) / ("full" if smoke is None else f"smoke-{smoke}")


def local_server_root(run_root: Path) -> Path:
    return local_model_root(run_root) / "servers"


def smoke_size_for_arm(arm: str, smoke_gate: int | None) -> int | None:
    if smoke_gate is None:
        return None
    if smoke_gate != SMOKE_ROWS:
        raise ValueError(f"the frozen local smoke gate is {SMOKE_ROWS}")
    return CONSENSUS_SMOKE_ROWS if arm == "trigger_consensus_executable" else SMOKE_ROWS


def selection_identity(smoke: int | None) -> tuple[int, str]:
    if smoke is None:
        return FULL_SCREEN_ROWS, FULL_SCREEN_IDS_SHA256
    if smoke == SMOKE_ROWS:
        return SMOKE_ROWS, SMOKE_IDS_SHA256
    if smoke == CONSENSUS_SMOKE_ROWS:
        return CONSENSUS_SMOKE_ROWS, CONSENSUS_SMOKE_IDS_SHA256
    raise ValueError("the frozen local smoke sizes are 2 and 5")


@dataclass(frozen=True)
class OllamaProbe:
    port: int
    tcp_open: bool
    api_ok: bool
    model_present: bool
    digest_ok: bool
    observed_digest: str | None
    error: str | None

    @property
    def ready(self) -> bool:
        return self.tcp_open and self.api_ok and self.model_present and self.digest_ok


def _tmux_has_session(name: str) -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", f"={name}"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def _read_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        return pid if pid > 1 else None
    except (FileNotFoundError, OSError, ValueError):
        return None


def _pid_alive(pid: int | None) -> bool:
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _process_command(pid: int) -> str:
    try:
        return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(
            "utf-8", errors="replace"
        )
    except OSError:
        return ""


def _process_environment(pid: int) -> dict[str, str]:
    try:
        raw = Path(f"/proc/{pid}/environ").read_bytes().split(b"\0")
    except OSError as error:
        raise RuntimeError(f"cannot inspect environment for server pid {pid}") from error
    environment: dict[str, str] = {}
    for item in raw:
        if b"=" not in item:
            continue
        key, value = item.split(b"=", 1)
        environment[key.decode(errors="replace")] = value.decode(errors="replace")
    return environment


def _tcp_open(port: int, timeout: float = 0.5) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        return connection.connect_ex(("127.0.0.1", port)) == 0


def probe_ollama(port: int, *, timeout: float = 3.0) -> OllamaProbe:
    """Read `/api/tags` on loopback and require the full frozen digest."""

    tcp = _tcp_open(port, min(timeout, 0.5))
    if not tcp:
        return OllamaProbe(port, False, False, False, False, None, "tcp_closed")
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/api/tags",
        method="GET",
        headers={"Accept": "application/json"},
    )
    # Explicit empty proxy mapping prevents proxy environment variables from
    # ever receiving loopback experiment traffic.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            if response.status != 200:
                return OllamaProbe(port, True, False, False, False, None, f"http_{response.status}")
            payload = json.loads(response.read().decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, urllib.error.URLError) as error:
        return OllamaProbe(port, True, False, False, False, None, type(error).__name__)
    models = payload.get("models") if isinstance(payload, Mapping) else None
    if not isinstance(models, list):
        return OllamaProbe(port, True, False, False, False, None, "invalid_tags_payload")
    record = next(
        (
            item
            for item in models
            if isinstance(item, Mapping) and (item.get("model") or item.get("name")) == MODEL
        ),
        None,
    )
    if record is None:
        return OllamaProbe(port, True, True, False, False, None, "model_absent")
    digest = record.get("digest")
    observed = digest if isinstance(digest, str) else None
    return OllamaProbe(
        port=port,
        tcp_open=True,
        api_ok=True,
        model_present=True,
        digest_ok=observed == MODEL_DIGEST,
        observed_digest=observed,
        error=None if observed == MODEL_DIGEST else "digest_mismatch",
    )


def _read_json_object(path: Path) -> Mapping[str, Any] | None:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"malformed JSON artifact {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    return value


def provenance_hashes(run_root: Path) -> dict[str, str]:
    return {
        relative: hashlib.sha256((run_root / relative).read_bytes()).hexdigest()
        for relative in PROVENANCE_FILES
    }


def runner_binding_mismatches(
    run_root: Path, slot: Slot, binding: object, *, smoke: int | None = None
) -> tuple[str, ...]:
    """Check the immutable subset shared by local ops and the experiment runner."""

    if not isinstance(binding, Mapping):
        return ("binding",)
    hashes = provenance_hashes(run_root)
    executed_rows, executed_ids_sha256 = selection_identity(smoke)
    expected = {
        "run_id": RUN_ID,
        "mode": "local_granite" if smoke is None else "local_granite_smoke",
        "arm": slot.arm,
        "dataset_id": DATASET_ID,
        "matrix_sha256": hashes["EXPERIMENT_MATRIX.json"],
        "runner_sha256": hashes["scripts/run_round8.py"],
        "dspy_executable_agent_sha256": hashes["scripts/dspy_executable_agent.py"],
        "executable_applet_sha256": hashes["scripts/executable_applet.py"],
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "data_left_dgx": False,
    }
    mismatches = [key for key, wanted in expected.items() if binding.get(key) != wanted]
    selection = binding.get("selection")
    if not isinstance(selection, Mapping):
        mismatches.append("selection")
    else:
        for key, wanted in {
            "executed_rows": executed_rows,
            "executed_group_ids_sha256": executed_ids_sha256,
        }.items():
            if selection.get(key) != wanted:
                mismatches.append(f"selection.{key}")
    protocol = binding.get("protocol")
    if not isinstance(protocol, Mapping):
        mismatches.append("protocol")
    else:
        for key, wanted in {
            "transport": "local_loopback",
            "data_left_dgx": False,
            "gold_at_inference_boundary": False,
            "live_connectors": False,
        }.items():
            if protocol.get(key) != wanted:
                mismatches.append(f"protocol.{key}")
    if slot.arm == "trigger_consensus_executable":
        policy = binding.get("trigger_consensus_policy")
        if not isinstance(policy, Mapping):
            mismatches.append("trigger_consensus_policy")
        elif policy.get("router_runtime_sha256") != hashes[
            "scripts/frozen_router.py"
        ]:
            mismatches.append(
                "trigger_consensus_policy.router_runtime_sha256"
            )
    return tuple(sorted(set(mismatches)))


def local_launch_binding(
    run_root: Path, slot: Slot, *, smoke: int | None = None
) -> dict[str, Any]:
    artifact_hashes = provenance_hashes(run_root)
    executed_rows, executed_ids_sha256 = selection_identity(smoke)
    namespace = "full" if smoke is None else f"smoke-{smoke}"
    return {
        "schema_version": "farm_round8_local_launch_binding_v3",
        "run_id": RUN_ID,
        "arm": slot.arm,
        "gpu": slot.gpu,
        "port": slot.port,
        "host": slot.host,
        "model": MODEL,
        "model_digest": MODEL_DIGEST,
        "artifact_root": f"local_granite/{MODEL_DIGEST}/{namespace}",
        "artifacts_sha256": artifact_hashes,
        "selection": {
            "executed_rows": executed_rows,
            "executed_group_ids_sha256": executed_ids_sha256,
        },
        "loopback_only": True,
        "data_left_dgx": False,
    }


def ensure_launch_binding(
    run_root: Path, slot: Slot, *, smoke: int | None = None
) -> None:
    path = local_artifact_root(run_root, smoke) / "state" / f"{slot.arm}.launch.json"
    expected = local_launch_binding(run_root, slot, smoke=smoke)
    existing = _read_json_object(path)
    if existing is not None:
        if dict(existing) != expected:
            raise RuntimeError(f"local launch binding changed for {slot.arm}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        json.dump(expected, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def preflight_arm(
    run_root: Path, slot: Slot, *, smoke: int | None = None
) -> bool:
    """Return whether resumable partial state exists; refuse active/complete ambiguity."""

    artifacts = local_artifact_root(run_root, smoke)
    session = slot.experiment_session_for(smoke)
    if _tmux_has_session(session):
        raise RuntimeError(f"active local experiment session exists: {session}")
    result_path = artifacts / "results" / f"{slot.arm}.json"
    result = _read_json_object(result_path)
    if result is not None:
        raise RuntimeError(f"completed local result exists for {slot.arm}; refusing rerun")
    progress_path = artifacts / "progress" / f"{slot.arm}.json"
    progress = _read_json_object(progress_path)
    if progress is not None:
        phase = str(progress.get("phase", progress.get("status", ""))).casefold()
        if phase in {"complete", "completed", "success", "succeeded"}:
            raise RuntimeError(f"progress for {slot.arm} says complete without a result")
        mismatches = runner_binding_mismatches(
            run_root, slot, progress.get("binding"), smoke=smoke
        )
        if mismatches:
            raise RuntimeError(
                f"resume binding changed for {slot.arm}: fields={list(mismatches)}"
            )
    pid_path = artifacts / "pids" / f"{slot.arm}.pid"
    pid = _read_pid(pid_path)
    if pid_path.exists() and pid is None:
        raise RuntimeError(f"malformed local experiment pid file: {pid_path}")
    if _pid_alive(pid):
        assert pid is not None
        command = _process_command(pid)
        if str(run_root) in command and slot.arm in command and slot.host in command:
            raise RuntimeError(f"live local experiment process exists for {slot.arm}: pid={pid}")
        raise RuntimeError(f"pid file for {slot.arm} points to an unrelated live process")
    partials = (
        progress_path,
        artifacts / "records" / f"{slot.arm}.jsonl",
        artifacts / "attempts" / f"{slot.arm}.jsonl",
        artifacts / "logs" / f"{slot.arm}.log",
        pid_path,
    )
    partial = any(path.exists() for path in partials)
    binding_path = artifacts / "state" / f"{slot.arm}.launch.json"
    launch_binding = _read_json_object(binding_path)
    if launch_binding is not None and dict(launch_binding) != local_launch_binding(
        run_root, slot, smoke=smoke
    ):
        raise RuntimeError(f"frozen local launch identity changed for {slot.arm}")
    if partial and launch_binding is None:
        raise RuntimeError(
            f"partial local artifacts for {slot.arm} lack a frozen launch binding; refusing unsafe resume"
        )
    return partial


def _is_valid_native_tool_response(call: object) -> bool:
    """Recognize one successfully parsed response from the local native API."""

    return (
        isinstance(call, Mapping)
        and call.get("ok") is True
        and call.get("schema_valid") is True
        and type(call.get("transport_attempts")) is int
        and call["transport_attempts"] == 1
        and type(call.get("model_tool_calls")) is int
        and call["model_tool_calls"] == 1
        and type(call.get("provider_requests")) is int
        and call["provider_requests"] == 1
        and call.get("error_code") is None
    )


def _is_compile_execute_success(record: Mapping[str, Any]) -> bool:
    """Require the complete model-to-typed-program-to-sandbox success chain."""

    execution = record.get("execution")
    receipt = execution.get("receipt_sha256") if isinstance(execution, Mapping) else None
    return (
        record.get("protocol_valid") is True
        and record.get("terminal_status") == "executed"
        and record.get("fallback_used") is False
        and isinstance(record.get("program"), Mapping)
        and isinstance(execution, Mapping)
        and execution.get("strict_parse") is True
        and isinstance(execution.get("compilation_attempts"), int)
        and not isinstance(execution.get("compilation_attempts"), bool)
        and execution["compilation_attempts"] >= 1
        and execution.get("compiled") is True
        and isinstance(execution.get("sandbox_attempts"), int)
        and not isinstance(execution.get("sandbox_attempts"), bool)
        and execution["sandbox_attempts"] >= 1
        and execution.get("sandbox_run") is True
        and isinstance(receipt, str)
        and re.fullmatch(r"[0-9a-f]{64}", receipt) is not None
    )


def validate_smoke_semantics(slot: Slot, records: Sequence[Mapping[str, Any]]) -> None:
    """Fail closed when a completed smoke did not test the executable agent path."""

    native_protocol_success = False
    consensus_nonzero_call = False
    consensus_routed = False
    consensus_two_view = False
    for number, record in enumerate(records, 1):
        calls = record.get("calls")
        accounting = record.get("accounting")
        if not isinstance(calls, list) or not isinstance(accounting, Mapping):
            raise RuntimeError(
                f"invalid smoke call accounting for {slot.arm} at record {number}"
            )
        semantic_calls = accounting.get("semantic_calls")
        if (
            not isinstance(semantic_calls, int)
            or isinstance(semantic_calls, bool)
            or semantic_calls < 0
            or semantic_calls != len(calls)
        ):
            raise RuntimeError(
                f"invalid smoke semantic call count for {slot.arm} at record {number}"
            )
        if slot.arm != "trigger_consensus_executable" and not calls:
            raise RuntimeError(
                f"smoke did not exercise the expected native call for {slot.arm} "
                f"at record {number}"
            )
        record_native_valid = any(
            _is_valid_native_tool_response(call) for call in calls
        )
        record_protocol_success = (
            bool(calls)
            and record.get("protocol_valid") is True
            and all(_is_valid_native_tool_response(call) for call in calls)
        )
        native_protocol_success = native_protocol_success or record_protocol_success
        if isinstance(record.get("program"), Mapping) and not _is_compile_execute_success(
            record
        ):
            raise RuntimeError(
                f"smoke emitted an unauthenticated executable program for {slot.arm} "
                f"at record {number}"
            )
        if slot.arm == "trigger_consensus_executable":
            trace = record.get("policy_trace")
            if not isinstance(trace, Mapping):
                raise RuntimeError(
                    f"consensus smoke lacks a policy trace at record {number}"
                )
            routed = trace.get("routed") is True
            consensus_routed = consensus_routed or routed
            consensus_nonzero_call = consensus_nonzero_call or (
                routed and bool(calls) and semantic_calls > 0
            )
            consensus_two_view = consensus_two_view or (
                routed
                and trace.get("schema_m5_called") is True
                and trace.get("fused_m10_called") is True
                and semantic_calls == 2
                and record_protocol_success
            )
    if slot.arm == "trigger_consensus_executable":
        if not consensus_routed:
            raise RuntimeError("consensus smoke did not exercise a routed case")
        if not consensus_nonzero_call:
            raise RuntimeError("consensus routed smoke made no native tool call")
        if not consensus_two_view:
            raise RuntimeError("consensus smoke did not exercise one valid two-view call chain")
    if not native_protocol_success:
        raise RuntimeError(
            f"smoke produced no complete valid native protocol chain for {slot.arm}"
        )


def validate_completed_artifacts(
    run_root: Path, slot: Slot, *, smoke: int | None = None
) -> Mapping[str, Any]:
    """Authenticate a completed arm before treating a smoke/full gate as met."""

    artifacts = local_artifact_root(run_root, smoke)
    expected_rows, expected_ids_sha256 = selection_identity(smoke)
    result_path = artifacts / "results" / f"{slot.arm}.json"
    progress_path = artifacts / "progress" / f"{slot.arm}.json"
    records_path = artifacts / "records" / f"{slot.arm}.jsonl"
    launch_path = artifacts / "state" / f"{slot.arm}.launch.json"
    result = _read_json_object(result_path)
    progress = _read_json_object(progress_path)
    if result is None or progress is None or not records_path.is_file():
        raise RuntimeError(f"completed artifacts are absent for {slot.arm}")
    launch_binding = _read_json_object(launch_path)
    if launch_binding is None or dict(launch_binding) != local_launch_binding(
        run_root, slot, smoke=smoke
    ):
        raise RuntimeError(f"completed launch identity changed for {slot.arm}")
    result_binding = result.get("binding")
    if result.get("status") != "completed":
        raise RuntimeError(f"result is not completed for {slot.arm}")
    mismatches = runner_binding_mismatches(
        run_root, slot, result_binding, smoke=smoke
    )
    if mismatches:
        raise RuntimeError(
            f"completed binding changed for {slot.arm}: fields={list(mismatches)}"
        )
    metrics = result.get("metrics")
    if not isinstance(metrics, Mapping) or metrics.get("rows") != expected_rows:
        raise RuntimeError(f"completed metric row count changed for {slot.arm}")
    if progress.get("phase") != "complete" or progress.get("binding") != result_binding:
        raise RuntimeError(f"completed progress binding changed for {slot.arm}")
    if (
        progress.get("completed_rows") != expected_rows
        or progress.get("target_rows") != expected_rows
    ):
        raise RuntimeError(f"completed progress row count changed for {slot.arm}")
    output = progress.get("output")
    if not isinstance(output, str) or Path(output).resolve() != result_path.resolve():
        raise RuntimeError(f"completed output path changed for {slot.arm}")
    if progress.get("output_sha256") != hashlib.sha256(result_path.read_bytes()).hexdigest():
        raise RuntimeError(f"completed result hash changed for {slot.arm}")

    group_ids: list[str] = []
    seen: set[str] = set()
    completed_records: list[Mapping[str, Any]] = []
    for number, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"malformed completed record for {slot.arm} at line {number}"
            ) from error
        group_id = record.get("group_id") if isinstance(record, Mapping) else None
        if (
            not isinstance(group_id, str)
            or not group_id
            or group_id in seen
            or record.get("schema_version") != "farm_round8_executable_record_v1"
            or record.get("arm_id") != slot.arm
        ):
            raise RuntimeError(
                f"invalid completed record identity for {slot.arm} at line {number}"
            )
        seen.add(group_id)
        group_ids.append(group_id)
        completed_records.append(record)
    observed_ids_sha256 = hashlib.sha256(
        json.dumps(group_ids).encode("utf-8")
    ).hexdigest()
    if len(group_ids) != expected_rows or observed_ids_sha256 != expected_ids_sha256:
        raise RuntimeError(f"completed record selection changed for {slot.arm}")
    if smoke is not None:
        validate_smoke_semantics(slot, completed_records)
    return result


def query_gpu_processes() -> dict[int, tuple[int, ...]]:
    """Return compute PIDs by physical GPU index or fail closed."""

    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 2:
            uuid_to_index[cells[1]] = int(cells[0])
    process_result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    processes: dict[int, list[int]] = {index: [] for index in uuid_to_index.values()}
    for line in process_result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) != 2 or cells[0] not in uuid_to_index:
            continue
        processes[uuid_to_index[cells[0]]].append(int(cells[1]))
    return {index: tuple(sorted(pids)) for index, pids in processes.items()}


def query_gpu_compute_apps() -> dict[int, tuple[dict[str, Any], ...]]:
    """Read compute PID/name/memory telemetry keyed by physical GPU index."""

    gpu_result = subprocess.run(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        capture_output=True,
        text=True,
        check=True,
    )
    uuid_to_index: dict[str, int] = {}
    for line in gpu_result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",")]
        if len(cells) == 2:
            uuid_to_index[cells[1]] = int(cells[0])
    process_result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    apps: dict[int, list[dict[str, Any]]] = {
        index: [] for index in uuid_to_index.values()
    }
    for line in process_result.stdout.splitlines():
        cells = [cell.strip() for cell in line.split(",", 3)]
        if len(cells) != 4 or cells[0] not in uuid_to_index:
            continue
        try:
            pid = int(cells[1])
        except ValueError:
            continue
        try:
            used_memory_mib: int | None = int(cells[3])
        except ValueError:
            used_memory_mib = None
        apps[uuid_to_index[cells[0]]].append(
            {
                "pid": pid,
                "process_name": cells[2],
                "used_memory_mib": used_memory_mib,
            }
        )
    return {
        gpu: tuple(sorted(records, key=lambda item: int(item["pid"])))
        for gpu, records in apps.items()
    }


def listening_pid(port: int) -> int:
    result = subprocess.run(
        ["ss", "-H", "-ltnp", "sport", "=", f":{port}"],
        capture_output=True,
        text=True,
        check=True,
    )
    pids = {int(value) for value in re.findall(r"\bpid=(\d+)", result.stdout)}
    if len(pids) != 1:
        raise RuntimeError(f"cannot bind loopback port {port} to exactly one listener PID")
    return pids.pop()


def _pid_descends_from(pid: int, ancestor: int) -> bool:
    current = pid
    seen: set[int] = set()
    for _ in range(64):
        if current == ancestor:
            return True
        if current <= 1 or current in seen:
            return False
        seen.add(current)
        try:
            stat = Path(f"/proc/{current}/stat").read_text(encoding="utf-8")
            suffix = stat.rsplit(")", 1)[1].strip().split()
            current = int(suffix[1])
        except (IndexError, OSError, ValueError):
            return False
    return False


def gpu_isolation_snapshot(run_root: Path) -> dict[str, Any]:
    """Prove each loaded Ollama worker belongs only to its assigned V100."""

    listener_pids: dict[int, int] = {}
    errors: list[str] = []
    conflicts: list[str] = []
    for slot in SLOTS:
        try:
            listener_pids[slot.port] = listening_pid(slot.port)
        except (RuntimeError, subprocess.SubprocessError) as error:
            errors.append(str(error))
            conflicts.append(str(error))
    apps = query_gpu_compute_apps()
    reserved_apps = [
        (gpu, app)
        for gpu, records in apps.items()
        if 0 <= gpu < len(SLOTS)
        for app in records
    ]
    server_rows: list[dict[str, Any]] = []
    owned_compute_pids: set[int] = set()
    for slot in SLOTS:
        server_pid = listener_pids.get(slot.port)
        if slot.port != 11434 and server_pid is not None:
            tracked_pid = _read_pid(local_server_root(run_root) / f"{slot.port}.pid")
            if tracked_pid != server_pid:
                issue = f"port {slot.port} listener PID does not match its managed PID"
                errors.append(issue)
                conflicts.append(issue)
        workers: list[dict[str, Any]] = []
        if server_pid is not None:
            for gpu, app in reserved_apps:
                pid = int(app["pid"])
                if _pid_descends_from(pid, server_pid):
                    workers.append({"gpu": gpu, **app})
                    owned_compute_pids.add(pid)
        observed_gpus = sorted({int(item["gpu"]) for item in workers})
        if server_pid is not None and not observed_gpus:
            errors.append(f"port {slot.port} has no loaded GPU worker")
        elif server_pid is not None and observed_gpus != [slot.gpu]:
            issue = (
                f"port {slot.port} expected loaded GPU {slot.gpu}, "
                f"observed {observed_gpus}"
            )
            errors.append(issue)
            conflicts.append(issue)
        server_rows.append(
            {
                "port": slot.port,
                "expected_gpu": slot.gpu,
                "listener_pid": server_pid,
                "observed_gpus": observed_gpus,
                "compute_workers": workers,
            }
        )
    unexpected = [
        {"gpu": gpu, **app}
        for gpu, app in reserved_apps
        if int(app["pid"]) not in owned_compute_pids
    ]
    if unexpected:
        issue = "unowned compute processes occupy reserved GPUs 0--4"
        errors.append(issue)
        conflicts.append(issue)
    return {
        "verified": not errors,
        "errors": errors,
        "conflicts": conflicts,
        "servers": server_rows,
        "unexpected_compute_apps": unexpected,
    }


def validate_managed_server(run_root: Path, models: Path, slot: Slot) -> None:
    """Validate the GPU/model-store environment for a managed port 11435--11438."""

    pid_path = local_server_root(run_root) / f"{slot.port}.pid"
    pid = _read_pid(pid_path)
    if not _pid_alive(pid):
        raise RuntimeError(f"ready port {slot.port} is not owned by a tracked Round8 server")
    assert pid is not None
    environment = _process_environment(pid)
    observed_models = environment.get("OLLAMA_MODELS")
    if environment.get("CUDA_VISIBLE_DEVICES") != str(slot.gpu):
        raise RuntimeError(f"server {slot.port} is not isolated to GPU {slot.gpu}")
    if not observed_models or Path(observed_models).resolve() != models.resolve():
        raise RuntimeError(f"server {slot.port} does not use the shared OLLAMA_MODELS")
    if environment.get("OLLAMA_HOST") not in {
        f"127.0.0.1:{slot.port}",
        f"http://127.0.0.1:{slot.port}",
    }:
        raise RuntimeError(f"server {slot.port} is not bound to its frozen loopback address")


def server_launch_command(
    run_root: Path,
    python: Path,
    ollama_bin: Path,
    models: Path,
    slot: Slot,
) -> list[str]:
    return [
        "tmux",
        "new-session",
        "-d",
        "-s",
        slot.server_session,
        "-c",
        str(run_root),
        str(python),
        str(run_root / "scripts" / "run_local_ollama_server.py"),
        "--run-root",
        str(run_root),
        "--ollama-bin",
        str(ollama_bin),
        "--ollama-models",
        str(models),
        "--port",
        str(slot.port),
        "--gpu",
        str(slot.gpu),
    ]


def experiment_launch_command(
    run_root: Path, python: Path, slot: Slot, *, smoke: int | None = None
) -> list[str]:
    command = [
        "tmux",
        "new-session",
        "-d",
        "-s",
        slot.experiment_session_for(smoke),
        "-c",
        str(run_root),
        str(python),
        str(run_root / "scripts" / "run_round8_local_job.py"),
        "--run-root",
        str(run_root),
        "--python",
        str(python),
        "--arm",
        slot.arm,
        "--gpu",
        str(slot.gpu),
        "--port",
        str(slot.port),
        "--model",
        MODEL,
        "--expected-digest",
        MODEL_DIGEST,
        "--resume",
    ]
    if smoke is not None:
        command.extend(("--smoke", str(smoke)))
    return command


def validate_runtime(run_root: Path, python: Path, ollama_bin: Path, models: Path) -> None:
    required = (
        run_root / "ROUND8_SCOPE.md",
        run_root / "EXPERIMENT_MATRIX.json",
        run_root / "scripts" / "run_round8.py",
        run_root / "scripts" / "dspy_executable_agent.py",
        run_root / "scripts" / "executable_applet.py",
        run_root / "scripts" / "run_round8_local_job.py",
        run_root / "scripts" / "run_local_ollama_server.py",
        run_root / "scripts" / "verify_round8_local.py",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise RuntimeError(f"missing local Round8 prerequisites: {missing}")
    if run_root.name != RUN_ID:
        raise RuntimeError(f"unexpected Round8 run root: {run_root}")
    matrix = _read_json_object(run_root / "EXPERIMENT_MATRIX.json")
    assert matrix is not None
    screen = matrix.get("screen")
    if matrix.get("run_id") != RUN_ID or not isinstance(screen, Mapping):
        raise RuntimeError("Round8 matrix identity or screen metadata changed")
    matrix_arms = tuple(
        item.get("id")
        for item in matrix.get("arms", ())
        if isinstance(item, Mapping)
    )
    if matrix_arms != ARMS:
        raise RuntimeError(
            f"Round8 matrix arms do not match local slots: observed={matrix_arms}"
        )
    if (
        screen.get("family_purged_rows") != FULL_SCREEN_ROWS
        or screen.get("group_ids_sha256") != FULL_SCREEN_IDS_SHA256
    ):
        raise RuntimeError("Round8 full-screen row identity changed")
    if not python.is_file() or not os.access(python, os.X_OK):
        raise RuntimeError(f"FARM Python is unavailable: {python}")
    if not (run_root / ".deps" / "dspy" / "__init__.py").is_file():
        raise RuntimeError("Round8 isolated dependencies are missing under .deps")
    if not ollama_bin.is_file() or not os.access(ollama_bin, os.X_OK):
        raise RuntimeError(f"Ollama executable is unavailable: {ollama_bin}")
    if not models.is_dir():
        raise RuntimeError(f"shared OLLAMA_MODELS directory is unavailable: {models}")
    for executable in ("tmux", "nvidia-smi", "ss"):
        if shutil.which(executable) is None:
            raise RuntimeError(f"required executable is unavailable: {executable}")


def wait_for_server(slot: Slot, timeout: float, interval: float) -> OllamaProbe:
    deadline = time.monotonic() + timeout
    last = probe_ollama(slot.port)
    while not last.ready and time.monotonic() < deadline:
        time.sleep(min(interval, max(0.0, deadline - time.monotonic())))
        last = probe_ollama(slot.port)
    return last


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=root)
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument(
        "--ollama-bin",
        type=Path,
        default=DEFAULT_OLLAMA,
    )
    parser.add_argument("--ollama-models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--server-ready-timeout", type=float, default=120.0)
    parser.add_argument("--server-poll-interval", type=float, default=2.0)
    parser.add_argument("--verify-initial-delay", type=float, default=3.0)
    parser.add_argument("--verify-interval", type=float, default=15.0)
    parser.add_argument(
        "--smoke",
        type=int,
        choices=(SMOKE_ROWS,),
        help="launch only the frozen two-case smoke gate",
    )
    parser.add_argument(
        "--smoke-gate",
        type=int,
        choices=(SMOKE_ROWS,),
        default=SMOKE_ROWS,
        help="completed smoke size required before a full launch",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if args.server_ready_timeout <= 0 or args.server_poll_interval <= 0:
        parser.error("server readiness timing must be positive")
    if args.verify_initial_delay < 0 or args.verify_interval <= 0:
        parser.error("verification timing is invalid")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    python = args.python.resolve()
    ollama_bin = args.ollama_bin.resolve()
    models = args.ollama_models.resolve()
    validate_runtime(run_root, python, ollama_bin, models)
    state_root = local_model_root(run_root) / "state"
    if args.dry_run:
        lock_context: Any = contextlib.nullcontext(None)
    else:
        state_root.mkdir(parents=True, exist_ok=True)
        lock_context = (state_root / "launch.lock").open("a", encoding="utf-8")
    with lock_context as lock:
        if lock is not None:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(
                    "another local Round8 launcher owns the launch lock"
                ) from error

        if args.smoke is None:
            for slot in SLOTS:
                gate_smoke = smoke_size_for_arm(slot.arm, args.smoke_gate)
                assert gate_smoke is not None
                smoke_session = slot.experiment_session_for(gate_smoke)
                smoke_pid = _read_pid(
                    local_artifact_root(run_root, gate_smoke)
                    / "pids"
                    / f"{slot.arm}.pid"
                )
                if _tmux_has_session(smoke_session) or _pid_alive(smoke_pid):
                    raise RuntimeError(
                        f"smoke gate is still active for {slot.arm}; full launch is blocked"
                    )
                validate_completed_artifacts(
                    run_root, slot, smoke=gate_smoke
                )
            print(
                f"SMOKE_GATE_OK rows=2/5 arms={len(SLOTS)} "
                f"digest={MODEL_DIGEST}"
            )

        dispositions: dict[str, str] = {}
        target_smokes: dict[str, int | None] = {}
        launch_slots: list[Slot] = []
        for slot in SLOTS:
            target_smoke = smoke_size_for_arm(slot.arm, args.smoke)
            target_smokes[slot.arm] = target_smoke
            artifact_root = local_artifact_root(run_root, target_smoke)
            result_path = artifact_root / "results" / f"{slot.arm}.json"
            if result_path.exists():
                session = slot.experiment_session_for(target_smoke)
                pid = _read_pid(artifact_root / "pids" / f"{slot.arm}.pid")
                if _tmux_has_session(session) or _pid_alive(pid):
                    raise RuntimeError(
                        f"completed local result is still active for {slot.arm}"
                    )
                validate_completed_artifacts(run_root, slot, smoke=target_smoke)
                dispositions[slot.arm] = "validated-complete"
                continue
            partial = preflight_arm(run_root, slot, smoke=target_smoke)
            dispositions[slot.arm] = "resume-partial" if partial else "fresh"
            launch_slots.append(slot)

        probes = {slot.port: probe_ollama(slot.port) for slot in SLOTS}
        if not probes[11434].ready:
            raise RuntimeError(
                "the pre-existing GPU0 Ollama server on 127.0.0.1:11434 must be "
                f"healthy with the exact Granite digest; observed={probes[11434]}"
            )

        gpu_processes = query_gpu_processes()
        server_commands: list[tuple[Slot, list[str]]] = []
        waiting_sessions: list[Slot] = []
        for slot in SLOTS[1:]:
            probe = probes[slot.port]
            server_pid = _read_pid(local_server_root(run_root) / f"{slot.port}.pid")
            session_active = _tmux_has_session(slot.server_session)
            if probe.ready:
                validate_managed_server(run_root, models, slot)
                print(f"REUSE_SERVER port={slot.port} gpu={slot.gpu} digest={MODEL_DIGEST}")
                continue
            if probe.tcp_open:
                raise RuntimeError(
                    f"port {slot.port} is occupied but failed exact Ollama/model preflight: {probe}"
                )
            if session_active:
                waiting_sessions.append(slot)
                continue
            if _pid_alive(server_pid):
                raise RuntimeError(
                    f"server pid for closed port {slot.port} is alive without its tmux session"
                )
            busy = gpu_processes.get(slot.gpu, ())
            if busy:
                raise RuntimeError(
                    f"GPU {slot.gpu} has existing compute processes {busy}; refusing Ollama start"
                )
            server_commands.append(
                (slot, server_launch_command(run_root, python, ollama_bin, models, slot))
            )

        experiment_commands = [
            (
                slot,
                experiment_launch_command(
                    run_root, python, slot, smoke=target_smokes[slot.arm]
                ),
            )
            for slot in launch_slots
        ]
        for slot in SLOTS:
            artifact_root = local_artifact_root(run_root, target_smokes[slot.arm])
            print(
                f"PREFLIGHT arm={slot.arm} gpu={slot.gpu} host={slot.host} "
                f"namespace={artifact_root.name} disposition={dispositions[slot.arm]}"
            )
        for slot, command in server_commands:
            if args.dry_run:
                print("DRY_RUN_SERVER " + " ".join(command))
            else:
                subprocess.run(command, check=True)
                waiting_sessions.append(slot)
                print(f"STARTED_SERVER session={slot.server_session} port={slot.port} gpu={slot.gpu}")
        if args.dry_run:
            for _, command in experiment_commands:
                print("DRY_RUN_EXPERIMENT " + " ".join(command))
            print("No server or experiment tmux session was launched.")
            return 0

        for slot in waiting_sessions:
            probe = wait_for_server(
                slot,
                timeout=args.server_ready_timeout,
                interval=args.server_poll_interval,
            )
            if not probe.ready:
                raise RuntimeError(
                    f"server {slot.port} did not expose the exact model digest before timeout: {probe}"
                )
            validate_managed_server(run_root, models, slot)

        for slot in SLOTS:
            final_probe = probe_ollama(slot.port)
            if not final_probe.ready:
                raise RuntimeError(f"final port/digest preflight failed for {slot.host}: {final_probe}")
        if args.smoke is None:
            isolation = gpu_isolation_snapshot(run_root)
            print("GPU_ISOLATION " + json.dumps(isolation, sort_keys=True))
            if not isolation["verified"]:
                raise RuntimeError(
                    "post-smoke GPU isolation is not proven; refusing full launch: "
                    f"{isolation['errors']}"
                )
        for slot in launch_slots:
            ensure_launch_binding(
                run_root, slot, smoke=target_smokes[slot.arm]
            )
        for slot, command in experiment_commands:
            target_smoke = target_smokes[slot.arm]
            subprocess.run(command, check=True)
            print(
                f"LAUNCHED_LOCAL arm={slot.arm} "
                f"session={slot.experiment_session_for(target_smoke)} "
                f"gpu={slot.gpu} host={slot.host}"
            )

    verify = [
        str(python),
        str(run_root / "scripts" / "verify_round8_local.py"),
        "--run-root",
        str(run_root),
        "--ollama-models",
        str(models),
        "--samples",
        "2",
        "--initial-delay",
        str(args.verify_initial_delay),
        "--interval",
        str(args.verify_interval),
        "--expect",
        "running-or-complete",
    ]
    if args.smoke is not None:
        verify.extend(("--smoke", str(args.smoke)))
    completed = subprocess.run(verify, check=False)
    if completed.returncode:
        print(
            "Local verification failed; servers and experiments were intentionally left untouched.",
            file=sys.stderr,
        )
    return completed.returncode


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"ROUND8_LOCAL_LAUNCH_REFUSED: {error}", file=sys.stderr)
        raise SystemExit(2) from error
