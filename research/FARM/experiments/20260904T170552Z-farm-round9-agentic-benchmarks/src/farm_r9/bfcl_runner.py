"""Pinned BFCL V4 Missing Parameters generation and evaluation adapter.

The official BFCL ``BaseHandler.inference_multi_turn_FC`` loop owns tool
execution and state transitions.  This module supplies only an Ollama-native
transport and result persistence around that loop; it does not reimplement the
BFCL evaluator.
"""

from __future__ import annotations

import copy
import fcntl
import json
import math
import os
import re
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from farm_r9.adapters.bfcl import (
    BFCL_BENCHMARK,
    BFCL_REPOSITORY_COMMIT,
    BFCL_SAMPLE_SIZE,
)
from farm_r9.artifact_io import (
    canonical_json,
    ordered_ids_sha256,
    read_json,
    read_jsonl,
    sha256_bytes,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.privacy import (
    DataClassification,
    DataSource,
    export_public_aggregate,
    redact_public_trace,
    require_cloud_route,
)


BFCL_CATEGORY = "multi_turn_miss_param"
BFCL_SCOPE = "partial:150_of_200"
BFCL_WORKER_POLICY = "thread-pool-atomic-case-resume-v1"
# At the pinned BFCL commit, ``inference_multi_turn_FC`` increments its step
# counter *after* executing a model response and force-quits when ``count > 20``.
# A protocol-transparent transport must therefore allow calls 0..20 (21 total)
# so BFCL, rather than this adapter, owns the terminal horizon.
BFCL_OFFICIAL_EXECUTION_STEP_LIMIT = 20
BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN = 21


class BFCLProtocolError(RuntimeError):
    """A protocol invariant failed; callers must record a failed case."""


class BFCLTransportError(RuntimeError):
    """A physical Ollama request failed after its bounded retries."""


class BFCLTrackedCaseError(BFCLProtocolError):
    """A terminal case failure carrying only privacy-safe call telemetry."""

    def __init__(self, failure_type: str, operations: Sequence[Mapping[str, Any]]):
        super().__init__("BFCL case failed after tracked model operations")
        self.failure_type = failure_type
        self.operations = tuple(copy.deepcopy(dict(item)) for item in operations)


class BFCLArm(str, Enum):
    SINGLE_STEP = "single_step_per_turn_fc"
    NATIVE_AGENT = "native_tool_agent"


@dataclass(frozen=True)
class ArmPolicy:
    arm: BFCLArm
    max_physical_calls_per_turn: int = 4

    def __post_init__(self) -> None:
        if isinstance(self.max_physical_calls_per_turn, bool) or not isinstance(
            self.max_physical_calls_per_turn, int
        ):
            raise TypeError("max_physical_calls_per_turn must be an integer")
        if not (
            1
            <= self.max_physical_calls_per_turn
            <= BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN
        ):
            raise ValueError(
                "max_physical_calls_per_turn must be in 1.."
                f"{BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN}"
            )

    def decision(self, calls_this_turn: int) -> str:
        """Return ``physical``, ``synthetic_stop``, or raise on budget failure."""
        if calls_this_turn < 0:
            raise ValueError("calls_this_turn cannot be negative")
        if self.arm is BFCLArm.SINGLE_STEP:
            return "physical" if calls_this_turn == 0 else "synthetic_stop"
        if calls_this_turn >= self.max_physical_calls_per_turn:
            raise BFCLProtocolError(
                "native agent exceeded its frozen per-turn call budget"
            )
        return "physical"


@dataclass(frozen=True)
class NormalizedToolCall:
    call_id: str
    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class NormalizedChatResponse:
    content: str
    tool_calls: tuple[NormalizedToolCall, ...]
    prompt_tokens: int
    completion_tokens: int
    latency_seconds: float
    synthetic: bool = False
    queue_wait_seconds: float = 0.0
    cache_hit: bool = False
    physical_attempts: int = 1


HTTPTransport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def _default_http_transport(
    endpoint: str, headers: Mapping[str, str], body: bytes, timeout: float
) -> tuple[int, bytes]:
    request = urllib.request.Request(
        endpoint, data=body, headers=dict(headers), method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        raise BFCLTransportError(type(error).__name__) from error


def _ollama_chat_endpoint(host: str) -> str:
    parsed = urlsplit(host)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise ValueError(
            "Ollama Cloud host must be an authenticated-key-free HTTPS origin"
        )
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama Cloud host cannot contain query or fragment")
    base = host.rstrip("/")
    return base + ("/chat" if base.endswith("/api") else "/api/chat")


def normalize_ollama_response(
    raw: Mapping[str, Any],
    *,
    expected_model: str,
    call_namespace: str,
    latency_seconds: float,
    queue_wait_seconds: float = 0.0,
    cache_hit: bool = False,
    physical_attempts: int = 1,
) -> NormalizedChatResponse:
    if raw.get("model") != expected_model:
        raise BFCLProtocolError("Ollama response model mismatch")
    message = raw.get("message")
    if not isinstance(message, Mapping):
        raise BFCLProtocolError("Ollama response message is missing")
    content = message.get("content") or ""
    calls = message.get("tool_calls") or []
    if not isinstance(content, str) or not isinstance(calls, list):
        raise BFCLProtocolError("Ollama response message fields are invalid")

    normalized: list[NormalizedToolCall] = []
    for index, call in enumerate(calls):
        if not isinstance(call, Mapping) or not isinstance(
            call.get("function"), Mapping
        ):
            raise BFCLProtocolError("Ollama tool call is malformed")
        function = call["function"]
        name, arguments = function.get("name"), function.get("arguments", {})
        if not isinstance(name, str) or not name:
            raise BFCLProtocolError("Ollama tool call name is invalid")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError as error:
                raise BFCLProtocolError(
                    "Ollama tool arguments are not valid JSON"
                ) from error
        if not isinstance(arguments, dict):
            raise BFCLProtocolError("Ollama tool arguments must be an object")
        call_id = call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = "call_" + sha256_text(f"{call_namespace}:{index}")[:24]
        normalized.append(NormalizedToolCall(call_id, name, copy.deepcopy(arguments)))

    return NormalizedChatResponse(
        content=content,
        tool_calls=tuple(normalized),
        prompt_tokens=int(raw.get("prompt_eval_count") or 0),
        completion_tokens=int(raw.get("eval_count") or 0),
        latency_seconds=float(latency_seconds),
        queue_wait_seconds=float(queue_wait_seconds),
        cache_hit=bool(cache_hit),
        physical_attempts=int(physical_attempts),
    )


class BFCLCloudTransport:
    """Ollama Cloud transport whose persistent journal never contains payloads."""

    def __init__(
        self,
        *,
        host: str,
        api_key: str,
        model: str,
        limiter: OllamaCloudLimiter,
        journal_path: Path,
        cache_directory: Path,
        timeout_seconds: float = 240.0,
        max_transport_attempts: int = 2,
        temperature: float = 0.0,
        seed: int = 9052026,
        think: str | None = "low",
        http_transport: HTTPTransport = _default_http_transport,
    ) -> None:
        if not api_key:
            raise ValueError("Ollama Cloud API key is required")
        if not model:
            raise ValueError("Ollama model is required")
        if max_transport_attempts not in {1, 2, 3}:
            raise ValueError("max_transport_attempts must be in 1..3")
        require_cloud_route(
            DataClassification.PUBLIC,
            benchmark_label=BFCL_BENCHMARK,
            source=DataSource.BFCL_V4,
        )
        self.endpoint = _ollama_chat_endpoint(host)
        self.api_key = api_key
        self.model = model
        self.limiter = limiter
        self.journal_path = journal_path
        self.cache_directory = cache_directory
        self.cache_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.cache_directory, 0o700)
        self.timeout_seconds = float(timeout_seconds)
        self.max_transport_attempts = max_transport_attempts
        self.temperature = float(temperature)
        self.seed = int(seed)
        self.think = think
        self.http_transport = http_transport

    def chat(
        self,
        *,
        semantic_id: str,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]],
    ) -> NormalizedChatResponse:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [dict(message) for message in messages],
            "stream": False,
            "options": {"temperature": self.temperature, "seed": self.seed},
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if self.think is not None:
            payload["think"] = self.think
        body = canonical_json(payload).encode("utf-8")
        request_hash = sha256_bytes(body)
        semantic_hash = sha256_text(semantic_id)
        model_hash = sha256_text(self.model)
        cache_path = self.cache_directory / f"{semantic_hash}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            if (
                not isinstance(cached, Mapping)
                or cached.get("semantic_id_sha256") != semantic_hash
                or cached.get("request_sha256") != request_hash
                or cached.get("model_sha256") != model_hash
                or not isinstance(cached.get("response"), Mapping)
            ):
                raise BFCLProtocolError("BFCL semantic cache binding mismatch")
            self._journal("cache_hit", semantic_id, request_hash, 0)
            return normalize_ollama_response(
                cached["response"],
                expected_model=self.model,
                call_namespace=semantic_id,
                # This invocation did not contact the provider.  The original
                # latency remains in the private cache record, not this replay.
                latency_seconds=0.0,
                queue_wait_seconds=0.0,
                cache_hit=True,
                physical_attempts=0,
            )
        headers = {
            "Authorization": "Bearer " + self.api_key,
            "Content-Type": "application/json",
        }

        for attempt in range(1, self.max_transport_attempts + 1):
            wait_started = time.monotonic()
            self._journal("request_waiting", semantic_id, request_hash, attempt)
            try:
                with self.limiter.acquire(timeout=self.timeout_seconds) as lease:
                    queue_wait = time.monotonic() - wait_started
                    started = time.monotonic()
                    status, response_body = self.http_transport(
                        self.endpoint, headers, body, self.timeout_seconds
                    )
                    latency = time.monotonic() - started
                    lease_log = lease.to_log_dict()
            except Exception as error:
                self._journal("transport_failure", semantic_id, request_hash, attempt)
                if attempt < self.max_transport_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise BFCLTransportError(type(error).__name__) from error

            if status == 429 or 500 <= status < 600:
                self._journal(
                    "retryable_http",
                    semantic_id,
                    request_hash,
                    attempt,
                    http_status=status,
                )
                if attempt < self.max_transport_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise BFCLTransportError(f"retryable_http_{status}")
            if not 200 <= status < 300:
                raise BFCLProtocolError(f"nonretryable_http_{status}")
            try:
                parsed = json.loads(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise BFCLProtocolError("Ollama response is not JSON") from error
            if not isinstance(parsed, Mapping):
                raise BFCLProtocolError("Ollama response root is not an object")
            normalized = normalize_ollama_response(
                parsed,
                expected_model=self.model,
                call_namespace=semantic_id,
                latency_seconds=latency,
                queue_wait_seconds=queue_wait,
                cache_hit=False,
                physical_attempts=attempt,
            )
            write_json_atomic(
                cache_path,
                {
                    "schema_version": "round9-bfcl-semantic-cache-v1",
                    "semantic_id_sha256": semantic_hash,
                    "request_sha256": request_hash,
                    "model_sha256": model_hash,
                    "provider_latency_seconds": latency,
                    "response": parsed,
                },
            )
            os.chmod(cache_path, 0o600)
            self._journal(
                "response_received",
                semantic_id,
                request_hash,
                attempt,
                response_sha256=sha256_bytes(response_body),
                http_status=status,
                queue_wait_ms=queue_wait * 1000,
                provider_latency_ms=latency * 1000,
                lease=lease_log,
            )
            return normalized
        raise AssertionError("unreachable")

    def _journal(
        self,
        event: str,
        semantic_id: str,
        request_hash: str,
        attempt: int,
        **generated_metadata: Any,
    ) -> None:
        # No request/response/model content is permitted in this file.  The model
        # identifier and semantic ID are hashed because BFCL cases can contain
        # upstream mock credentials.
        record = {
            "event": event,
            "semantic_id_sha256": sha256_text(semantic_id),
            "request_sha256": request_hash,
            "model_sha256": sha256_text(self.model),
            "attempt": attempt,
            **generated_metadata,
        }
        payload = (canonical_json(record) + "\n").encode("utf-8")
        self.journal_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(
            self.journal_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600
        )
        try:
            os.fchmod(descriptor, 0o600)
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, payload)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def synthetic_stop_response() -> NormalizedChatResponse:
    return NormalizedChatResponse(
        "", (), 0, 0, 0.0, synthetic=True, physical_attempts=0
    )


def verify_official_checkout(bfcl_root: Path) -> str:
    bfcl_root = bfcl_root.resolve()
    if not (bfcl_root / "bfcl_eval" / "model_handler" / "base_handler.py").is_file():
        raise BFCLProtocolError("BFCL official package tree is incomplete")
    try:
        commit = subprocess.run(
            ["git", "-C", str(bfcl_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise BFCLProtocolError("cannot verify BFCL checkout revision") from error
    if commit != BFCL_REPOSITORY_COMMIT:
        raise BFCLProtocolError(
            f"BFCL checkout is not pinned to required commit {BFCL_REPOSITORY_COMMIT}"
        )
    return commit


def validate_frozen_sample(
    sample_test_path: Path, sample_manifest_path: Path
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = read_jsonl(sample_test_path)
    manifest = read_json(sample_manifest_path)
    if not isinstance(manifest, dict):
        raise BFCLProtocolError("BFCL sample manifest must be an object")
    if manifest.get("evaluation_scope") != BFCL_SCOPE:
        raise BFCLProtocolError(
            "BFCL sample is not the frozen partial:150_of_200 scope"
        )
    if len(rows) != BFCL_SAMPLE_SIZE or manifest.get("sample_size") != BFCL_SAMPLE_SIZE:
        raise BFCLProtocolError("BFCL frozen sample must contain exactly 150 cases")
    if sha256_file(sample_test_path) != manifest.get("test_payload_sha256"):
        raise BFCLProtocolError("BFCL frozen sample checksum mismatch")
    if ordered_ids_sha256(rows, id_key="id") != manifest.get("ordered_case_ids_sha256"):
        raise BFCLProtocolError("BFCL frozen sample ordered-ID checksum mismatch")
    ids = [row.get("id") for row in rows]
    if len(set(ids)) != BFCL_SAMPLE_SIZE:
        raise BFCLProtocolError("BFCL frozen sample IDs are not unique")
    return rows, manifest


def _install_official_import_path(bfcl_root: Path) -> None:
    root = str(bfcl_root.resolve())
    existing = sys.modules.get("bfcl_eval")
    if existing is not None:
        existing_path = str(Path(existing.__file__).resolve())
        if not existing_path.startswith(root + os.sep):
            raise BFCLProtocolError("a different BFCL package is already imported")
    if root not in sys.path:
        sys.path.insert(0, root)


def build_official_handler_class(
    *, transport: BFCLCloudTransport, policy: ArmPolicy
) -> type:
    """Create an OpenAI-style BFCL handler backed by native Ollama messages."""
    from bfcl_eval.constants.default_prompts import MAXIMUM_STEP_LIMIT
    from bfcl_eval.constants.enums import ModelStyle
    from bfcl_eval.model_handler.api_inference.openai_completion import (
        OpenAICompletionsHandler,
    )
    from bfcl_eval.model_handler.base_handler import BaseHandler

    if MAXIMUM_STEP_LIMIT != BFCL_OFFICIAL_EXECUTION_STEP_LIMIT:
        raise BFCLProtocolError(
            "pinned BFCL execution-step horizon differs from the audited protocol"
        )

    class OllamaOfficialBFCLHandler(OpenAICompletionsHandler):
        def __init__(
            self, model_name, temperature, registry_name, is_fc_model, **kwargs
        ):
            # Do not initialize the OpenAI SDK.  Evaluation only needs the official
            # decoder; generation routes every physical request through our limiter.
            BaseHandler.__init__(
                self, model_name, temperature, registry_name, is_fc_model, **kwargs
            )
            self.model_style = ModelStyle.OPENAI_COMPLETIONS
            self._round9_thread = threading.local()

        def inference(self, test_entry, include_input_log, exclude_state_log):
            """Delegate to BFCL while retaining numeric per-case telemetry."""
            self._round9_thread.operations = []
            try:
                result, metadata = super().inference(
                    test_entry,
                    include_input_log=include_input_log,
                    exclude_state_log=exclude_state_log,
                )
                metadata = dict(metadata)
                metadata["round9_operational"] = copy.deepcopy(
                    self._round9_thread.operations
                )
                return result, metadata
            except BFCLTransportError:
                # Infrastructure failures remain resumable and are never
                # converted into terminal benchmark records.
                raise
            except Exception as error:
                raise BFCLTrackedCaseError(
                    type(error).__name__, self._round9_thread.operations
                ) from error
            finally:
                self._round9_thread.operations = []

        def _pre_query_processing_FC(self, inference_data, test_entry):
            return {
                "message": [],
                "_case_id": test_entry["id"],
                "_turn_index": 0,
                "_calls_this_turn": 0,
                "_synthetic_stops": 0,
            }

        def _add_next_turn_user_message_FC(self, inference_data, user_message):
            inference_data["_turn_index"] += 1
            inference_data["_calls_this_turn"] = 0
            inference_data["message"].extend(user_message)
            return inference_data

        def _query_FC(self, inference_data):
            inference_data["inference_input_log"] = "disabled_by_round9_privacy_policy"
            decision = policy.decision(inference_data["_calls_this_turn"])
            if decision == "synthetic_stop":
                inference_data["_synthetic_stops"] += 1
                return synthetic_stop_response(), 0.0
            semantic_id = (
                f"{policy.arm.value}:{transport.model}:{inference_data['_case_id']}:"
                f"turn{inference_data['_turn_index']}:call{inference_data['_calls_this_turn']}"
            )
            response = transport.chat(
                semantic_id=semantic_id,
                messages=inference_data["message"],
                tools=inference_data["tools"],
            )
            inference_data["_calls_this_turn"] += 1
            operations = getattr(self._round9_thread, "operations", None)
            if not isinstance(operations, list):
                raise BFCLProtocolError("BFCL call telemetry was not initialized")
            operations.append(
                {
                    "input_tokens": response.prompt_tokens,
                    "completion_tokens": response.completion_tokens,
                    "provider_latency_ms": response.latency_seconds * 1000,
                    "queue_wait_ms": response.queue_wait_seconds * 1000,
                    "cache_hit": response.cache_hit,
                    "physical_attempts": response.physical_attempts,
                }
            )
            return response, response.latency_seconds

        def _parse_query_response_FC(self, response):
            calls = [
                {
                    call.name: json.dumps(
                        call.arguments,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                }
                for call in response.tool_calls
            ]
            assistant: dict[str, Any] = {
                "role": "assistant",
                "content": response.content,
            }
            if response.tool_calls:
                assistant["tool_calls"] = [
                    {
                        "function": {"name": call.name, "arguments": call.arguments},
                    }
                    for call in response.tool_calls
                ]
            return {
                "model_responses": calls if calls else response.content,
                "model_responses_message_for_chat_history": assistant,
                "tool_call_ids": [call.call_id for call in response.tool_calls],
                "tool_names": [call.name for call in response.tool_calls],
                "input_token": response.prompt_tokens,
                "output_token": response.completion_tokens,
                "round9_synthetic_stop": response.synthetic,
            }

        def _add_execution_results_FC(
            self,
            inference_data,
            execution_results,
            model_response_data,
        ):
            """Append native Ollama tool-result messages.

            BFCL's OpenAI handler uses ``tool_call_id`` here.  The native
            Ollama ``/api/chat`` contract instead identifies the result with
            ``tool_name``.  Keeping this override next to the response parser
            prevents a successful first tool call from losing its observation
            on the next physical request.
            """
            tool_names = model_response_data.get("tool_names")
            if not isinstance(tool_names, list) or len(tool_names) != len(
                execution_results
            ):
                raise BFCLProtocolError(
                    "BFCL execution results do not align with native tool names"
                )
            for execution_result, tool_name in zip(execution_results, tool_names):
                if not isinstance(tool_name, str) or not tool_name:
                    raise BFCLProtocolError("BFCL native tool name is invalid")
                inference_data["message"].append(
                    {
                        "role": "tool",
                        "tool_name": tool_name,
                        "content": execution_result,
                    }
                )
            return inference_data

    OllamaOfficialBFCLHandler.__name__ = "OllamaOfficialBFCLHandler"
    return OllamaOfficialBFCLHandler


def load_hydrated_sample(
    bfcl_root: Path, sample_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    _install_official_import_path(bfcl_root)
    from bfcl_eval.utils import load_dataset_entry

    official_rows = load_dataset_entry(
        BFCL_CATEGORY,
        include_prereq=False,
        include_language_specific_hint=True,
    )
    by_id = {row["id"]: row for row in official_rows}
    if len(by_id) != 200:
        raise BFCLProtocolError("official BFCL missing-parameter population is not 200")
    hydrated: list[dict[str, Any]] = []
    for frozen in sample_rows:
        case_id = frozen["id"]
        if case_id not in by_id:
            raise BFCLProtocolError("frozen BFCL case is absent from official checkout")
        official = by_id[case_id]
        # Compare the fields retained in the frozen raw sample.  The official
        # loader adds compiled function schemas, which are deliberately absent
        # from the source JSONL and therefore excluded from this equality check.
        for key, value in frozen.items():
            if official.get(key) != value:
                raise BFCLProtocolError("frozen BFCL case differs from official source")
        if not isinstance(official.get("function"), list) or not official["function"]:
            raise BFCLProtocolError(
                "official BFCL loader did not hydrate function schemas"
            )
        hydrated.append(copy.deepcopy(official))
    return hydrated


def official_result_path(result_root: Path, registry_name: str) -> Path:
    return (
        result_root
        / registry_name
        / "multi_turn"
        / "BFCL_v4_multi_turn_miss_param_result.json"
    )


def make_bfcl_handler_identity(
    *, registry_name: str, arm: BFCLArm, provider_model: str
) -> str:
    """Return a stable Python-safe identity for BFCL's executable mock state.

    BFCL interpolates ``BaseHandler.model_name`` into a Python expression when
    dispatching a mock tool.  Provider model IDs commonly contain ``:`` (for
    example ``deepseek-v4-flash:0731``), which BFCL does not sanitize and which
    makes every dispatched call invalid Python.  The Cloud transport retains
    the exact provider ID; only BFCL's private mock-state namespace uses this
    non-reversible, identifier-safe value.
    """
    if not registry_name or not provider_model:
        raise ValueError("registry_name and provider_model are required")
    binding = canonical_json(
        {
            "registry_name": registry_name,
            "arm": arm.value,
            "provider_model_sha256": sha256_text(provider_model),
        }
    )
    identity = "farm_r9_" + sha256_text(binding)[:24]
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", identity):
        raise AssertionError("generated BFCL handler identity is not Python-safe")
    return identity


def ensure_run_binding(
    path: Path,
    *,
    registry_name: str,
    arm: BFCLArm,
    model: str,
    sample_sha256: str,
    official_commit: str,
    temperature: float,
    seed: int,
    thinking_mode: str | None,
    max_physical_calls_per_turn: int,
    workers: int,
) -> dict[str, Any]:
    """Create or verify the immutable identity attached to resumable results."""
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,100}", registry_name):
        raise ValueError(
            "registry_name may not contain slashes, underscores, spaces, or secrets"
        )
    binding = {
        "schema_version": "farm-round9-bfcl-run-binding-v2",
        "benchmark": BFCL_BENCHMARK,
        "evaluation_scope": BFCL_SCOPE,
        "registry_name": registry_name,
        "arm": arm.value,
        "model_sha256": sha256_text(model),
        "sample_sha256": sample_sha256,
        "official_commit": official_commit,
        "temperature": float(temperature),
        "seed": int(seed),
        "thinking_mode": thinking_mode,
        "max_physical_calls_per_turn": int(max_physical_calls_per_turn),
        "worker_policy": BFCL_WORKER_POLICY,
        "workers": int(workers),
    }
    if path.exists():
        if read_json(path) != binding:
            raise BFCLProtocolError(
                "existing BFCL results have a different immutable run binding"
            )
    else:
        write_json_atomic(path, binding)
        os.chmod(path, 0o600)
    return binding


def _safe_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    return redact_public_trace(
        dict(metadata),
        source_classification=DataClassification.PUBLIC,
        source=DataSource.BFCL_V4,
    )


def make_result_record(
    case_id: str,
    result: Any,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Keep evaluator input exact while redacting diagnostic metadata."""
    return {"id": case_id, "result": result, **_safe_metadata(metadata)}


def make_failure_record(case_id: str, error: BaseException) -> dict[str, Any]:
    # Do not stringify an arbitrary exception; provider errors can echo payloads.
    failure_type = type(error).__name__
    operations: list[dict[str, Any]] = []
    if isinstance(error, BFCLTrackedCaseError):
        failure_type = error.failure_type
        operations = [copy.deepcopy(dict(item)) for item in error.operations]
    return {
        "id": case_id,
        "result": f"Round9 BFCL protocol failure: {failure_type}",
        "protocol_failure": True,
        "failure_type": failure_type,
        "input_token_count": [],
        "output_token_count": [],
        "latency": [],
        "inference_log": [],
        "round9_operational": operations,
    }


def aggregate_operational_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int | float | bool | None]:
    """Aggregate numeric telemetry without exposing BFCL cases or tool traces."""
    n = len(records)
    if n <= 0:
        raise BFCLProtocolError("BFCL operational aggregation requires records")

    calls_by_case: list[int] = []
    calls: list[dict[str, Any]] = []
    required = {
        "input_tokens",
        "completion_tokens",
        "provider_latency_ms",
        "queue_wait_ms",
        "cache_hit",
        "physical_attempts",
    }
    for record in records:
        operations = record.get("round9_operational")
        if not isinstance(operations, list):
            raise BFCLProtocolError("BFCL result is missing operational telemetry")
        calls_by_case.append(len(operations))
        for operation in operations:
            if not isinstance(operation, Mapping) or set(operation) != required:
                raise BFCLProtocolError("BFCL operation telemetry has an invalid shape")
            input_tokens = operation["input_tokens"]
            completion_tokens = operation["completion_tokens"]
            physical_attempts = operation["physical_attempts"]
            cache_hit = operation["cache_hit"]
            provider_latency = operation["provider_latency_ms"]
            queue_wait = operation["queue_wait_ms"]
            if any(
                isinstance(value, bool) or not isinstance(value, int) or value < 0
                for value in (input_tokens, completion_tokens, physical_attempts)
            ):
                raise BFCLProtocolError(
                    "BFCL operation counts must be non-negative integers"
                )
            if not isinstance(cache_hit, bool):
                raise BFCLProtocolError("BFCL cache-hit telemetry must be boolean")
            if any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0
                for value in (provider_latency, queue_wait)
            ):
                raise BFCLProtocolError(
                    "BFCL operation timings must be finite and non-negative"
                )
            if cache_hit and (
                physical_attempts != 0
                or float(provider_latency) != 0.0
                or float(queue_wait) != 0.0
            ):
                raise BFCLProtocolError(
                    "BFCL cache replay cannot report a physical request"
                )
            if not cache_hit and physical_attempts < 1:
                raise BFCLProtocolError(
                    "BFCL live response must report a physical attempt"
                )
            calls.append(copy.deepcopy(dict(operation)))

    calls_by_case.sort()
    latencies = sorted(
        float(call["provider_latency_ms"])
        for call in calls
        if not bool(call["cache_hit"])
    )
    input_tokens_total = sum(int(call["input_tokens"]) for call in calls)
    completion_tokens_total = sum(int(call["completion_tokens"]) for call in calls)
    metrics = {
        "semantic_calls_total": len(calls),
        "semantic_calls_mean_per_case": len(calls) / n,
        "semantic_calls_p50": float(calls_by_case[len(calls_by_case) // 2]),
        "semantic_calls_p95": float(calls_by_case[math.ceil(0.95 * n) - 1]),
        "physical_attempts_total": sum(
            int(call["physical_attempts"]) for call in calls
        ),
        "cache_hits_total": sum(bool(call["cache_hit"]) for call in calls),
        "input_tokens_total": input_tokens_total,
        "completion_tokens_total": completion_tokens_total,
        "tokens_total": input_tokens_total + completion_tokens_total,
        "provider_latency_ms_total": sum(latencies),
        "provider_latency_ms_mean_per_call": (
            sum(latencies) / len(latencies) if latencies else 0.0
        ),
        "provider_latency_ms_p50": (
            latencies[len(latencies) // 2] if latencies else 0.0
        ),
        "provider_latency_ms_p95": (
            latencies[math.ceil(0.95 * len(latencies)) - 1] if latencies else 0.0
        ),
        "queue_wait_ms_total": sum(float(call["queue_wait_ms"]) for call in calls),
        "cost_available": False,
        "cost_usd_total": None,
        "cost_usd_mean_per_case": None,
    }
    # Reuse the fail-closed public aggregate contract as a final consistency
    # and disclosure check; only the numeric subtree is returned.
    public = export_public_aggregate(
        {
            "benchmark_label": BFCL_BENCHMARK,
            "n": n,
            "operational_metrics": metrics,
        },
        source_classification=DataClassification.PUBLIC,
    )
    return public["operational_metrics"]


def build_public_provider_gate_summary(
    response: NormalizedChatResponse,
    *,
    model: str,
    journal_sha256: str,
) -> dict[str, Any]:
    """Validate one synthetic native tool response and emit no response text."""
    if not re.fullmatch(r"[0-9a-f]{64}", journal_sha256):
        raise ValueError("journal_sha256 must be a lowercase SHA-256 digest")
    if len(response.tool_calls) != 1:
        raise BFCLProtocolError("provider gate expected exactly one native tool call")
    call = response.tool_calls[0]
    if call.name != "lookup_public_code":
        raise BFCLProtocolError("provider gate returned the wrong native tool name")
    if set(call.arguments) != {"item"} or not isinstance(call.arguments["item"], str):
        raise BFCLProtocolError("provider gate returned invalid native tool arguments")
    operation = {
        "input_tokens": response.prompt_tokens,
        "completion_tokens": response.completion_tokens,
        "provider_latency_ms": response.latency_seconds * 1000,
        "queue_wait_ms": response.queue_wait_seconds * 1000,
        "cache_hit": response.cache_hit,
        "physical_attempts": response.physical_attempts,
    }
    return {
        "schema_version": "farm-round9-bfcl-provider-gate-v1",
        "status": "pass",
        "public_synthetic_only": True,
        "model_sha256": sha256_text(model),
        "native_tool_call_count": 1,
        "tool_name_matches": True,
        "argument_object_valid": True,
        "operational_metrics": aggregate_operational_metrics(
            [{"round9_operational": [operation]}]
        ),
        "journal_sha256": journal_sha256,
    }


def _load_resume_records(
    result_path: Path, selected_ids: Sequence[str]
) -> dict[str, dict[str, Any]]:
    if not result_path.exists():
        return {}
    rows = read_jsonl(result_path)
    ids = [row.get("id") for row in rows]
    if len(ids) != len(set(ids)):
        raise BFCLProtocolError("existing BFCL result contains duplicate IDs")
    selected = set(selected_ids)
    if not set(ids).issubset(selected):
        raise BFCLProtocolError(
            "existing BFCL result contains an ID outside frozen sample"
        )
    return {str(row["id"]): row for row in rows}


def generate_official_results(
    *,
    handler: Any,
    cases: Sequence[Mapping[str, Any]],
    result_path: Path,
    workers: int,
) -> dict[str, Any]:
    if isinstance(workers, bool) or not 1 <= workers <= 3:
        raise ValueError("BFCL workers must be in 1..3")
    selected_ids = [str(case["id"]) for case in cases]
    completed = _load_resume_records(result_path, selected_ids)
    pending = [copy.deepcopy(case) for case in cases if case["id"] not in completed]

    def infer(case: dict[str, Any]) -> dict[str, Any]:
        try:
            result, metadata = handler.inference(
                case,
                include_input_log=False,
                exclude_state_log=True,
            )
            return make_result_record(case["id"], result, metadata)
        except BFCLTransportError:
            # A provider/network outage is resumable infrastructure state, not
            # a terminal model error to score as benchmark failure.
            raise
        except Exception as error:
            return make_failure_record(case["id"], error)

    result_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    if pending:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(infer, case): case["id"] for case in pending}
            while futures:
                done, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in done:
                    case_id = futures.pop(future)
                    record = future.result()
                    if record.get("id") != case_id:
                        raise BFCLProtocolError(
                            "BFCL worker returned the wrong case ID"
                        )
                    completed[case_id] = record
                    ordered = [
                        completed[value] for value in selected_ids if value in completed
                    ]
                    write_jsonl_atomic(result_path, ordered)
                    os.chmod(result_path, 0o600)

    if set(completed) != set(selected_ids):
        raise BFCLProtocolError(
            "BFCL generation ended before all frozen cases completed"
        )
    failure_count = sum(bool(row.get("protocol_failure")) for row in completed.values())
    ordered_records = [completed[value] for value in selected_ids]
    return {
        "evaluation_scope": BFCL_SCOPE,
        "n": len(completed),
        "protocol_failures": failure_count,
        "result_file_sha256": sha256_file(result_path),
        "resumable_completed_cases": len(completed),
        "operational_metrics": aggregate_operational_metrics(ordered_records),
    }


def register_official_model(
    *,
    bfcl_root: Path,
    registry_name: str,
    actual_model: str,
    handler_class: type,
) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.:-]{0,100}", registry_name):
        raise ValueError(
            "registry_name may not contain slashes, underscores, spaces, or secrets"
        )
    _install_official_import_path(bfcl_root)
    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig

    MODEL_CONFIG_MAPPING[registry_name] = ModelConfig(
        model_name=actual_model,
        display_name=registry_name,
        url="https://ollama.com/",
        org="Ollama Cloud",
        license="provider-hosted",
        model_handler=handler_class,
        input_price=None,
        output_price=None,
        is_fc_model=True,
    )


def run_official_partial_evaluator(
    *,
    result_root: Path,
    score_root: Path,
    registry_name: str,
) -> dict[str, Any]:
    from bfcl_eval.eval_checker.eval_runner import runner as official_eval_runner

    result_file = official_result_path(result_root, registry_name)
    results = read_jsonl(result_file)
    if len(results) != BFCL_SAMPLE_SIZE:
        raise BFCLProtocolError(
            "official partial evaluation requires all frozen 150 results"
        )
    score_root.mkdir(parents=True, exist_ok=True, mode=0o700)
    official_eval_runner(
        [registry_name],
        [BFCL_CATEGORY],
        result_root,
        score_root,
        allow_missing=True,
    )
    score_file = (
        score_root
        / registry_name
        / "multi_turn"
        / "BFCL_v4_multi_turn_miss_param_score.json"
    )
    score_rows = read_jsonl(score_file)
    if not score_rows or score_rows[0].get("total_count") != BFCL_SAMPLE_SIZE:
        raise BFCLProtocolError(
            "official BFCL evaluator did not score exactly 150 cases"
        )
    for directory, _, filenames in os.walk(score_root / registry_name):
        os.chmod(directory, 0o700)
        for filename in filenames:
            os.chmod(Path(directory) / filename, 0o600)
    header = score_rows[0]
    return {
        "evaluation_scope": BFCL_SCOPE,
        "leaderboard_comparable": False,
        "official_evaluator": True,
        "n": int(header["total_count"]),
        "correct": int(header["correct_count"]),
        "accuracy": float(header["accuracy"]),
        "score_file_sha256": sha256_file(score_file),
        "operational_metrics": aggregate_operational_metrics(results),
    }


def write_protocol_summary(
    path: Path,
    *,
    generation: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    arm: BFCLArm,
    model: str,
    registry_name: str,
    official_commit: str,
) -> None:
    summary = {
        "benchmark": BFCL_BENCHMARK,
        "evaluation_scope": BFCL_SCOPE,
        "leaderboard_comparable": False,
        "official_generation_harness": True,
        "official_state_based_evaluator": True,
        "arm": arm.value,
        "model_sha256": sha256_text(model),
        "registry_name": registry_name,
        "official_commit": official_commit,
        "generation": dict(generation),
        "evaluation": dict(evaluation),
    }
    generation_operational = generation.get("operational_metrics")
    evaluation_operational = evaluation.get("operational_metrics")
    if generation_operational is not None and evaluation_operational is not None:
        if generation_operational != evaluation_operational:
            raise BFCLProtocolError("generation and evaluation telemetry disagree")
    operational = generation_operational or evaluation_operational
    if operational is not None:
        summary["operational_metrics"] = copy.deepcopy(operational)
    if path.exists():
        if read_json(path) != summary:
            raise BFCLProtocolError(
                "refusing to overwrite a different BFCL protocol summary"
            )
    else:
        write_json_atomic(path, summary)
        os.chmod(path, 0o600)


__all__ = [
    "ArmPolicy",
    "BFCLArm",
    "BFCLCloudTransport",
    "BFCLProtocolError",
    "BFCLTransportError",
    "BFCL_OFFICIAL_EXECUTION_STEP_LIMIT",
    "BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN",
    "BFCL_SCOPE",
    "aggregate_operational_metrics",
    "build_public_provider_gate_summary",
    "build_official_handler_class",
    "ensure_run_binding",
    "generate_official_results",
    "load_hydrated_sample",
    "make_bfcl_handler_identity",
    "make_failure_record",
    "make_result_record",
    "normalize_ollama_response",
    "official_result_path",
    "register_official_model",
    "run_official_partial_evaluator",
    "validate_frozen_sample",
    "verify_official_checkout",
    "write_protocol_summary",
]
