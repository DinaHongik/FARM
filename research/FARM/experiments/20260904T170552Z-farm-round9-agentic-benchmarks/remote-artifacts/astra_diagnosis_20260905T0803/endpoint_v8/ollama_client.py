"""Resumable Ollama chat transport with fail-closed privacy and global leases."""
from __future__ import annotations

import json
import os
import socket
import time
import urllib.error
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import urlsplit

from farm_r9.artifact_io import canonical_json, read_json, sha256_bytes, sha256_text, write_json_atomic
from farm_r9.cloud_limiter import OllamaCloudLimiter
from farm_r9.privacy import DataClassification, DataSource, redact_public_trace, require_cloud_route


class OllamaTransportError(RuntimeError):
    pass


class OllamaProtocolError(RuntimeError):
    pass


class OllamaJSONError(ValueError):
    pass


@dataclass(frozen=True)
class ChatResult:
    semantic_id: str
    model: str
    content: str
    tool_calls: tuple[dict[str, Any], ...]
    prompt_tokens: int
    completion_tokens: int
    provider_latency_ms: float
    queue_wait_ms: float
    physical_attempts: int
    request_sha256: str
    response_sha256: str
    cache_hit: bool
    raw_response: dict[str, Any]


Transport = Callable[[str, Mapping[str, str], bytes, float], tuple[int, bytes]]


def _default_transport(endpoint: str, headers: Mapping[str, str], body: bytes, timeout: float) -> tuple[int, bytes]:
    request = urllib.request.Request(endpoint, data=body, headers=dict(headers), method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return int(response.status), response.read()
    except urllib.error.HTTPError as error:
        return int(error.code), error.read()
    except (urllib.error.URLError, TimeoutError, socket.timeout, OSError) as error:
        raise OllamaTransportError(type(error).__name__) from error


def _endpoint(host: str, *, cloud: bool) -> str:
    parsed = urlsplit(host)
    if cloud:
        if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Cloud Ollama host must be an authenticated-key-free HTTPS origin")
    else:
        if parsed.scheme != "http" or parsed.hostname != "127.0.0.1" or parsed.port is None:
            raise ValueError("local Ollama host must be http://127.0.0.1:PORT")
    if parsed.query or parsed.fragment:
        raise ValueError("Ollama host cannot contain query or fragment")
    base = host.rstrip("/")
    return base + ("/chat" if base.endswith("/api") else "/api/chat")


class OllamaChatClient:
    def __init__(
        self,
        *,
        host: str,
        api_key: str | None,
        model: str,
        cache_directory: Path,
        journal_path: Path,
        cloud: bool,
        limiter: OllamaCloudLimiter | None,
        timeout_seconds: float = 240.0,
        max_transport_attempts: int = 2,
        transport: Transport = _default_transport,
    ) -> None:
        if not model:
            raise ValueError("model is required")
        if cloud and (not api_key or limiter is None):
            raise ValueError("Cloud Ollama requires an API key and shared limiter")
        if not cloud and api_key is not None:
            raise ValueError("local Ollama must not receive an API key")
        if max_transport_attempts not in {1, 2, 3}:
            raise ValueError("transport attempts must be 1..3")
        self.endpoint = _endpoint(host, cloud=cloud)
        self.api_key, self.model, self.cloud, self.limiter = api_key, model, cloud, limiter
        self.cache_directory, self.journal_path = cache_directory, journal_path
        self.timeout_seconds, self.max_transport_attempts, self.transport = timeout_seconds, max_transport_attempts, transport
        self.cache_directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(self.cache_directory, 0o700)

    def chat(
        self,
        *,
        semantic_id: str,
        benchmark_label: str,
        data_classification: DataClassification,
        data_source: DataSource,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] = (),
        temperature: float = 0.0,
        seed: int = 42,
        think: str | None = "low",
    ) -> ChatResult:
        if self.cloud:
            require_cloud_route(
                data_classification,
                benchmark_label=benchmark_label,
                source=data_source,
            )
        payload: dict[str, Any] = {
            "model": self.model, "messages": [dict(message) for message in messages],
            "stream": False, "options": {"temperature": temperature, "seed": seed},
        }
        if tools:
            payload["tools"] = [dict(tool) for tool in tools]
        if think is not None:
            payload["think"] = think
        body = canonical_json(payload).encode("utf-8")
        request_hash = sha256_bytes(body)
        cache_path = self.cache_directory / f"{sha256_text(semantic_id)}.json"
        if cache_path.exists():
            cached = read_json(cache_path)
            if cached.get("semantic_id") != semantic_id or cached.get("request_sha256") != request_hash or cached.get("model") != self.model:
                raise OllamaProtocolError("semantic cache binding mismatch")
            return self._result(cached, cache_hit=True)

        headers = {"Content-Type": "application/json"}
        if self.api_key is not None:
            headers["Authorization"] = "Bearer " + self.api_key
        attempts, last_error = 0, None
        for attempt in range(1, self.max_transport_attempts + 1):
            attempts = attempt
            queue_started = time.monotonic()
            self._journal({
                "event": "request_waiting", "semantic_id_sha256": sha256_text(semantic_id),
                "request_sha256": request_hash, "model": self.model, "attempt": attempt,
            })
            try:
                if self.cloud:
                    assert self.limiter is not None
                    with self.limiter.acquire(timeout=self.timeout_seconds) as lease:
                        queue_wait = time.monotonic() - queue_started
                        started = time.monotonic()
                        status, response_body = self.transport(self.endpoint, headers, body, self.timeout_seconds)
                        provider_latency = time.monotonic() - started
                        lease_log = lease.to_log_dict()
                else:
                    queue_wait = 0.0
                    started = time.monotonic()
                    status, response_body = self.transport(self.endpoint, headers, body, self.timeout_seconds)
                    provider_latency = time.monotonic() - started
                    lease_log = None
            except Exception as error:
                last_error = OllamaTransportError(type(error).__name__)
                self._journal({
                    "event": "transport_failure", "semantic_id_sha256": sha256_text(semantic_id),
                    "request_sha256": request_hash, "model": self.model, "attempt": attempt,
                })
                if attempt < self.max_transport_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise last_error from error
            if status == 429 or 500 <= status < 600:
                last_error = OllamaTransportError(f"retryable_http_{status}")
                self._journal({
                    "event": "retryable_http", "semantic_id_sha256": sha256_text(semantic_id),
                    "request_sha256": request_hash, "model": self.model, "attempt": attempt, "http_status": status,
                })
                if attempt < self.max_transport_attempts:
                    time.sleep(min(2 ** (attempt - 1), 4))
                    continue
                raise last_error
            if not 200 <= status < 300:
                raise OllamaTransportError(f"http_{status}")
            try:
                response = json.loads(response_body)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise OllamaProtocolError("response is not JSON") from error
            if not isinstance(response, dict) or response.get("model") != self.model:
                raise OllamaProtocolError("response model mismatch")
            message = response.get("message")
            if not isinstance(message, dict):
                raise OllamaProtocolError("response message missing")
            content = message.get("content") or ""
            tool_calls = message.get("tool_calls") or []
            if not isinstance(content, str) or not isinstance(tool_calls, list):
                raise OllamaProtocolError("response message fields invalid")
            raw_for_cache = response
            if data_classification is DataClassification.PUBLIC:
                raw_for_cache = redact_public_trace(
                    response,
                    source_classification=data_classification,
                    source=data_source,
                )
                content = redact_public_trace(
                    content,
                    source_classification=data_classification,
                    source=data_source,
                )
                tool_calls = redact_public_trace(
                    tool_calls,
                    source_classification=data_classification,
                    source=data_source,
                )
            record = {
                "semantic_id": semantic_id, "benchmark_label": benchmark_label,
                "data_classification": data_classification.value,
                "data_source": data_source.value,
                "model": self.model,
                "content": content, "tool_calls": tool_calls,
                "prompt_tokens": int(response.get("prompt_eval_count") or 0),
                "completion_tokens": int(response.get("eval_count") or 0),
                "provider_latency_ms": provider_latency * 1000,
                "queue_wait_ms": queue_wait * 1000, "physical_attempts": attempts,
                "request_sha256": request_hash, "response_sha256": sha256_bytes(response_body),
                "raw_response": raw_for_cache,
            }
            write_json_atomic(cache_path, record)
            os.chmod(cache_path, 0o600)
            self._journal({
                "event": "response_cached", "semantic_id_sha256": sha256_text(semantic_id),
                "request_sha256": request_hash, "response_sha256": record["response_sha256"],
                "model": self.model, "attempt": attempt, "http_status": status,
                "queue_wait_ms": record["queue_wait_ms"], "provider_latency_ms": record["provider_latency_ms"],
                "lease": lease_log,
            })
            return self._result(record, cache_hit=False)
        assert last_error is not None
        raise last_error

    def _result(self, record: Mapping[str, Any], *, cache_hit: bool) -> ChatResult:
        return ChatResult(
            semantic_id=str(record["semantic_id"]), model=str(record["model"]), content=str(record["content"]),
            tool_calls=tuple(record.get("tool_calls") or []), prompt_tokens=int(record.get("prompt_tokens") or 0),
            completion_tokens=int(record.get("completion_tokens") or 0), provider_latency_ms=float(record.get("provider_latency_ms") or 0),
            queue_wait_ms=float(record.get("queue_wait_ms") or 0), physical_attempts=int(record.get("physical_attempts") or 0),
            request_sha256=str(record["request_sha256"]), response_sha256=str(record["response_sha256"]),
            cache_hit=cache_hit, raw_response=dict(record.get("raw_response") or {}),
        )

    def _journal(self, record: Mapping[str, Any]) -> None:
        import fcntl
        payload = canonical_json(dict(record)) + "\n"
        self.journal_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        descriptor = os.open(self.journal_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            os.write(descriptor, payload.encode("utf-8"))
            os.fsync(descriptor)
        finally:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)


def parse_json_object(content: str) -> tuple[dict[str, Any], bool]:
    """Parse one object; the bool records strict whole-response JSON."""
    stripped = content.strip()
    try:
        value = json.loads(stripped)
        if not isinstance(value, dict):
            raise OllamaJSONError("model output must be one JSON object")
        return value, True
    except json.JSONDecodeError:
        pass
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            inner = "\n".join(lines[1:-1]).strip()
            try:
                value = json.loads(inner)
                if isinstance(value, dict):
                    return value, False
            except json.JSONDecodeError:
                pass
    decoder = json.JSONDecoder()
    for index, character in enumerate(stripped):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(stripped[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value, False
    raise OllamaJSONError("no valid JSON object in model output")
