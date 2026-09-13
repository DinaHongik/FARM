from __future__ import annotations

import json
import os
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from farm_r9.artifact_io import (
    ordered_ids_sha256,
    sha256_file,
    write_json_atomic,
    write_jsonl_atomic,
)
from farm_r9.bfcl_runner import (
    ArmPolicy,
    BFCLArm,
    BFCLCloudTransport,
    BFCL_OFFICIAL_EXECUTION_STEP_LIMIT,
    BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN,
    BFCLProtocolError,
    BFCLTransportError,
    NormalizedChatResponse,
    NormalizedToolCall,
    aggregate_operational_metrics,
    build_official_handler_class,
    build_public_provider_gate_summary,
    ensure_run_binding,
    generate_official_results,
    make_failure_record,
    make_bfcl_handler_identity,
    make_result_record,
    normalize_ollama_response,
    validate_frozen_sample,
    write_protocol_summary,
)


class _Lease:
    def to_log_dict(self):
        return {
            "lease_id": "generated",
            "slot": 0,
            "max_concurrent": 3,
            "pid": 1,
            "hostname": "test",
            "acquired_at_utc": "2026-09-05T00:00:00Z",
            "waited_seconds": 0.0,
        }


class _Limiter:
    def __init__(self):
        self.entries = 0

    @contextmanager
    def acquire(self, *, timeout):
        self.entries += 1
        yield _Lease()


def test_arm_policies_distinguish_single_step_from_agent_budget():
    baseline = ArmPolicy(BFCLArm.SINGLE_STEP, 4)
    assert baseline.decision(0) == "physical"
    assert baseline.decision(1) == "synthetic_stop"
    agent = ArmPolicy(BFCLArm.NATIVE_AGENT, 4)
    assert [agent.decision(index) for index in range(4)] == ["physical"] * 4
    with pytest.raises(BFCLProtocolError):
        agent.decision(4)


def test_official_horizon_policy_leaves_termination_to_pinned_bfcl_loop():
    assert BFCL_OFFICIAL_EXECUTION_STEP_LIMIT == 20
    assert BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN == 21
    policy = ArmPolicy(
        BFCLArm.NATIVE_AGENT,
        BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN,
    )
    assert [
        policy.decision(index)
        for index in range(BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN)
    ] == ["physical"] * BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN
    with pytest.raises(BFCLProtocolError):
        policy.decision(BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN)


def test_agent_policy_rejects_budget_beyond_official_horizon():
    with pytest.raises(ValueError, match="1\\.\\.21"):
        ArmPolicy(
            BFCLArm.NATIVE_AGENT,
            BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN + 1,
        )


def test_official_horizon_v3_manifest_keeps_paired_worker_count():
    manifest = json.loads(
        (ROOT / "manifests" / "bfcl_native_official_horizon_v3.json").read_text(
            encoding="utf-8"
        )
    )
    assert manifest["arm"] == BFCLArm.NATIVE_AGENT.value
    assert (
        manifest["max_physical_calls_per_turn"]
        == BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN
    )
    assert manifest["official_execution_step_limit"] == 20
    assert manifest["workers"] == 2
    assert manifest["expected_score"] is None
    assert manifest["expected_protocol_failures"] is None


def test_normalizer_preserves_live_arguments_for_official_evaluator():
    result = normalize_ollama_response(
        {
            "model": "m",
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "authenticate_twitter",
                            "arguments": {
                                "username": "demo",
                                "password": "mock-password",
                            },
                        }
                    }
                ],
            },
            "prompt_eval_count": 12,
            "eval_count": 5,
        },
        expected_model="m",
        call_namespace="case:turn0:call0",
        latency_seconds=0.25,
    )
    assert result.tool_calls[0].arguments["password"] == "mock-password"
    assert result.tool_calls[0].call_id.startswith("call_")
    assert result.prompt_tokens == 12


def test_transport_journal_is_hash_only_and_uses_limiter(tmp_path):
    secret = "mock-password-never-in-journal"
    limiter = _Limiter()
    http_calls = 0

    def fake_http(endpoint, headers, body, timeout):
        nonlocal http_calls
        http_calls += 1
        assert secret in body.decode()
        assert headers["Authorization"] == "Bearer provider-key"
        response = {
            "model": "test-model",
            "message": {
                "content": "",
                "tool_calls": [
                    {"function": {"name": "login", "arguments": {"password": secret}}}
                ],
            },
        }
        return 200, json.dumps(response).encode()

    journal = tmp_path / "journal.jsonl"
    cache = tmp_path / "cache"
    transport = BFCLCloudTransport(
        host="https://ollama.example",
        api_key="provider-key",
        model="test-model",
        limiter=limiter,
        journal_path=journal,
        cache_directory=cache,
        max_transport_attempts=1,
        http_transport=fake_http,
    )
    response = transport.chat(
        semantic_id="case-without-sensitive-content",
        messages=[{"role": "user", "content": f"password={secret}"}],
        tools=[],
    )
    assert response.tool_calls[0].arguments["password"] == secret
    assert response.cache_hit is False
    assert response.physical_attempts == 1
    assert limiter.entries == 1
    cached_response = transport.chat(
        semantic_id="case-without-sensitive-content",
        messages=[{"role": "user", "content": f"password={secret}"}],
        tools=[],
    )
    assert cached_response.tool_calls[0].arguments["password"] == secret
    assert cached_response.cache_hit is True
    assert cached_response.physical_attempts == 0
    assert cached_response.latency_seconds == 0.0
    assert http_calls == 1
    assert limiter.entries == 1
    cache_files = list(cache.glob("*.json"))
    assert len(cache_files) == 1
    assert stat_mode(cache_files[0]) == 0o600
    journal_text = journal.read_text()
    assert secret not in journal_text
    assert "provider-key" not in journal_text
    assert "case-without-sensitive-content" not in journal_text
    assert stat_mode(journal) == 0o600


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


def test_result_keeps_official_calls_but_redacts_diagnostics():
    official_result = [[{"login": '{"password":"mock"}'}]]
    record = make_result_record(
        "multi_turn_miss_param_0",
        official_result,
        {
            "inference_log": [{"password": "mock", "content": "password=mock"}],
            "input_token_count": [[10]],
        },
    )
    assert record["result"] == official_result
    assert record["inference_log"][0]["password"] == "[REDACTED]"
    assert "password=mock" not in record["inference_log"][0]["content"]


def test_failure_record_does_not_stringify_sensitive_exception():
    failure = make_failure_record(
        "multi_turn_miss_param_0", RuntimeError("password=mock-secret")
    )
    assert failure["protocol_failure"] is True
    assert "mock-secret" not in json.dumps(failure)


def test_frozen_sample_validation_enforces_scope_count_and_hash(tmp_path):
    rows = [{"id": f"multi_turn_miss_param_{index}"} for index in range(150)]
    sample = tmp_path / "sample.jsonl"
    manifest = tmp_path / "manifest.json"
    write_jsonl_atomic(sample, rows)
    write_json_atomic(
        manifest,
        {
            "evaluation_scope": "partial:150_of_200",
            "sample_size": 150,
            "test_payload_sha256": sha256_file(sample),
            "ordered_case_ids_sha256": ordered_ids_sha256(rows, id_key="id"),
        },
    )
    loaded, _ = validate_frozen_sample(sample, manifest)
    assert len(loaded) == 150

    bad = json.loads(manifest.read_text())
    bad["evaluation_scope"] = "full"
    write_json_atomic(manifest, bad)
    with pytest.raises(BFCLProtocolError):
        validate_frozen_sample(sample, manifest)


def test_run_binding_prevents_cross_arm_resume(tmp_path):
    path = tmp_path / "RUN_BINDING.json"
    common = dict(
        registry_name="round9-bfcl",
        model="m",
        sample_sha256="a" * 64,
        official_commit="b" * 40,
        temperature=0.0,
        seed=9052026,
        thinking_mode="low",
        max_physical_calls_per_turn=4,
        workers=1,
    )
    ensure_run_binding(path, arm=BFCLArm.NATIVE_AGENT, **common)
    with pytest.raises(BFCLProtocolError):
        ensure_run_binding(path, arm=BFCLArm.SINGLE_STEP, **common)


@pytest.mark.parametrize(
    ("changed_key", "changed_value"),
    [
        ("temperature", 0.2),
        ("seed", 42),
        ("thinking_mode", None),
        ("max_physical_calls_per_turn", 3),
        ("workers", 2),
    ],
)
def test_run_binding_prevents_mixed_protocol_resume(
    tmp_path, changed_key, changed_value
):
    path = tmp_path / "RUN_BINDING.json"
    frozen = {
        "registry_name": "round9-bfcl",
        "arm": BFCLArm.NATIVE_AGENT,
        "model": "deepseek-v4-flash:0731",
        "sample_sha256": "a" * 64,
        "official_commit": "b" * 40,
        "temperature": 0.0,
        "seed": 9052026,
        "thinking_mode": "low",
        "max_physical_calls_per_turn": 4,
        "workers": 1,
    }
    ensure_run_binding(path, **frozen)
    changed = {**frozen, changed_key: changed_value}
    with pytest.raises(BFCLProtocolError):
        ensure_run_binding(path, **changed)


def test_transport_failure_is_not_written_as_terminal_model_failure(tmp_path):
    class FailingHandler:
        def inference(self, case, **kwargs):
            raise BFCLTransportError("TimeoutError")

    result_path = tmp_path / "results.jsonl"
    with pytest.raises(BFCLTransportError):
        generate_official_results(
            handler=FailingHandler(),
            cases=[{"id": "case-1"}],
            result_path=result_path,
            workers=1,
        )
    assert not result_path.exists()


def test_operational_aggregate_has_consistent_totals_means_and_quantiles(tmp_path):
    live_one = {
        "input_tokens": 10,
        "completion_tokens": 2,
        "provider_latency_ms": 10.0,
        "queue_wait_ms": 3.0,
        "cache_hit": False,
        "physical_attempts": 1,
    }
    cached = {
        "input_tokens": 20,
        "completion_tokens": 4,
        "provider_latency_ms": 0.0,
        "queue_wait_ms": 0.0,
        "cache_hit": True,
        "physical_attempts": 0,
    }
    live_after_retry = {
        "input_tokens": 30,
        "completion_tokens": 6,
        "provider_latency_ms": 30.0,
        "queue_wait_ms": 4.0,
        "cache_hit": False,
        "physical_attempts": 2,
    }
    metrics = aggregate_operational_metrics(
        [
            {"round9_operational": [live_one, cached]},
            {"round9_operational": [live_after_retry]},
        ]
    )
    assert metrics == {
        "semantic_calls_total": 3,
        "semantic_calls_mean_per_case": 1.5,
        "semantic_calls_p50": 2.0,
        "semantic_calls_p95": 2.0,
        "physical_attempts_total": 3,
        "cache_hits_total": 1,
        "input_tokens_total": 60,
        "completion_tokens_total": 12,
        "tokens_total": 72,
        "provider_latency_ms_total": 40.0,
        "provider_latency_ms_mean_per_call": 20.0,
        "provider_latency_ms_p50": 30.0,
        "provider_latency_ms_p95": 30.0,
        "queue_wait_ms_total": 7.0,
        "cost_available": False,
        "cost_usd_total": None,
        "cost_usd_mean_per_case": None,
    }

    summary = tmp_path / "summary.json"
    generation = {"n": 2, "operational_metrics": metrics}
    evaluation = {"n": 2, "operational_metrics": metrics}
    write_protocol_summary(
        summary,
        generation=generation,
        evaluation=evaluation,
        arm=BFCLArm.NATIVE_AGENT,
        model="deepseek-v4-flash:0731",
        registry_name="round9-bfcl",
        official_commit="b" * 40,
    )
    assert json.loads(summary.read_text())["operational_metrics"] == metrics


def test_operational_aggregate_rejects_cache_hit_with_physical_request():
    invalid = {
        "input_tokens": 1,
        "completion_tokens": 1,
        "provider_latency_ms": 0.0,
        "queue_wait_ms": 0.0,
        "cache_hit": True,
        "physical_attempts": 1,
    }
    with pytest.raises(BFCLProtocolError):
        aggregate_operational_metrics([{"round9_operational": [invalid]}])


def test_public_provider_gate_summary_exposes_only_aggregate_contract():
    response = NormalizedChatResponse(
        content="provider prose must not be exported",
        tool_calls=(
            NormalizedToolCall(
                call_id="provider-call-id-must-not-be-exported",
                name="lookup_public_code",
                arguments={"item": "alpha"},
            ),
        ),
        prompt_tokens=25,
        completion_tokens=7,
        latency_seconds=0.5,
        queue_wait_seconds=0.25,
        cache_hit=False,
        physical_attempts=1,
    )
    summary = build_public_provider_gate_summary(
        response,
        model="deepseek-v4-flash:0731",
        journal_sha256="a" * 64,
    )
    encoded = json.dumps(summary)
    assert summary["status"] == "pass"
    assert summary["native_tool_call_count"] == 1
    assert summary["operational_metrics"]["tokens_total"] == 32
    assert "alpha" not in encoded
    assert "provider prose" not in encoded
    assert "provider-call-id" not in encoded
    assert "deepseek-v4-flash" not in encoded


def test_native_ollama_tool_result_executes_in_official_mock_and_uses_tool_name():
    pytest.importorskip("bfcl_eval")

    provider_model = "deepseek-v4-flash:0731"
    registry_name = "round9-bfcl-deepseek-agent"
    identity = make_bfcl_handler_identity(
        registry_name=registry_name,
        arm=BFCLArm.NATIVE_AGENT,
        provider_model=provider_model,
    )
    assert ":" not in identity
    assert identity.isidentifier()

    class FakeDeepSeekTransport:
        model = provider_model

        def __init__(self):
            self.calls = 0
            self.observed_assistant_message = None
            self.observed_tool_message = None

        def chat(self, *, semantic_id, messages, tools):
            assert tools
            self.calls += 1
            if self.calls == 1:
                return NormalizedChatResponse(
                    content="",
                    tool_calls=(
                        NormalizedToolCall(
                            call_id="call-public-fixture",
                            name="get_flight_cost",
                            arguments={
                                "travel_from": "JFK",
                                "travel_to": "HND",
                                "travel_date": "2026-12-24",
                                "travel_class": "first",
                            },
                        ),
                    ),
                    prompt_tokens=10,
                    completion_tokens=5,
                    latency_seconds=0.01,
                )
            self.observed_assistant_message = messages[-2]
            self.observed_tool_message = messages[-1]
            return NormalizedChatResponse(
                content="",
                tool_calls=(),
                prompt_tokens=12,
                completion_tokens=1,
                latency_seconds=0.01,
            )

    transport = FakeDeepSeekTransport()
    handler_class = build_official_handler_class(
        transport=transport,
        policy=ArmPolicy(BFCLArm.NATIVE_AGENT, 4),
    )
    handler = handler_class(
        model_name=identity,
        temperature=0.0,
        registry_name=registry_name,
        is_fc_model=True,
    )
    case = {
        "id": "multi_turn_miss_param_round9_public_fixture",
        "question": [[{"role": "user", "content": "Find this flight."}]],
        "function": [
            {
                "name": "get_flight_cost",
                "description": "Get the cost of a flight.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "travel_from": {"type": "string"},
                        "travel_to": {"type": "string"},
                        "travel_date": {"type": "string"},
                        "travel_class": {"type": "string"},
                    },
                    "required": [
                        "travel_from",
                        "travel_to",
                        "travel_date",
                        "travel_class",
                    ],
                },
            }
        ],
        "involved_classes": ["TravelAPI"],
        "initial_config": {},
    }
    result, metadata = handler.inference(
        case,
        include_input_log=False,
        exclude_state_log=True,
    )
    assert transport.calls == 2
    assert transport.observed_assistant_message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "function": {
                    "name": "get_flight_cost",
                    "arguments": {
                        "travel_from": "JFK",
                        "travel_to": "HND",
                        "travel_date": "2026-12-24",
                        "travel_class": "first",
                    },
                }
            }
        ],
    }
    assert transport.observed_tool_message == {
        "role": "tool",
        "tool_name": "get_flight_cost",
        "content": transport.observed_tool_message["content"],
    }
    assert not transport.observed_tool_message["content"].startswith(
        "Error during execution:"
    )
    assert result[0][0][0].keys() == {"get_flight_cost"}
    assert metadata["input_token_count"] == [[10, 12]]
    assert metadata["round9_operational"] == [
        {
            "input_tokens": 10,
            "completion_tokens": 5,
            "provider_latency_ms": 10.0,
            "queue_wait_ms": 0.0,
            "cache_hit": False,
            "physical_attempts": 1,
        },
        {
            "input_tokens": 12,
            "completion_tokens": 1,
            "provider_latency_ms": 10.0,
            "queue_wait_ms": 0.0,
            "cache_hit": False,
            "physical_attempts": 1,
        },
    ]
    assert transport.model == provider_model


def test_official_horizon_prevents_four_call_adapter_failure():
    """Reproduce the v2 failure at the actual pinned BFCL handler seam."""
    pytest.importorskip("bfcl_eval")

    provider_model = "deepseek-v4-flash:0731"

    class FiveResponseTransport:
        model = provider_model

        def __init__(self):
            self.calls = 0

        def chat(self, *, semantic_id, messages, tools):
            self.calls += 1
            if self.calls <= 4:
                return NormalizedChatResponse(
                    content="",
                    tool_calls=(
                        NormalizedToolCall(
                            call_id=f"public-call-{self.calls}",
                            name="get_flight_cost",
                            arguments={
                                "travel_from": "JFK",
                                "travel_to": "HND",
                                "travel_date": "2026-12-24",
                                "travel_class": "first",
                            },
                        ),
                    ),
                    prompt_tokens=10,
                    completion_tokens=5,
                    latency_seconds=0.01,
                )
            return NormalizedChatResponse(
                content="The tool sequence is complete.",
                tool_calls=(),
                prompt_tokens=12,
                completion_tokens=4,
                latency_seconds=0.01,
            )

    case = {
        "id": "multi_turn_miss_param_round9_horizon_fixture",
        "question": [[{"role": "user", "content": "Find this flight."}]],
        "function": [
            {
                "name": "get_flight_cost",
                "description": "Get the cost of a flight.",
                "parameters": {
                    "type": "dict",
                    "properties": {
                        "travel_from": {"type": "string"},
                        "travel_to": {"type": "string"},
                        "travel_date": {"type": "string"},
                        "travel_class": {"type": "string"},
                    },
                    "required": [
                        "travel_from",
                        "travel_to",
                        "travel_date",
                        "travel_class",
                    ],
                },
            }
        ],
        "involved_classes": ["TravelAPI"],
        "initial_config": {},
    }

    capped_transport = FiveResponseTransport()
    capped_class = build_official_handler_class(
        transport=capped_transport,
        policy=ArmPolicy(BFCLArm.NATIVE_AGENT, 4),
    )
    capped_handler = capped_class(
        model_name=make_bfcl_handler_identity(
            registry_name="round9-bfcl-cap-repro",
            arm=BFCLArm.NATIVE_AGENT,
            provider_model=provider_model,
        ),
        temperature=0.0,
        registry_name="round9-bfcl-cap-repro",
        is_fc_model=True,
    )
    with pytest.raises(BFCLProtocolError):
        capped_handler.inference(
            case,
            include_input_log=False,
            exclude_state_log=True,
        )
    assert capped_transport.calls == 4

    corrected_transport = FiveResponseTransport()
    corrected_class = build_official_handler_class(
        transport=corrected_transport,
        policy=ArmPolicy(
            BFCLArm.NATIVE_AGENT,
            BFCL_OFFICIAL_MAX_PHYSICAL_CALLS_PER_TURN,
        ),
    )
    corrected_handler = corrected_class(
        model_name=make_bfcl_handler_identity(
            registry_name="round9-bfcl-horizon-correction",
            arm=BFCLArm.NATIVE_AGENT,
            provider_model=provider_model,
        ),
        temperature=0.0,
        registry_name="round9-bfcl-horizon-correction",
        is_fc_model=True,
    )
    result, metadata = corrected_handler.inference(
        case,
        include_input_log=False,
        exclude_state_log=True,
    )
    assert corrected_transport.calls == 5
    assert len(result) == 1
    assert len(metadata["round9_operational"]) == 5
