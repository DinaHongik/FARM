from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import run_round8 as runner  # noqa: E402
from executable_applet import (  # noqa: E402
    AppletCompiler,
    SandboxConnectorRegistry,
)


def _trigger(index: int) -> dict:
    return {
        "url": f"trigger://mail/event-{index}",
        "kind": "trigger",
        "channel": "mail" if index == 0 else f"trigger-service-{index}",
        "channel_display": "Mail" if index == 0 else f"Trigger Service {index}",
        "function_name": "New message" if index == 0 else f"Trigger {index}",
        "description": "Fires for a new message" if index == 0 else f"Trigger description {index}",
        "retrieval_rank": index + 1,
        "retrieval_score": 1.0 / (index + 1),
        "input_fields": [],
        "ingredients": [
            {"slug": "body", "type": "String", "required": True},
            {"slug": "count", "type": "Integer", "required": True},
        ],
    }


def _action(index: int) -> dict:
    return {
        "url": f"action://chat/action-{index}",
        "kind": "action",
        "channel": "chat" if index == 0 else f"action-service-{index}",
        "channel_display": "Chat" if index == 0 else f"Action Service {index}",
        "function_name": "Post message" if index == 0 else f"Action {index}",
        "description": "Posts a message" if index == 0 else f"Action description {index}",
        "retrieval_rank": index + 1,
        "retrieval_score": 1.0 / (index + 1),
        "ingredients": [],
        "input_fields": [
            {"slug": "body", "type": "String", "required": True},
            {"slug": "room", "type": "String", "required": True},
        ],
    }


def _fixture() -> tuple[dict, dict]:
    triggers = [_trigger(index) for index in range(10)]
    actions = [_action(index) for index in range(10)]
    row = {
        "group_id": "screen-case-one",
        "semantic_family_id": "family-one",
        "query": "When a new mail message arrives, post the body to chat",
        "trigger_candidates": [
            {
                "url": item["url"],
                "retrieval_rank": index + 1,
                "retrieval_score": item["retrieval_score"],
                "channel": item["channel"],
                "seed_ranks": {"seed42": index + 1, "seed1337": index + 1, "seed2025": index + 1},
                "text_plain": item["description"],
                "text_schema": item["description"] + " schema",
            }
            for index, item in enumerate(triggers)
        ],
        "action_candidates": [
            {
                "url": item["url"],
                "retrieval_rank": index + 1,
                "retrieval_score": item["retrieval_score"],
                "channel": item["channel"],
                "seed_ranks": {"seed42": index + 1, "seed1337": index + 1, "seed2025": index + 1},
                "text_plain": item["description"],
                "text_schema": item["description"] + " schema",
            }
            for index, item in enumerate(actions)
        ],
        "valid_pairs": [
            {"trigger_url": triggers[0]["url"], "action_url": actions[0]["url"]}
        ],
    }
    corpora = {
        "trigger": {item["url"]: item for item in triggers},
        "action": {item["url"]: item for item in actions},
    }
    return row, corpora


class FakeResponse:
    def __init__(self, arguments: dict, *, tool: str = "submit_selection", status: int = 200, model: str = "qwen3.5:397b", done: bool = True, done_reason: str = "stop"):
        self.status_code = status
        self._payload = {
            "model": model,
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"function": {"name": tool, "arguments": arguments}}],
            },
            "done": done,
            "done_reason": done_reason,
            "prompt_eval_count": 17,
            "eval_count": 5,
        }
        self.content = json.dumps(self._payload).encode()

    def json(self) -> dict:
        return self._payload


class FakeHTTP:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = list(responses)
        self.requests: list[dict] = []

    def post(self, endpoint: str, **kwargs):
        self.requests.append({"endpoint": endpoint, **kwargs})
        return self.responses.pop(0)

    def close(self) -> None:
        return None


class Round8RunnerTests(unittest.TestCase):
    def test_discovery_selector_accepts_only_the_frozen_family_purged_screen(self) -> None:
        rows = []
        for index in range(231):
            row, _ = _fixture()
            row["group_id"] = f"g-{index}"
            row["semantic_family_id"] = f"family-{index}"
            row["prior_agent_output_must_not_propagate"] = {"choice": "gold-like"}
            rows.append(row)
        expected = runner.ids_hash(rows)
        with patch.object(runner, "SCREEN_HASH", expected):
            selected, metadata = runner.select_discovery_rows(rows)

        self.assertEqual(231, len(selected))
        self.assertTrue(
            all("prior_agent_output_must_not_propagate" not in row for row in selected)
        )
        self.assertEqual(0, metadata["reserved_confirmation_rows_read"])
        self.assertEqual("already_consumed_discovery_only", metadata["source_scope"])

    def test_discovery_selector_rejects_nonfrozen_row_counts(self) -> None:
        row, _ = _fixture()
        with self.assertRaisesRegex(ValueError, "exactly 231"):
            runner.select_discovery_rows([row])

    def test_consensus_smoke_selects_first_n_routed_rows_and_hashes_that_selection(self) -> None:
        rows = [
            {"group_id": f"g{index}", "private_score": score}
            for index, score in enumerate((0.2, 0.41, 0.3, 0.8, 0.7))
        ]
        with patch.object(
            runner,
            "frozen_routing_score",
            side_effect=lambda row, _router: row["private_score"],
        ):
            selected = runner.select_smoke_rows(
                rows,
                arm="trigger_consensus_executable",
                count=2,
                consensus_router={"frozen": True},
            )
        self.assertEqual([row["group_id"] for row in selected], ["g1", "g3"])
        self.assertEqual(
            runner.ids_hash(selected),
            runner.ids_hash([rows[1], rows[3]]),
        )
        self.assertNotEqual(runner.ids_hash(selected), runner.ids_hash(rows[:2]))

    def test_model_view_is_rank_url_gold_and_case_id_free(self) -> None:
        row, corpora = _fixture()
        prepared = runner.prepare_case(row, corpora)
        public = runner.public_case_payload(prepared)
        serialized = json.dumps(public, sort_keys=True)

        self.assertNotIn("screen-case-one", serialized)
        self.assertNotIn("family-one", serialized)
        self.assertNotIn("://", serialized)
        self.assertNotIn("retrieval_rank", serialized)
        self.assertNotIn("retrieval_score", serialized)
        self.assertNotIn("valid_pairs", serialized)
        self.assertNotIn("gold", serialized.casefold())
        self.assertNotIn("baseline", serialized.casefold())
        self.assertEqual(10, len(public["trigger_candidates"]))
        self.assertEqual(10, len(public["action_candidates"]))
        self.assertIn(prepared.baseline_trigger_alias, {c["candidate_id"] for c in public["trigger_candidates"]})
        self.assertIn(prepared.baseline_action_alias, {c["candidate_id"] for c in public["action_candidates"]})
        self.assertTrue(all("description" in item for item in public["trigger_candidates"]))

    def test_model_view_exposes_context_contract_not_private_canary_values(self) -> None:
        row, corpora = _fixture()
        first_action = row["action_candidates"][0]["url"]
        corpora["action"][first_action]["input_fields"].append(
            {"slug": "url", "type": "String", "required": True}
        )

        prepared = runner.prepare_case(row, corpora)
        public = runner.public_case_payload(prepared)

        self.assertNotIn("context_payload", public)
        self.assertTrue(
            any(
                field["path"].endswith(".url")
                for field in public["available_context_fields"]
            )
        )

    def test_consensus_views_are_independently_aliased_rank_hidden_and_gold_free(self) -> None:
        row, corpora = _fixture()
        prepared = runner.prepare_case(
            row,
            corpora,
            consensus_routing_score=0.5,
            require_consensus_metadata=True,
        )
        schema = prepared.consensus_views["schema_m5"]
        fused = prepared.consensus_views["fused_m10"]
        self.assertEqual(len(schema.public_candidates), 5)
        self.assertEqual(len(fused.public_candidates), 10)
        self.assertTrue(all(alias.startswith("S") for alias in schema.alias_to_primary))
        self.assertTrue(all(alias.startswith("F") for alias in fused.alias_to_primary))
        self.assertTrue(set(schema.alias_to_primary).isdisjoint(fused.alias_to_primary))
        self.assertEqual(set(schema.primary_candidate_ids), {
            next(
                alias
                for alias, identity in prepared.alias_to_identity.items()
                if identity == f"trigger://mail/event-{index}"
            )
            for index in range(5)
        })
        serialized = json.dumps(
            {
                "schema": schema.public_candidates,
                "fused": fused.public_candidates,
            },
            sort_keys=True,
        )
        for forbidden in (
            "://",
            "retrieval_rank",
            "retrieval_score",
            "seed_ranks",
            "valid_pairs",
            "baseline",
            "gold",
        ):
            self.assertNotIn(forbidden, serialized.casefold())

    def test_consensus_runner_uses_dynamic_view_schemas_and_keeps_action(self) -> None:
        row, corpora = _fixture()
        prepared = runner.prepare_case(
            row,
            corpora,
            consensus_routing_score=0.5,
            require_consensus_metadata=True,
        )
        target_identity = "trigger://mail/event-1"
        target_primary = next(
            alias
            for alias, identity in prepared.alias_to_identity.items()
            if identity == target_identity
        )
        choices = []
        for name in ("schema_m5", "fused_m10"):
            view = prepared.consensus_views[name]
            choices.append(
                next(
                    alias
                    for alias, primary in view.alias_to_primary.items()
                    if primary == target_primary
                )
            )

        class ScriptedClient:
            def __init__(self):
                self.calls = []

            def call_tool(self, **kwargs):
                self.calls.append(kwargs)
                choice = choices[len(self.calls) - 1]
                value = kwargs["output_model"].model_validate(
                    {"choice": choice}, strict=True
                )
                return runner.ToolCall(
                    kwargs["role"], True, value, 1, 1, 1, 3, 1, 0.01, None, True
                )

        client = ScriptedClient()
        outcome = runner.run_case(
            "trigger_consensus_executable",
            prepared,
            client,
            "qwen3.5:397b",
        )
        self.assertEqual(outcome.terminal_status, "executed")
        self.assertEqual(outcome.final_trigger_alias, target_primary)
        self.assertEqual(
            outcome.final_action_alias, prepared.baseline_action_alias
        )
        self.assertEqual(outcome.semantic_calls, 2)
        self.assertEqual(
            [call["role"] for call in client.calls],
            ["trigger_schema_m5", "trigger_fused_m10"],
        )
        for request, prefix in zip(client.calls, ("S", "F"), strict=True):
            enum = request["parameter_schema"]["properties"]["choice"]["enum"]
            self.assertIn("DEFER_TO_RETRIEVER", enum)
            self.assertTrue(all(x == "DEFER_TO_RETRIEVER" or x.startswith(prefix) for x in enum))
            self.assertNotIn(target_primary, enum)
        self.assertEqual(
            outcome.policy_trace["decision_reason"],
            "consensus_compiled_and_executed",
        )

    def test_deterministic_autowire_prefers_exact_typed_trigger_output_then_context(self) -> None:
        row, corpora = _fixture()
        prepared = runner.prepare_case(row, corpora)
        program = runner.deterministic_autowire(
            prepared,
            prepared.baseline_trigger_alias,
            prepared.baseline_action_alias,
        )

        sources = {binding.target_path: binding.source for binding in program.bindings}
        self.assertEqual("trigger_output", sources["body"].kind)
        self.assertEqual("body", sources["body"].path)
        self.assertEqual("context", sources["room"].kind)
        self.assertTrue(sources["room"].path.startswith("config."))

        compiled = AppletCompiler(
            prepared.candidates,
            context_fields=prepared.context_fields,
        ).compile(program)
        result = SandboxConnectorRegistry(prepared.candidates).execute(
            compiled,
            context=prepared.context,
        )
        self.assertTrue(result.success)

    def test_dspy_failure_falls_back_to_private_baseline_not_shuffled_first(self) -> None:
        row, corpora = _fixture()
        prepared = None
        for index in range(100):
            row["group_id"] = f"fallback-case-{index}"
            candidate = runner.prepare_case(row, corpora)
            if candidate.baseline_trigger_alias != "T01" or candidate.baseline_action_alias != "A01":
                prepared = candidate
                break
        self.assertIsNotNone(prepared)

        class FailingClient:
            def call_tool(self, **kwargs):
                return runner.ToolCall(
                    kwargs["role"], False, None, 1, 0, 1, 0, 0, 0.0, "forced_failure", False
                )

        outcome = runner.run_case(
            "typed_schema_plan", prepared, FailingClient(), "qwen3.5:397b"
        )
        self.assertEqual("safe_fallback", outcome.terminal_status)
        self.assertEqual(prepared.baseline_trigger_alias, outcome.final_trigger_alias)
        self.assertEqual(prepared.baseline_action_alias, outcome.final_action_alias)

    def test_native_adapter_uses_api_chat_and_strict_terminal_tool(self) -> None:
        response = FakeResponse(
            {"trigger_choice": "DEFER_TO_RETRIEVER", "action_choice": "A03"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "attempts.jsonl"
            http = FakeHTTP([response])
            client = runner.NativeOllamaToolClient(
                host="https://ollama.example",
                api_key="never-persist-this",
                model="qwen3.5:397b",
                journal_path=journal,
                client=http,
                retry_delay=0,
            )
            call = client.call_tool(
                semantic_id="opaque-semantic-call",
                role="joint_endpoint_selector",
                public_request={"query": "do a safe thing"},
                tool_name="submit_selection",
                description="Submit one selection.",
                output_model=runner.DirectSelection,
            )

            self.assertTrue(call.ok)
            self.assertEqual("A03", call.value.action_choice)
            self.assertTrue(http.requests[0]["endpoint"].endswith("/api/chat"))
            payload = http.requests[0]["json"]
            self.assertNotIn("format", payload)
            self.assertEqual("submit_selection", payload["tools"][0]["function"]["name"])
            self.assertFalse(payload["stream"])
            content = journal.read_text()
            self.assertNotIn("never-persist-this", content)
            self.assertNotIn("do a safe thing", content)
            self.assertEqual(2, len(content.splitlines()))

    def test_local_ollama_requires_literal_loopback_and_is_output_isolated(self) -> None:
        self.assertEqual(
            "http://127.0.0.1:11434",
            runner.validate_local_ollama_host("http://127.0.0.1:11434"),
        )
        for unsafe in (
            "http://0.0.0.0:11434",
            "http://192.168.1.7:11434",
            "https://127.0.0.1:11434",
            "http://localhost.example:11434",
            "http://user@localhost:11434",
            "http://localhost",
            "http://localhost:22110",
        ):
            with self.subTest(unsafe=unsafe), self.assertRaises(ValueError):
                runner.validate_local_ollama_host(unsafe)

        root = Path("/tmp/round8")
        digest = "a" * 64
        self.assertEqual(
            root / "local_granite" / digest / "full",
            runner.output_root_for_mode(
                root, synthetic=False, local=True, smoke=None, model_digest=digest
            ),
        )
        self.assertEqual(
            root / "local_granite" / digest / "smoke-2",
            runner.output_root_for_mode(
                root, synthetic=False, local=True, smoke=2, model_digest=digest
            ),
        )
        self.assertNotEqual(
            runner.output_root_for_mode(
                root, synthetic=False, local=True, smoke=1, model_digest=digest
            ),
            runner.output_root_for_mode(
                root, synthetic=False, local=True, smoke=2, model_digest=digest
            ),
        )
        self.assertFalse(runner.transport_metadata(local=True)["data_left_dgx"])

    def test_local_native_adapter_sends_no_authorization_and_omits_think(self) -> None:
        response = FakeResponse(
            {"trigger_choice": "DEFER_TO_RETRIEVER", "action_choice": "A03"},
            model="granite4:small-h",
        )
        with tempfile.TemporaryDirectory() as temporary:
            http = FakeHTTP([response])
            client = runner.NativeOllamaToolClient(
                host="http://127.0.0.1:11434",
                api_key=None,
                model="granite4:small-h",
                journal_path=Path(temporary) / "attempts.jsonl",
                client=http,
                think=None,
            )
            result = client.call_tool(
                semantic_id="local-call",
                role="joint_endpoint_selector",
                public_request={"query": "safe local request"},
                tool_name="submit_selection",
                description="Submit one selection.",
                output_model=runner.DirectSelection,
            )

        self.assertTrue(result.ok)
        request = http.requests[0]
        self.assertNotIn("Authorization", request["headers"])
        self.assertNotIn("think", request["json"])

    def test_cloud_native_adapter_keeps_bearer_header_and_low_think(self) -> None:
        response = FakeResponse(
            {"trigger_choice": "DEFER_TO_RETRIEVER", "action_choice": "A03"},
        )
        with tempfile.TemporaryDirectory() as temporary:
            http = FakeHTTP([response])
            client = runner.NativeOllamaToolClient(
                host="https://ollama.example",
                api_key="cloud-secret",
                model="qwen3.5:397b",
                journal_path=Path(temporary) / "attempts.jsonl",
                client=http,
            )
            client.call_tool(
                semantic_id="cloud-call",
                role="joint_endpoint_selector",
                public_request={"query": "safe cloud request"},
                tool_name="submit_selection",
                description="Submit one selection.",
                output_model=runner.DirectSelection,
            )

        request = http.requests[0]
        self.assertEqual("Bearer cloud-secret", request["headers"]["Authorization"])
        self.assertEqual("low", request["json"]["think"])

    def test_dynamic_plan_schema_enumerates_paths_and_binder_fixes_endpoints(self) -> None:
        row, corpora = _fixture()
        prepared = runner.prepare_case(row, corpora)
        schema = runner.plan_tool_schema(prepared)
        program = schema["properties"]["program"]
        trigger_ids = program["properties"]["trigger"]["properties"]["candidate_id"]["enum"]
        action_ids = program["properties"]["action"]["properties"]["candidate_id"]["enum"]
        self.assertEqual(
            {item.candidate_id for item in prepared.candidates.triggers}, set(trigger_ids)
        )
        self.assertEqual(
            {item.candidate_id for item in prepared.candidates.actions}, set(action_ids)
        )
        serialized = json.dumps(schema, sort_keys=True)
        self.assertNotIn("event_finished", serialized)
        self.assertNotIn("://", serialized)

        fixed = runner.plan_tool_schema(
            prepared,
            fixed_trigger=prepared.baseline_trigger_alias,
            fixed_action=prepared.baseline_action_alias,
        )
        fixed_program = fixed["properties"]["program"]
        self.assertEqual(
            prepared.baseline_trigger_alias,
            fixed_program["properties"]["trigger"]["properties"]["candidate_id"]["const"],
        )
        self.assertEqual(
            prepared.baseline_action_alias,
            fixed_program["properties"]["action"]["properties"]["candidate_id"]["const"],
        )

    def test_exact_dynamic_schema_is_enforced_and_journal_has_only_error_shape(self) -> None:
        response = FakeResponse({"trigger_choice": "T99", "action_choice": "A01"})
        schema = runner._tool_schema(
            runner.DirectSelection,
            {
                "trigger_choice": ["DEFER_TO_RETRIEVER", "T01"],
                "action_choice": ["DEFER_TO_RETRIEVER", "A01"],
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            journal = Path(temporary) / "attempts.jsonl"
            http = FakeHTTP([response])
            client = runner.NativeOllamaToolClient(
                "https://ollama.example", "secret", "qwen3.5:397b", journal, client=http
            )
            call = client.call_tool(
                semantic_id="schema-call",
                role="joint_endpoint_selector",
                public_request={"query": "private query value"},
                tool_name="submit_selection",
                description="Submit one selection.",
                output_model=runner.DirectSelection,
                parameter_schema=schema,
            )
            events = [json.loads(line) for line in journal.read_text().splitlines()]

        self.assertFalse(call.ok)
        self.assertFalse(call.schema_valid)
        self.assertEqual("invalid_tool_schema", call.error_code)
        diagnostics = events[-1]["validation_errors"]
        self.assertTrue(diagnostics)
        self.assertEqual({"loc", "type"}, set(diagnostics[0]))
        journal_text = json.dumps(events)
        self.assertNotIn("T99", journal_text)
        self.assertNotIn("private query value", journal_text)

    def test_response_envelope_must_match_model_and_terminal_completion(self) -> None:
        responses = (
            (FakeResponse({"trigger_choice": "T01", "action_choice": "A01"}, model="other:model"), "response_model_mismatch"),
            (FakeResponse({"trigger_choice": "T01", "action_choice": "A01"}, done=False), "response_not_done"),
            (FakeResponse({"trigger_choice": "T01", "action_choice": "A01"}, done_reason="length"), "nonterminal_done_reason"),
        )
        for index, (response, expected) in enumerate(responses):
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as temporary:
                client = runner.NativeOllamaToolClient(
                    "https://ollama.example", "secret", "qwen3.5:397b",
                    Path(temporary) / f"attempt-{index}.jsonl", client=FakeHTTP([response]),
                )
                call = client.call_tool(
                    semantic_id=f"envelope-{index}", role="joint_endpoint_selector",
                    public_request={"query": "safe"}, tool_name="submit_selection",
                    description="Submit.", output_model=runner.DirectSelection,
                )
                self.assertFalse(call.ok)
                self.assertEqual(expected, call.error_code)

    def test_local_http_client_ignores_proxy_environment_and_digest_is_full(self) -> None:
        with patch("httpx.Client") as constructor:
            client = runner.NativeOllamaToolClient(
                "http://127.0.0.1:11434", None, "granite4:small-h",
                Path("/tmp/not-used.jsonl"),
            )
            constructor.assert_called_once_with(timeout=240, trust_env=False)
            client.close()

        common = [
            "--run-root", str(ROOT), "--arm", "typed_direct",
            "--local-ollama-host", "http://127.0.0.1:11434",
            "--model", "granite4:small-h", "--expected-digest",
        ]
        with self.assertRaises(SystemExit):
            runner.parse_args([*common, "2f8a7367d441"])
        parsed = runner.parse_args([*common, "2" * 64])
        self.assertEqual("2" * 64, parsed.expected_digest)

    def test_resume_requires_matching_progress_and_canonical_record_prefix(self) -> None:
        selected = [
            {"group_id": "g1"}, {"group_id": "g2"}, {"group_id": "g3"}
        ]
        binding = {"arm": "typed_direct", "runner_sha256": "a" * 64}
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"
            attempts = root / "attempts.jsonl"
            progress = root / "progress.json"
            records.write_text(
                json.dumps({
                    "schema_version": "farm_round8_executable_record_v1",
                    "group_id": "g1", "arm_id": "typed_direct",
                }) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "progress"):
                runner.load_resume_state(
                    records, attempts, progress, binding, selected, "typed_direct"
                )

            runner.write_json_atomic(progress, {
                "phase": "running", "binding": binding,
                "completed_rows": 1, "target_rows": 3,
            })
            resumed = runner.load_resume_state(
                records, attempts, progress, binding, selected, "typed_direct"
            )
            self.assertEqual(["g1"], [row["group_id"] for row in resumed])

            records.write_text(
                "\n".join(json.dumps({
                    "schema_version": "farm_round8_executable_record_v1",
                    "group_id": group, "arm_id": "typed_direct",
                }) for group in ("g2", "g1")) + "\n",
                encoding="utf-8",
            )
            runner.write_json_atomic(progress, {
                "phase": "running", "binding": binding,
                "completed_rows": 2, "target_rows": 3,
            })
            with self.assertRaisesRegex(RuntimeError, "canonical"):
                runner.load_resume_state(
                    records, attempts, progress, binding, selected, "typed_direct"
                )

            records.write_text(
                "\n".join(json.dumps({
                    "schema_version": "farm_round8_executable_record_v1",
                    "group_id": "g1", "arm_id": "typed_direct",
                }) for _ in range(2)) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                runner.load_resume_state(
                    records, attempts, progress, binding, selected, "typed_direct"
                )

    def test_resume_refuses_changed_binding_and_invalid_completed_result(self) -> None:
        selected = [{"group_id": "g1"}]
        binding = {"arm": "typed_direct", "runner_sha256": "a" * 64}
        record = {
            "schema_version": "farm_round8_executable_record_v1",
            "group_id": "g1", "arm_id": "typed_direct",
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = root / "records.jsonl"; attempts = root / "attempts.jsonl"
            progress = root / "progress.json"; result = root / "result.json"
            records.write_text(json.dumps(record) + "\n", encoding="utf-8")
            runner.write_json_atomic(progress, {
                "phase": "running", "binding": {**binding, "runner_sha256": "b" * 64},
                "completed_rows": 1, "target_rows": 1,
            })
            with self.assertRaisesRegex(RuntimeError, "binding"):
                runner.load_resume_state(
                    records, attempts, progress, binding, selected, "typed_direct"
                )

            runner.write_json_atomic(result, {
                "status": "completed", "binding": binding, "metrics": {"rows": 2}
            })
            runner.write_json_atomic(progress, {
                "phase": "complete", "binding": binding,
                "completed_rows": 1, "target_rows": 1,
                "output": str(result), "output_sha256": runner.sha256_file(result),
            })
            with self.assertRaisesRegex(RuntimeError, "row count"):
                runner.validate_completed_state(
                    result, progress, records, binding, selected, "typed_direct"
                )

            runner.write_json_atomic(result, {
                "status": "completed", "binding": binding, "metrics": {"rows": 1}
            })
            # The old progress digest must make the tampered replacement fail.
            with self.assertRaisesRegex(RuntimeError, "hash"):
                runner.validate_completed_state(
                    result, progress, records, binding, selected, "typed_direct"
                )

    def test_invalid_tool_arguments_fail_without_hidden_semantic_retry(self) -> None:
        response = FakeResponse(
            {"trigger_choice": "T01", "action_choice": "A01", "invented": True},
        )
        with tempfile.TemporaryDirectory() as temporary:
            http = FakeHTTP([response])
            client = runner.NativeOllamaToolClient(
                host="https://ollama.example",
                api_key="secret",
                model="qwen3.5:397b",
                journal_path=Path(temporary) / "attempts.jsonl",
                client=http,
                retry_delay=0,
            )
            call = client.call_tool(
                semantic_id="opaque-semantic-call",
                role="joint_endpoint_selector",
                public_request={"query": "safe"},
                tool_name="submit_selection",
                description="Submit one selection.",
                output_model=runner.DirectSelection,
            )

        self.assertFalse(call.ok)
        self.assertEqual("invalid_tool_schema", call.error_code)
        self.assertFalse(call.schema_valid)
        self.assertEqual(1, call.transport_attempts)
        self.assertEqual(1, len(http.requests))

    def test_summary_reports_oracle_conditioning_rescues_execution_and_cost(self) -> None:
        records = [
            runner.synthetic_metric_record(before=False, after=True, oracle=True, service="mail"),
            runner.synthetic_metric_record(before=True, after=False, oracle=True, service="mail"),
            runner.synthetic_metric_record(before=True, after=True, oracle=False, service="calendar"),
        ]
        summary = runner.summarize_records(records)

        self.assertEqual(1, summary["function_joint"]["rescues"])
        self.assertEqual(1, summary["function_joint"]["regressions"])
        self.assertEqual(2, summary["oracle_conditioned_selection"]["function_joint"]["eligible"])
        self.assertIn("compile_rate", summary["execution"])
        self.assertIn("prompt_tokens", summary["cost"])
        self.assertIn("mail", summary["per_service"]["trigger"])

    def test_synthetic_cli_writes_fresh_resumable_artifacts_for_each_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / ROOT.name
            run_root.mkdir()
            (run_root / "EXPERIMENT_MATRIX.json").write_text(
                (ROOT / "EXPERIMENT_MATRIX.json").read_text(), encoding="utf-8"
            )
            for arm in runner.ARMS:
                code = runner.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--arm",
                        arm,
                        "--synthetic",
                        "--smoke",
                        "2",
                    ]
                )
                self.assertEqual(0, code)
                result = run_root / "synthetic" / "results" / f"{arm}.json"
                records = run_root / "synthetic" / "records" / f"{arm}.jsonl"
                attempts = run_root / "synthetic" / "attempts" / f"{arm}.jsonl"
                self.assertTrue(result.is_file())
                self.assertEqual(2, len(records.read_text().splitlines()))
                self.assertGreaterEqual(len(attempts.read_text().splitlines()), 4)
                payload = json.loads(result.read_text())
                self.assertEqual("completed", payload["status"])
                self.assertEqual(2, payload["metrics"]["rows"])


if __name__ == "__main__":
    unittest.main()
