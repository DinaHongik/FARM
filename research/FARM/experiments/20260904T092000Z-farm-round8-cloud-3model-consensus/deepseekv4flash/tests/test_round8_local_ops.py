from __future__ import annotations

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS))

import launch_round8_local as local_ops  # noqa: E402
import run_local_ollama_server as server_job  # noqa: E402
import run_round8_local_job as experiment_job  # noqa: E402
import verify_round8_local as local_verify  # noqa: E402


class FakeTagsResponse:
    def __init__(self, digest: str):
        self.status = 200
        self._digest = digest

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self) -> bytes:
        return json.dumps(
            {"models": [{"model": local_ops.MODEL, "digest": self._digest}]}
        ).encode()


class FakeOpener:
    def __init__(self, digest: str):
        self.digest = digest

    def open(self, *_args, **_kwargs):
        return FakeTagsResponse(self.digest)


class Round8LocalOpsTests(unittest.TestCase):
    @staticmethod
    def make_bound_run_root(temporary: str) -> Path:
        run_root = Path(temporary) / local_ops.RUN_ID
        for relative in local_ops.PROVENANCE_FILES:
            path = run_root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f"fixture:{relative}\n", encoding="utf-8")
        return run_root

    @staticmethod
    def make_runner_binding(
        run_root: Path, slot: local_ops.Slot, *, smoke: int | None = None
    ) -> dict:
        hashes = local_ops.provenance_hashes(run_root)
        executed_rows, executed_ids_sha256 = local_ops.selection_identity(smoke)
        binding = {
            "run_id": local_ops.RUN_ID,
            "mode": "local_granite" if smoke is None else "local_granite_smoke",
            "arm": slot.arm,
            "dataset_id": local_ops.DATASET_ID,
            "matrix_sha256": hashes["EXPERIMENT_MATRIX.json"],
            "runner_sha256": hashes["scripts/run_round8.py"],
            "dspy_executable_agent_sha256": hashes[
                "scripts/dspy_executable_agent.py"
            ],
            "executable_applet_sha256": hashes["scripts/executable_applet.py"],
            "model": local_ops.MODEL,
            "model_digest": local_ops.MODEL_DIGEST,
            "data_left_dgx": False,
            "selection": {
                "executed_rows": executed_rows,
                "executed_group_ids_sha256": executed_ids_sha256,
            },
            "protocol": {
                "transport": "local_loopback",
                "data_left_dgx": False,
                "gold_at_inference_boundary": False,
                "live_connectors": False,
            },
        }
        if slot.arm == "trigger_consensus_executable":
            binding["trigger_consensus_policy"] = {
                "router_runtime_sha256": hashes["scripts/frozen_router.py"]
            }
        return binding

    @staticmethod
    def smoke_record(
        slot: local_ops.Slot,
        group_id: str,
        *,
        valid_native: bool = True,
        executed: bool = True,
        routed: bool = True,
        call_count: int | None = None,
    ) -> dict:
        if call_count is None:
            call_count = 3 if slot.arm == "dspy_factorized" else 1
        calls = [
            {
                "role": f"role-{index}",
                "ok": valid_native,
                "schema_valid": valid_native,
                "transport_attempts": 1,
                "model_tool_calls": 1 if valid_native else 0,
                "provider_requests": 1,
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "latency_seconds": 0.1,
                "error_code": None if valid_native else "invalid_tool_call_count",
            }
            for index in range(call_count)
        ]
        record = {
            "schema_version": "farm_round8_executable_record_v1",
            "group_id": group_id,
            "arm_id": slot.arm,
            "calls": calls,
            "accounting": {"semantic_calls": len(calls)},
            "protocol_valid": valid_native,
            "terminal_status": "executed" if executed else "safe_fallback",
            "fallback_used": not executed,
            "program": {"version": "farm.applet/v1"} if executed else None,
            "execution": {
                "strict_parse": valid_native,
                "compilation_attempts": 1 if executed else 0,
                "compiled": executed,
                "sandbox_attempts": 1 if executed else 0,
                "sandbox_run": executed,
                "receipt_sha256": "a" * 64 if executed else None,
            },
        }
        if slot.arm == "trigger_consensus_executable":
            record["policy_trace"] = {
                "routed": routed,
                "schema_m5_called": bool(call_count),
                "fused_m10_called": call_count > 1,
            }
        return record

    def test_frozen_arm_gpu_port_mapping_and_digest_namespace(self) -> None:
        self.assertEqual(server_job.DIGEST, local_ops.MODEL_DIGEST)
        self.assertEqual(experiment_job.DIGEST, local_ops.MODEL_DIGEST)
        self.assertEqual(experiment_job.MODEL, local_ops.MODEL)
        self.assertEqual(
            Path("/raid/session/aicontents/.local/ollama/bin/ollama"),
            local_ops.DEFAULT_OLLAMA,
        )
        self.assertEqual(
            [(slot.arm, slot.gpu, slot.port) for slot in local_ops.SLOTS],
            [
                ("typed_direct", 0, 11434),
                ("typed_schema_plan", 1, 11435),
                ("typed_execute_repair", 2, 11436),
                ("dspy_factorized", 3, 11437),
                ("trigger_consensus_executable", 4, 11438),
            ],
        )
        root = Path("/raid/session/aicontents/farm/experiments") / local_ops.RUN_ID
        self.assertEqual(
            local_ops.local_artifact_root(root),
            root / "local_granite" / local_ops.MODEL_DIGEST / "full",
        )
        self.assertEqual(
            local_ops.local_server_root(root),
            root / "local_granite" / local_ops.MODEL_DIGEST / "servers",
        )

    def test_experiment_commands_use_local_flags_full_digest_and_resume(self) -> None:
        root = Path("/raid/session/aicontents/farm/experiments") / local_ops.RUN_ID
        python = Path("/raid/session/aicontents/farm/.venv/bin/python")
        commands = [local_ops.experiment_launch_command(root, python, slot) for slot in local_ops.SLOTS]

        self.assertEqual(len({command[command.index("-s") + 1] for command in commands}), 5)
        for slot, command in zip(local_ops.SLOTS, commands, strict=True):
            self.assertIn("-d", command)
            self.assertEqual(command[command.index("--arm") + 1], slot.arm)
            self.assertEqual(command[command.index("--gpu") + 1], str(slot.gpu))
            self.assertEqual(command[command.index("--port") + 1], str(slot.port))
            self.assertEqual(command[command.index("--model") + 1], local_ops.MODEL)
            self.assertEqual(
                command[command.index("--expected-digest") + 1], local_ops.MODEL_DIGEST
            )
            self.assertIn("--resume", command)
            self.assertNotIn("--env-file", command)

    def test_server_commands_only_support_ports_11435_through_11438(self) -> None:
        root = Path("/raid/session/aicontents/farm/experiments") / local_ops.RUN_ID
        python = Path("/raid/session/aicontents/farm/.venv/bin/python")
        ollama = Path("/home/aicontents/.local/ollama/bin/ollama")
        models = local_ops.DEFAULT_MODELS
        commands = [
            local_ops.server_launch_command(root, python, ollama, models, slot)
            for slot in local_ops.SLOTS[1:]
        ]

        self.assertEqual(
            {command[command.index("--port") + 1] for command in commands},
            {"11435", "11436", "11437", "11438"},
        )
        self.assertTrue(all("11434" not in command for command in commands))

    def test_port_probe_requires_full_exact_digest_and_disables_proxies(self) -> None:
        with mock.patch.object(local_ops, "_tcp_open", return_value=True), mock.patch(
            "urllib.request.build_opener", return_value=FakeOpener(local_ops.MODEL_DIGEST)
        ) as opener:
            exact = local_ops.probe_ollama(11434)
        self.assertTrue(exact.ready)
        proxy_handler = opener.call_args.args[0]
        self.assertEqual(proxy_handler.proxies, {})

        with mock.patch.object(local_ops, "_tcp_open", return_value=True), mock.patch(
            "urllib.request.build_opener",
            return_value=FakeOpener(local_ops.MODEL_DIGEST[:12]),
        ):
            prefix_only = local_ops.probe_ollama(11434)
        self.assertFalse(prefix_only.ready)
        self.assertEqual(prefix_only.error, "digest_mismatch")

    def test_resume_requires_exact_immutable_local_launch_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_bound_run_root(temporary)
            slot = local_ops.SLOTS[0]
            local_ops.ensure_launch_binding(run_root, slot)
            log = local_ops.local_artifact_root(run_root) / "logs" / f"{slot.arm}.log"
            log.parent.mkdir(parents=True)
            log.write_text("partial\n", encoding="utf-8")

            with mock.patch.object(local_ops, "_tmux_has_session", return_value=False):
                self.assertTrue(local_ops.preflight_arm(run_root, slot))

            binding_path = (
                local_ops.local_artifact_root(run_root)
                / "state"
                / f"{slot.arm}.launch.json"
            )
            binding = json.loads(binding_path.read_text())
            binding["port"] = 11437
            binding_path.write_text(json.dumps(binding), encoding="utf-8")
            with mock.patch.object(local_ops, "_tmux_has_session", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "identity changed"):
                    local_ops.preflight_arm(run_root, slot)

    def test_launch_binding_pins_all_code_matrix_and_executed_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_bound_run_root(temporary)
            binding = local_ops.local_launch_binding(run_root, local_ops.SLOTS[0])
            self.assertEqual(
                set(local_ops.PROVENANCE_FILES), set(binding["artifacts_sha256"])
            )
            self.assertIn("scripts/frozen_router.py", binding["artifacts_sha256"])
            self.assertEqual(
                "farm_round8_local_launch_binding_v3", binding["schema_version"]
            )
            self.assertTrue(
                all(
                    len(digest) == 64
                    for digest in binding["artifacts_sha256"].values()
                )
            )
            self.assertEqual(
                {
                    "executed_rows": local_ops.FULL_SCREEN_ROWS,
                    "executed_group_ids_sha256": local_ops.FULL_SCREEN_IDS_SHA256,
                },
                binding["selection"],
            )

            local_ops.ensure_launch_binding(run_root, local_ops.SLOTS[0])
            (run_root / "scripts" / "executable_applet.py").write_text(
                "changed\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "binding changed"):
                with mock.patch.object(local_ops, "_tmux_has_session", return_value=False):
                    local_ops.ensure_launch_binding(run_root, local_ops.SLOTS[0])

    def test_runner_resume_binding_pins_code_transport_and_screen(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_bound_run_root(temporary)
            slot = local_ops.SLOTS[1]
            binding = self.make_runner_binding(run_root, slot)
            self.assertEqual(
                (), local_ops.runner_binding_mismatches(run_root, slot, binding)
            )

            binding["selection"]["executed_rows"] = 1
            binding["protocol"]["data_left_dgx"] = True
            self.assertEqual(
                ("protocol.data_left_dgx", "selection.executed_rows"),
                local_ops.runner_binding_mismatches(run_root, slot, binding),
            )

    def test_consensus_runner_binding_authenticates_frozen_router(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_bound_run_root(temporary)
            slot = local_ops.SLOTS[-1]
            binding = self.make_runner_binding(run_root, slot)
            self.assertEqual(
                (), local_ops.runner_binding_mismatches(run_root, slot, binding)
            )

            binding["trigger_consensus_policy"]["router_runtime_sha256"] = "0" * 64
            self.assertEqual(
                ("trigger_consensus_policy.router_runtime_sha256",),
                local_ops.runner_binding_mismatches(run_root, slot, binding),
            )

    def test_per_arm_smoke_gate_uses_two_rows_and_consensus_uses_five(self) -> None:
        root = Path("/raid/session/aicontents/farm/experiments") / local_ops.RUN_ID
        python = Path("/raid/session/aicontents/farm/.venv/bin/python")
        smoke_sizes = {
            slot.arm: local_ops.smoke_size_for_arm(slot.arm, local_ops.SMOKE_ROWS)
            for slot in local_ops.SLOTS
        }
        self.assertEqual(local_ops.SMOKE_ROWS, smoke_sizes["typed_direct"])
        self.assertEqual(
            local_ops.CONSENSUS_SMOKE_ROWS,
            smoke_sizes["trigger_consensus_executable"],
        )
        for slot in local_ops.SLOTS:
            smoke = smoke_sizes[slot.arm]
            assert smoke is not None
            command = local_ops.experiment_launch_command(
                root, python, slot, smoke=smoke
            )
            self.assertEqual(str(smoke), command[command.index("--smoke") + 1])
            self.assertIn(f"smoke-{smoke}", command[command.index("-s") + 1])

    def test_completed_smoke_gate_authenticates_binding_hash_and_record_ids(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_bound_run_root(temporary)
            slot = local_ops.SLOTS[0]
            smoke = local_ops.SMOKE_ROWS
            artifacts = local_ops.local_artifact_root(run_root, smoke)
            for name in ("results", "progress", "records"):
                (artifacts / name).mkdir(parents=True, exist_ok=True)
            local_ops.ensure_launch_binding(run_root, slot, smoke=smoke)
            binding = self.make_runner_binding(run_root, slot, smoke=smoke)
            group_ids = ["q_2d91e68dc1e30567af6f", "q_5359a78978595d4b5cf7"]
            records_path = artifacts / "records" / f"{slot.arm}.jsonl"
            records_path.write_text(
                "".join(
                    json.dumps(
                        self.smoke_record(slot, group_id)
                    )
                    + "\n"
                    for group_id in group_ids
                ),
                encoding="utf-8",
            )
            result_path = artifacts / "results" / f"{slot.arm}.json"
            result_path.write_text(
                json.dumps(
                    {
                        "status": "completed",
                        "binding": binding,
                        "metrics": {"rows": smoke},
                    }
                ),
                encoding="utf-8",
            )
            import hashlib

            (artifacts / "progress" / f"{slot.arm}.json").write_text(
                json.dumps(
                    {
                        "phase": "complete",
                        "binding": binding,
                        "completed_rows": smoke,
                        "target_rows": smoke,
                        "output": str(result_path),
                        "output_sha256": hashlib.sha256(
                            result_path.read_bytes()
                        ).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            validated = local_ops.validate_completed_artifacts(
                run_root, slot, smoke=smoke
            )
            self.assertEqual("completed", validated["status"])

            healthy_records = records_path.read_text(encoding="utf-8")
            fallback_records = []
            for line in healthy_records.splitlines():
                record = json.loads(line)
                record.update(
                    {
                        "terminal_status": "safe_fallback",
                        "fallback_used": True,
                        "program": None,
                    }
                )
                record["execution"].update(
                    {
                        "compilation_attempts": 0,
                        "compiled": False,
                        "sandbox_attempts": 0,
                        "sandbox_run": False,
                        "receipt_sha256": None,
                    }
                )
                fallback_records.append(json.dumps(record) + "\n")
            records_path.write_text("".join(fallback_records), encoding="utf-8")
            validated_fallback = local_ops.validate_completed_artifacts(
                run_root, slot, smoke=smoke
            )
            self.assertEqual("completed", validated_fallback["status"])

            records_path.write_text(
                healthy_records.replace(group_ids[1], "x"), encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "selection changed"):
                local_ops.validate_completed_artifacts(run_root, slot, smoke=smoke)

    def test_smoke_semantics_require_native_response_and_execution(self) -> None:
        slot = local_ops.SLOTS[0]
        group_ids = ("row-1", "row-2")
        failed_calls = [
            self.smoke_record(
                slot, group_id, valid_native=False, executed=False
            )
            for group_id in group_ids
        ]
        with self.assertRaisesRegex(RuntimeError, "no complete valid native protocol chain"):
            local_ops.validate_smoke_semantics(slot, failed_calls)

        fallback_only = [
            self.smoke_record(slot, group_id, executed=False)
            for group_id in group_ids
        ]
        # A valid conservative rejection is an experiment outcome, not a smoke
        # infrastructure failure.  The full run must be allowed to measure it.
        local_ops.validate_smoke_semantics(slot, fallback_only)

        invalid_program = self.smoke_record(slot, group_ids[0])
        invalid_program["execution"]["sandbox_run"] = False
        with self.assertRaisesRegex(RuntimeError, "unauthenticated executable program"):
            local_ops.validate_smoke_semantics(slot, [invalid_program, fallback_only[1]])

        successful = [
            self.smoke_record(slot, group_ids[0]),
            self.smoke_record(slot, group_ids[1], executed=False),
        ]
        local_ops.validate_smoke_semantics(slot, successful)

    def test_consensus_smoke_must_route_call_and_execute(self) -> None:
        slot = local_ops.SLOTS[-1]
        records = [
            self.smoke_record(
                slot,
                f"row-{index}",
                executed=False,
                routed=False,
                call_count=0,
            )
            for index in range(local_ops.CONSENSUS_SMOKE_ROWS)
        ]
        with self.assertRaisesRegex(RuntimeError, "did not exercise a routed case"):
            local_ops.validate_smoke_semantics(slot, records)

        records[0] = self.smoke_record(
            slot, "row-0", executed=False, routed=True, call_count=0
        )
        with self.assertRaisesRegex(RuntimeError, "made no native tool call"):
            local_ops.validate_smoke_semantics(slot, records)

        records[0] = self.smoke_record(slot, "row-0", executed=False, routed=True, call_count=1)
        with self.assertRaisesRegex(RuntimeError, "valid two-view call chain"):
            local_ops.validate_smoke_semantics(slot, records)

        records[0] = self.smoke_record(
            slot, "row-0", executed=False, routed=True, call_count=2
        )
        local_ops.validate_smoke_semantics(slot, records)

    def test_full_launcher_checks_every_smoke_gate_before_any_server_probe(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / local_ops.RUN_ID
            probe = mock.Mock()
            with mock.patch.object(local_ops, "validate_runtime"), mock.patch.object(
                local_ops, "validate_completed_artifacts",
                side_effect=RuntimeError("completed artifacts are absent"),
            ), mock.patch.object(local_ops, "probe_ollama", probe), mock.patch.object(
                local_ops, "_tmux_has_session", return_value=False
            ):
                with self.assertRaisesRegex(RuntimeError, "artifacts are absent"):
                    local_ops.main(
                        [
                            "--run-root",
                            str(run_root),
                            "--python",
                            "/bin/true",
                            "--ollama-bin",
                            "/bin/true",
                            "--ollama-models",
                            str(run_root),
                            "--dry-run",
                        ]
                    )
            probe.assert_not_called()
            self.assertFalse(local_ops.local_model_root(run_root).exists())

    def test_smoke_dry_run_builds_mixed_two_and_five_row_commands_read_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / local_ops.RUN_ID
            ready = local_ops.OllamaProbe(
                port=11434,
                tcp_open=True,
                api_ok=True,
                model_present=True,
                digest_ok=True,
                observed_digest=local_ops.MODEL_DIGEST,
                error=None,
            )
            output = io.StringIO()
            with mock.patch.object(local_ops, "validate_runtime"), mock.patch.object(
                local_ops, "preflight_arm", return_value=False
            ), mock.patch.object(local_ops, "probe_ollama", return_value=ready), mock.patch.object(
                local_ops, "validate_managed_server"
            ), mock.patch.object(local_ops, "query_gpu_processes", return_value={}), mock.patch.object(
                local_ops, "_tmux_has_session", return_value=False
            ), contextlib.redirect_stdout(output):
                status = local_ops.main(
                    [
                        "--run-root",
                        str(run_root),
                        "--python",
                        "/bin/true",
                        "--ollama-bin",
                        "/bin/true",
                        "--ollama-models",
                        str(run_root),
                        "--smoke",
                        "2",
                        "--dry-run",
                    ]
                )
            self.assertEqual(0, status)
            commands = [
                line
                for line in output.getvalue().splitlines()
                if line.startswith("DRY_RUN_EXPERIMENT")
            ]
            self.assertEqual(5, len(commands))
            self.assertEqual(4, sum("--smoke 2" in line for line in commands))
            self.assertEqual(1, sum("--smoke 5" in line for line in commands))
            self.assertFalse(local_ops.local_model_root(run_root).exists())

    def test_gpu_isolation_maps_worker_descendants_one_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            listeners = {
                slot.port: 2000 + slot.gpu for slot in local_ops.SLOTS
            }
            server_root = local_ops.local_server_root(run_root)
            server_root.mkdir(parents=True)
            for slot in local_ops.SLOTS[1:]:
                (server_root / f"{slot.port}.pid").write_text(
                    f"{listeners[slot.port]}\n", encoding="utf-8"
                )
            apps = {
                slot.gpu: (
                    {
                        "pid": 3000 + slot.gpu,
                        "process_name": "ollama_llama_server",
                        "used_memory_mib": 1024,
                    },
                )
                for slot in local_ops.SLOTS
            }

            def descends(pid: int, ancestor: int) -> bool:
                return pid - 3000 == ancestor - 2000

            with mock.patch.object(
                local_ops, "listening_pid", side_effect=lambda port: listeners[port]
            ), mock.patch.object(
                local_ops, "query_gpu_compute_apps", return_value=apps
            ), mock.patch.object(local_ops, "_pid_descends_from", side_effect=descends):
                isolation = local_ops.gpu_isolation_snapshot(run_root)
            self.assertTrue(isolation["verified"], isolation["errors"])
            self.assertEqual([], isolation["unexpected_compute_apps"])

    def test_partial_artifacts_without_binding_are_not_resumed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = self.make_bound_run_root(temporary)
            slot = local_ops.SLOTS[0]
            log = local_ops.local_artifact_root(run_root) / "logs" / f"{slot.arm}.log"
            log.parent.mkdir(parents=True)
            log.write_text("unknown old process\n", encoding="utf-8")

            with mock.patch.object(local_ops, "_tmux_has_session", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "unsafe resume"):
                    local_ops.preflight_arm(run_root, slot)

    def test_job_wrappers_reject_crossed_gpu_port_arm_mapping(self) -> None:
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                experiment_job.parse_args(
                    [
                        "--run-root",
                        "/tmp/run",
                        "--python",
                        "/usr/bin/python3",
                        "--arm",
                        "typed_direct",
                        "--gpu",
                        "1",
                        "--port",
                        "11435",
                        "--model",
                        local_ops.MODEL,
                        "--expected-digest",
                        local_ops.MODEL_DIGEST,
                        "--resume",
                    ]
                )
            with self.assertRaises(SystemExit):
                server_job.parse_args(
                    [
                        "--run-root",
                        "/tmp/run",
                        "--ollama-bin",
                        "/bin/true",
                        "--ollama-models",
                        "/tmp/models",
                        "--port",
                        "11435",
                        "--gpu",
                        "2",
                    ]
                )
            consensus = [
                "--run-root",
                "/tmp/run",
                "--python",
                "/usr/bin/python3",
                "--arm",
                "trigger_consensus_executable",
                "--gpu",
                "4",
                "--port",
                "11438",
                "--model",
                local_ops.MODEL,
                "--expected-digest",
                local_ops.MODEL_DIGEST,
                "--resume",
                "--smoke",
            ]
            with self.assertRaises(SystemExit):
                experiment_job.parse_args([*consensus, "2"])
            parsed = experiment_job.parse_args([*consensus, "5"])
            self.assertEqual(5, parsed.smoke)

    def test_local_job_arm_lock_is_nonblocking_and_inherited(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            lock_path = Path(temporary) / "state" / "typed_direct.run.lock"
            descriptor = experiment_job._acquire_arm_lock(lock_path)
            try:
                self.assertTrue(Path(lock_path).is_file())
                self.assertTrue(os.get_inheritable(descriptor))
                with self.assertRaisesRegex(RuntimeError, "another process owns"):
                    experiment_job._acquire_arm_lock(lock_path)
            finally:
                os.close(descriptor)

    def test_local_two_sample_verification_requires_all_five_exact_servers(self) -> None:
        def arm(slot: local_ops.Slot) -> dict:
            return {
                "arm": slot.arm,
                "status": "running",
                "tmux_session": True,
                "process_alive": True,
                "process_matches_runner": True,
                "completed_cases": 1,
                "record_lines": 1,
                "log_bytes": 10,
            }

        def server(slot: local_ops.Slot) -> dict:
            return {
                "port": slot.port,
                "status": "ready",
                "observed_digest": local_ops.MODEL_DIGEST,
            }

        snapshot = {
            "arms": [arm(slot) for slot in local_ops.SLOTS],
            "servers": [server(slot) for slot in local_ops.SLOTS],
            "gpu_isolation": {"verified": True, "conflicts": []},
        }
        ok, errors = local_verify.assess((snapshot, snapshot), "running")
        self.assertTrue(ok, errors)

        bad = json.loads(json.dumps(snapshot))
        bad["servers"][2]["observed_digest"] = local_ops.MODEL_DIGEST[:12]
        ok, errors = local_verify.assess((snapshot, bad), "running")
        self.assertFalse(ok)
        self.assertTrue(any("digest changed" in error for error in errors))

        conflict = json.loads(json.dumps(snapshot))
        conflict["gpu_isolation"] = {
            "verified": False,
            "conflicts": ["port 11434 observed GPU 1"],
        }
        ok, errors = local_verify.assess((snapshot, conflict), "running")
        self.assertFalse(ok)
        self.assertTrue(any("GPU isolation conflict" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
