from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import launch_round8  # noqa: E402
import sync_round8_code  # noqa: E402
import verify_round8  # noqa: E402


class Round8OpsTests(unittest.TestCase):
    def test_launcher_builds_four_distinct_detached_tmux_commands(self) -> None:
        root = Path("/raid/session/aicontents/farm/experiments") / launch_round8.RUN_ID
        env_file = Path("/raid/session/aicontents/farm/.env")
        python = Path("/raid/session/aicontents/farm/.venv/bin/python")
        commands = [
            launch_round8.tmux_launch_command(
                run_root=root,
                env_file=env_file,
                python=python,
                arm=arm,
                slot=slot,
                smoke=None,
            )
            for slot, arm in enumerate(launch_round8.ARMS)
        ]

        sessions = {command[command.index("-s") + 1] for command in commands}
        slots = {command[command.index("--slot") + 1] for command in commands}
        self.assertEqual(len(sessions), 4)
        self.assertEqual(slots, {"0", "1", "2", "3"})
        for arm, command in zip(launch_round8.ARMS, commands, strict=True):
            self.assertIn("new-session", command)
            self.assertIn("-d", command)
            self.assertEqual(command[command.index("--arm") + 1], arm)
            self.assertIn("--resume", command)
            self.assertNotIn(".env=", " ".join(command))

    def test_preflight_allows_stale_partial_but_refuses_active_or_complete(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "logs").mkdir()
            (root / "logs" / "typed_direct.log").write_text("partial\n", encoding="utf-8")
            with mock.patch.object(launch_round8, "_tmux_has_session", return_value=False):
                state = launch_round8.preflight_arm(root, "typed_direct")
            self.assertTrue(state.resumable_partial)

            (root / "results").mkdir()
            (root / "results" / "typed_direct.json").write_text("{}\n", encoding="utf-8")
            with mock.patch.object(launch_round8, "_tmux_has_session", return_value=False):
                with self.assertRaisesRegex(RuntimeError, "completed result"):
                    launch_round8.preflight_arm(root, "typed_direct")

            (root / "results" / "typed_direct.json").unlink()
            with mock.patch.object(launch_round8, "_tmux_has_session", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "active tmux"):
                    launch_round8.preflight_arm(root, "typed_direct")

    def test_verifier_observes_session_process_and_progress_without_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for directory in ("pids", "progress", "records", "logs"):
                (root / directory).mkdir()
            arm = "typed_direct"
            (root / "pids" / f"{arm}.pid").write_text("1234\n", encoding="utf-8")
            (root / "progress" / f"{arm}.json").write_text(
                json.dumps({"status": "running", "completed_cases": 7, "total_cases": 231}),
                encoding="utf-8",
            )
            (root / "records" / f"{arm}.jsonl").write_text("{}\n{}\n", encoding="utf-8")
            (root / "logs" / f"{arm}.log").write_text("START\n", encoding="utf-8")

            observed = verify_round8.inspect_arm(
                root,
                arm,
                tmux_probe=lambda _: True,
                process_probe=lambda pid, run_root, selected_arm: (
                    pid == 1234,
                    run_root == root and selected_arm == arm,
                    "run_round8.py",
                ),
            )

            self.assertEqual(observed["status"], "running")
            self.assertEqual(observed["completed_cases"], 7)
            self.assertEqual(observed["record_lines"], 2)
            self.assertTrue((root / "pids" / f"{arm}.pid").exists())

    def test_two_sample_verification_accepts_running_to_complete(self) -> None:
        def arm_state(arm: str, status: str, count: int) -> dict:
            running = status == "running"
            return {
                "arm": arm,
                "status": status,
                "tmux_session": running,
                "process_alive": running,
                "process_matches_runner": running,
                "completed_cases": count,
                "record_lines": count,
                "log_bytes": count + 100,
            }

        first = {"arms": [arm_state(arm, "running", 10) for arm in verify_round8.ARMS]}
        second = {"arms": [arm_state(arm, "complete", 231) for arm in verify_round8.ARMS]}

        ok, errors = verify_round8.assess_snapshots(
            (first, second), "running-or-complete"
        )

        self.assertTrue(ok, errors)
        self.assertEqual(errors, ())

    def test_two_sample_verification_rejects_progress_regression(self) -> None:
        def snapshot(count: int) -> dict:
            return {
                "arms": [
                    {
                        "arm": arm,
                        "status": "running",
                        "tmux_session": True,
                        "process_alive": True,
                        "process_matches_runner": True,
                        "completed_cases": count,
                        "record_lines": count,
                        "log_bytes": 100,
                    }
                    for arm in verify_round8.ARMS
                ]
            }

        ok, errors = verify_round8.assess_snapshots(
            (snapshot(10), snapshot(9)), "running"
        )

        self.assertFalse(ok)
        self.assertTrue(any("regressed" in error for error in errors))

    def test_two_sample_verification_requires_progress_or_log_evidence(self) -> None:
        empty_running = {
            "arms": [
                {
                    "arm": arm,
                    "status": "running",
                    "tmux_session": True,
                    "process_alive": True,
                    "process_matches_runner": True,
                    "completed_cases": None,
                    "record_lines": 0,
                    "log_bytes": 0,
                }
                for arm in verify_round8.ARMS
            ]
        }

        ok, errors = verify_round8.assess_snapshots(
            (empty_running, empty_running), "running"
        )

        self.assertFalse(ok)
        self.assertTrue(any("no progress/log evidence" in error for error in errors))

    def test_sync_allowlist_excludes_secrets_dependencies_and_results(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary)
            for relative, content in (
                ("ROUND8_SCOPE.md", "scope"),
                ("scripts/tool.py", "pass"),
                ("tests/test_tool.py", "pass"),
                (".env", "secret"),
                (".deps/dspy/__init__.py", "version"),
                ("results/typed_direct.json", "{}"),
                ("logs/typed_direct.log", "running"),
                ("local_granite/digest/full/results/typed_direct.json", "{}"),
            ):
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(content, encoding="utf-8")

            selected = {path.as_posix() for path in sync_round8_code.selected_files(source)}

            self.assertEqual(
                selected,
                {"ROUND8_SCOPE.md", "scripts/tool.py", "tests/test_tool.py"},
            )

    def test_sync_refuses_cloud_local_and_local_server_sessions(self) -> None:
        expected = {
            "farm-r8-typed-direct",
            "farm-r8-local-typed-direct",
            "farm-r8-ollama-11435",
        }

        def probe(command, **_kwargs):
            name = command[command.index("-t") + 1].removeprefix("=")
            return mock.Mock(returncode=0 if name in expected else 1)

        with mock.patch.object(sync_round8_code.subprocess, "run", side_effect=probe):
            self.assertEqual(expected, set(sync_round8_code.active_sessions()))

    def test_sync_detects_live_local_process_without_tmux(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pid_path = (
                root
                / "local_granite"
                / ("a" * 64)
                / "smoke-5"
                / "pids"
                / "trigger_consensus_executable.pid"
            )
            pid_path.parent.mkdir(parents=True)
            pid_path.write_text(f"{os.getpid()}\n", encoding="utf-8")
            active = sync_round8_code.active_runtime_pids(root)
            self.assertEqual(((pid_path.relative_to(root), os.getpid()),), active)


if __name__ == "__main__":
    unittest.main()
