from __future__ import annotations

import multiprocessing
import sys
import tempfile
import time
import unittest
from pathlib import Path


ROUND9_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND9_ROOT / "src"))

from farm_r9.cloud_limiter import (
    MAX_OLLAMA_CLOUD_CONCURRENCY,
    LimiterTimeoutError,
    OllamaCloudLimiter,
)


def _stress_worker(
    lock_directory: str,
    start: multiprocessing.synchronize.Event,
    active: multiprocessing.sharedctypes.Synchronized,
    peak: multiprocessing.sharedctypes.Synchronized,
    guard: multiprocessing.synchronize.Lock,
) -> None:
    limiter = OllamaCloudLimiter(
        lock_directory,
        max_concurrent=3,
        poll_interval=0.002,
    )
    if not start.wait(timeout=10):
        raise RuntimeError("stress-test start gate timed out")

    def simulated_http_request() -> str:
        with guard:
            active.value += 1
            peak.value = max(peak.value, active.value)
        try:
            time.sleep(0.08)
            return "ok"
        finally:
            with guard:
                active.value -= 1

    assert limiter.request(simulated_http_request, timeout=10) == "ok"


def _failing_request_worker(
    lock_directory: str,
    entered_request: multiprocessing.synchronize.Event,
) -> None:
    limiter = OllamaCloudLimiter(
        lock_directory,
        max_concurrent=1,
        poll_interval=0.001,
    )

    def failing_request() -> None:
        entered_request.set()
        time.sleep(0.08)
        raise RuntimeError("deliberate transport failure")

    try:
        limiter.request(failing_request, timeout=1)
    except RuntimeError as error:
        if str(error) == "deliberate transport failure":
            return
        raise
    raise AssertionError("the simulated request unexpectedly succeeded")


class OllamaCloudLimiterTests(unittest.TestCase):
    def test_requests_across_processes_never_exceed_three(self) -> None:
        context = multiprocessing.get_context("spawn")
        start = context.Event()
        active = context.Value("i", 0)
        peak = context.Value("i", 0)
        guard = context.Lock()

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_directory = str(Path(temporary_directory) / "leases")
            processes = [
                context.Process(
                    target=_stress_worker,
                    args=(lock_directory, start, active, peak, guard),
                )
                for _ in range(9)
            ]

            for process in processes:
                process.start()
            start.set()

            for process in processes:
                process.join(timeout=15)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
                    self.fail(f"worker {process.pid} did not finish")
                self.assertEqual(process.exitcode, 0)

        self.assertEqual(active.value, 0)
        # Reaching three proves the test generated contention; never exceeding
        # it proves the process-global admission boundary.
        self.assertEqual(peak.value, 3)

    def test_request_exception_releases_the_slot_for_another_process(self) -> None:
        context = multiprocessing.get_context("spawn")
        entered_request = context.Event()

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_directory = str(Path(temporary_directory) / "leases")
            failing_process = context.Process(
                target=_failing_request_worker,
                args=(lock_directory, entered_request),
            )
            failing_process.start()
            self.assertTrue(entered_request.wait(timeout=5))

            contender = OllamaCloudLimiter(
                lock_directory,
                max_concurrent=1,
                poll_interval=0.001,
            )
            started = time.monotonic()
            self.assertEqual(
                contender.request(lambda: "recovered", timeout=1),
                "recovered",
            )
            waited = time.monotonic() - started

            failing_process.join(timeout=5)
            if failing_process.is_alive():
                failing_process.terminate()
                failing_process.join(timeout=5)
                self.fail("failing request worker did not finish")

        self.assertEqual(failing_process.exitcode, 0)
        self.assertGreaterEqual(waited, 0.04)

    def test_acquire_times_out_when_all_slots_are_held(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_directory = Path(temporary_directory) / "leases"
            holder = OllamaCloudLimiter(
                lock_directory,
                max_concurrent=1,
                poll_interval=0.002,
            )
            contender = OllamaCloudLimiter(
                lock_directory,
                max_concurrent=1,
                poll_interval=0.002,
            )

            with holder.acquire(timeout=0.2):
                started = time.monotonic()
                with self.assertRaises(LimiterTimeoutError) as caught:
                    with contender.acquire(timeout=0.05):
                        self.fail("a fourth-style contender acquired a held slot")
                elapsed = time.monotonic() - started

            self.assertGreaterEqual(elapsed, 0.04)
            self.assertLess(elapsed, 0.5)
            self.assertEqual(caught.exception.timeout, 0.05)
            self.assertGreaterEqual(caught.exception.waited_seconds, 0.04)

    def test_metadata_is_allowlisted_and_request_data_is_not_persisted(self) -> None:
        secret_sentinel = "Authorization: Bearer should-never-be-in-a-slot-file"
        expected_keys = {
            "lease_id",
            "slot",
            "max_concurrent",
            "pid",
            "hostname",
            "acquired_at_utc",
            "waited_seconds",
        }

        with tempfile.TemporaryDirectory() as temporary_directory:
            lock_directory = Path(temporary_directory) / "leases"
            limiter = OllamaCloudLimiter(lock_directory)
            with limiter.acquire(timeout=0.2) as metadata:
                self.assertEqual(set(metadata.to_log_dict()), expected_keys)
                self.assertEqual(metadata.max_concurrent, 3)
                self.assertIn(metadata.slot, (0, 1, 2))
                self.assertEqual(metadata.pid, multiprocessing.current_process().pid)
                self.assertTrue(metadata.lease_id)

            self.assertEqual(
                limiter.request(lambda: secret_sentinel, timeout=0.2),
                secret_sentinel,
            )
            persisted = "".join(
                path.read_text(encoding="utf-8")
                for path in sorted(lock_directory.glob("*.slot"))
            )

        self.assertNotIn(secret_sentinel, persisted)

    def test_capacity_cannot_exceed_cloud_plan_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaisesRegex(ValueError, "between 1 and 3"):
                OllamaCloudLimiter(
                    temporary_directory,
                    max_concurrent=MAX_OLLAMA_CLOUD_CONCURRENCY + 1,
                )


if __name__ == "__main__":
    unittest.main()
