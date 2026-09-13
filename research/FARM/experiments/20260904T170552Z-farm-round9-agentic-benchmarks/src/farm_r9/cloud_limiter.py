"""Cross-process admission control for Ollama Cloud HTTP requests.

Every process in an experiment must use the same ``lock_directory``.  The
kernel-held ``flock`` locks, rather than the diagnostic file contents, are the
source of truth.  A process crash closes its descriptors and therefore frees
its lease automatically.
"""

from __future__ import annotations

import fcntl
import json
import math
import os
import socket
import time
import uuid
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterator, Optional, TypeVar, Union


MAX_OLLAMA_CLOUD_CONCURRENCY = 3

PathLike = Union[str, os.PathLike[str]]
ResultT = TypeVar("ResultT")


class LimiterTimeoutError(TimeoutError):
    """Raised when no Cloud request slot becomes available before timeout."""

    def __init__(self, *, timeout: float, waited_seconds: float) -> None:
        self.timeout = timeout
        self.waited_seconds = waited_seconds
        super().__init__(
            "timed out waiting for an Ollama Cloud request slot "
            f"after {waited_seconds:.3f}s (timeout={timeout:.3f}s)"
        )


@dataclass(frozen=True)
class LeaseMetadata:
    """Generated, credential-free fields suitable for structured logs."""

    lease_id: str
    slot: int
    max_concurrent: int
    pid: int
    hostname: str
    acquired_at_utc: str
    waited_seconds: float

    def to_log_dict(self) -> dict[str, object]:
        """Return only the fixed, generated metadata allowlist."""

        return asdict(self)


class OllamaCloudLimiter:
    """Limit Ollama Cloud requests to at most three across local processes.

    The lock directory is part of the experiment's launch contract: all jobs
    sharing an Ollama account must receive the same directory.
    """

    def __init__(
        self,
        lock_directory: PathLike,
        *,
        max_concurrent: int = MAX_OLLAMA_CLOUD_CONCURRENCY,
        poll_interval: float = 0.025,
    ) -> None:
        if isinstance(max_concurrent, bool) or not isinstance(max_concurrent, int):
            raise TypeError("max_concurrent must be an integer")
        if not 1 <= max_concurrent <= MAX_OLLAMA_CLOUD_CONCURRENCY:
            raise ValueError(
                "max_concurrent must be between 1 and "
                f"{MAX_OLLAMA_CLOUD_CONCURRENCY}"
            )
        if not math.isfinite(poll_interval) or poll_interval <= 0:
            raise ValueError("poll_interval must be a positive finite number")

        self.lock_directory = Path(lock_directory).expanduser().resolve()
        self.max_concurrent = max_concurrent
        self.poll_interval = float(poll_interval)
        self.lock_directory.mkdir(mode=0o700, parents=True, exist_ok=True)

    @contextmanager
    def acquire(self, *, timeout: Optional[float] = None) -> Iterator[LeaseMetadata]:
        """Acquire one request lease and release it on every exit path."""

        timeout = self._validate_timeout(timeout)
        started = time.monotonic()
        deadline = None if timeout is None else started + timeout
        first_slot = os.getpid() % self.max_concurrent

        while True:
            for offset in range(self.max_concurrent):
                slot = (first_slot + offset) % self.max_concurrent
                descriptor = self._open_slot(slot)
                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    os.close(descriptor)
                    continue
                except BaseException:
                    os.close(descriptor)
                    raise

                metadata = LeaseMetadata(
                    lease_id=uuid.uuid4().hex,
                    slot=slot,
                    max_concurrent=self.max_concurrent,
                    pid=os.getpid(),
                    hostname=socket.gethostname(),
                    acquired_at_utc=self._utc_now(),
                    waited_seconds=max(0.0, time.monotonic() - started),
                )
                try:
                    self._write_slot(descriptor, metadata, state="held")
                except BaseException:
                    self._unlock_and_close(descriptor)
                    raise

                try:
                    yield metadata
                finally:
                    try:
                        self._write_slot(descriptor, metadata, state="released")
                    finally:
                        self._unlock_and_close(descriptor)
                return

            now = time.monotonic()
            if deadline is not None and now >= deadline:
                raise LimiterTimeoutError(
                    timeout=timeout,
                    waited_seconds=max(0.0, now - started),
                )

            sleep_for = self.poll_interval
            if deadline is not None:
                sleep_for = min(sleep_for, max(0.0, deadline - now))
            if sleep_for > 0:
                time.sleep(sleep_for)

    def request(
        self,
        send: Callable[[], ResultT],
        *,
        timeout: Optional[float] = None,
    ) -> ResultT:
        """Run exactly one complete HTTP request while holding one lease.

        ``send`` should be a zero-argument closure that performs the transport
        call.  Keeping retries outside this method ensures each physical HTTP
        attempt obtains its own lease and is independently accounted for.
        """

        if not callable(send):
            raise TypeError("send must be callable")
        with self.acquire(timeout=timeout):
            return send()

    def _open_slot(self, slot: int) -> int:
        path = self.lock_directory / f"ollama-cloud-{slot}.slot"
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        descriptor = os.open(path, flags, 0o600)
        try:
            os.fchmod(descriptor, 0o600)
        except BaseException:
            os.close(descriptor)
            raise
        return descriptor

    @staticmethod
    def _validate_timeout(timeout: Optional[float]) -> Optional[float]:
        if timeout is None:
            return None
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
            raise TypeError("timeout must be a number or None")
        timeout = float(timeout)
        if not math.isfinite(timeout) or timeout < 0:
            raise ValueError("timeout must be a non-negative finite number")
        return timeout

    @staticmethod
    def _write_slot(descriptor: int, metadata: LeaseMetadata, *, state: str) -> None:
        payload = metadata.to_log_dict()
        payload["state"] = state
        if state == "released":
            payload["released_at_utc"] = OllamaCloudLimiter._utc_now()
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        os.lseek(descriptor, 0, os.SEEK_SET)
        os.ftruncate(descriptor, 0)
        os.write(descriptor, encoded)

    @staticmethod
    def _unlock_and_close(descriptor: int) -> None:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)

    @staticmethod
    def _utc_now() -> str:
        return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


__all__ = [
    "LeaseMetadata",
    "LimiterTimeoutError",
    "MAX_OLLAMA_CLOUD_CONCURRENCY",
    "OllamaCloudLimiter",
]
