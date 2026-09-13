#!/usr/bin/env python3
"""Exec one GPU-isolated, loopback-only Ollama daemon on ports 11435--11438."""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


RUN_ID = "20260904T074517Z-farm-round8-executable-agent"
DIGEST = "2f8a7367d4416b508336f6170c295e8465ff004731ae563631196781299f750d"


def _read_live_pid(path: Path) -> int | None:
    try:
        pid = int(path.read_text(encoding="utf-8").strip())
        if pid <= 1:
            return None
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, OSError, ValueError):
        return None


def _atomic_pid(path: Path) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        os.write(descriptor, f"{os.getpid()}\n".encode("ascii"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--ollama-bin", type=Path, required=True)
    parser.add_argument("--ollama-models", type=Path, required=True)
    parser.add_argument(
        "--port", type=int, choices=(11435, 11436, 11437, 11438), required=True
    )
    parser.add_argument("--gpu", type=int, choices=(1, 2, 3, 4), required=True)
    args = parser.parse_args(argv)
    if args.gpu != args.port - 11434:
        parser.error("the frozen local server mapping is port 11434 + physical GPU index")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_root = args.run_root.resolve()
    ollama_bin = args.ollama_bin.resolve()
    models = args.ollama_models.resolve()
    if run_root.name != RUN_ID:
        raise RuntimeError(f"unexpected Round8 run root: {run_root}")
    if not ollama_bin.is_file() or not os.access(ollama_bin, os.X_OK):
        raise RuntimeError(f"Ollama executable is unavailable: {ollama_bin}")
    if not models.is_dir():
        raise RuntimeError(f"OLLAMA_MODELS is unavailable: {models}")
    server_root = run_root / "local_granite" / DIGEST / "servers"
    server_root.mkdir(parents=True, exist_ok=True)
    pid_path = server_root / f"{args.port}.pid"
    live_pid = _read_live_pid(pid_path)
    if live_pid is not None and live_pid != os.getpid():
        raise RuntimeError(f"server pid already owns port {args.port}: pid={live_pid}")

    os.umask(0o077)
    _atomic_pid(pid_path)
    log_path = server_root / f"{args.port}.log"
    descriptor = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(descriptor, sys.stdout.fileno())
    os.dup2(descriptor, sys.stderr.fileno())
    if descriptor not in {sys.stdout.fileno(), sys.stderr.fileno()}:
        os.close(descriptor)
    print(
        f"START_OLLAMA {datetime.now(timezone.utc).isoformat()} port={args.port} "
        f"gpu={args.gpu} pid={os.getpid()}",
        flush=True,
    )

    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu),
            "OLLAMA_HOST": f"127.0.0.1:{args.port}",
            "OLLAMA_MODELS": str(models),
            "OLLAMA_KEEP_ALIVE": "-1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
            "OLLAMA_NUM_PARALLEL": "1",
            "NO_PROXY": "127.0.0.1,localhost",
            "no_proxy": "127.0.0.1,localhost",
        }
    )
    os.execve(str(ollama_bin), [str(ollama_bin), "serve"], environment)
    return 127


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"ROUND8_LOCAL_OLLAMA_REFUSED: {error}", file=sys.stderr, flush=True)
        raise SystemExit(2) from error
