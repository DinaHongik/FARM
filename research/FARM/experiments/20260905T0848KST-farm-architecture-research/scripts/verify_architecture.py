#!/usr/bin/env python3
"""Save reproducible engineering verification, never a benchmark accuracy score."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROUND = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROUND / "src"), str(ROUND.parents[1])]
from farm_arch.sidecar import endpoint_from_sidecar


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", type=Path, default=ROUND / "results/schema_sidecar_v1")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--round9-src", type=Path, help="optional legacy compiler regression; requires Pydantic dependencies")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("verification output already exists; choose a new version")
    stream = io.StringIO()
    tests = unittest.defaultTestLoader.discover(str(ROUND / "tests"))
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(tests)
    legacy = subprocess.run([sys.executable, str(ROUND / "diagnostics/schema_contract_regression.py"), "-v"],
        env={**os.environ, "FARM_SCHEMA_IMPL": "legacy"}, capture_output=True, text=True, check=False)
    corrected = subprocess.run([sys.executable, str(ROUND / "diagnostics/schema_contract_regression.py"), "-v"],
        env={**os.environ, "FARM_SCHEMA_IMPL": "evidence"}, capture_output=True, text=True, check=False)
    compiler = {"status": "not_requested"}
    if args.round9_src:
        compiler = {"status": "run"}
        for implementation in ("legacy", "evidence"):
            run = subprocess.run([sys.executable, str(ROUND / "diagnostics/legacy_compiler_regression.py"), "-v"],
                env={**os.environ, "FARM_COMPILER_IMPL": implementation,
                    "PYTHONPATH": str(args.round9_src) + os.pathsep + os.environ.get("PYTHONPATH", "")},
                capture_output=True, text=True, check=False)
            compiler[implementation] = {"returncode": run.returncode, "log": run.stdout + run.stderr}
    manifest = json.loads((args.sidecar / "manifest.json").read_text())
    integration = {}
    for side in ("trigger", "action"):
        target = args.sidecar / (side + "s.jsonl")
        if sha(target) != manifest["output_sha256"][side]:
            raise ValueError("sidecar differs from saved build manifest")
        counts = Counter()
        with target.open() as handle:
            for line in handle:
                endpoint = endpoint_from_sidecar(json.loads(line))
                if endpoint.side != side:
                    raise ValueError("sidecar contains wrong endpoint side")
                counts.update({"endpoints_adapted": 1, "fields": len(endpoint.fields),
                    "fields_with_help": sum(bool(f.help_text) for f in endpoint.fields),
                    "unknown_types": sum(f.value_type is None for f in endpoint.fields),
                    "unknown_requiredness": sum(f.required is None for f in endpoint.fields)})
        integration[side] = dict(counts)
    v2 = ROUND.parents[1] / "data/v2"
    current = {str(p.relative_to(v2)): sha(p) for p in sorted(v2.rglob("*")) if p.is_file()}
    unchanged = current == manifest["frozen_v2_hashes"]
    inventory = {str(p.relative_to(ROUND)): sha(p) for directory in ("src", "scripts", "tests", "diagnostics")
        for p in sorted((ROUND / directory).rglob("*.py"))}
    passed = result.wasSuccessful() and legacy.returncode == 1 and corrected.returncode == 0 and unchanged
    if args.round9_src:
        passed = passed and compiler["legacy"]["returncode"] == 1 and compiler["evidence"]["returncode"] == 0
        # A missing import is not an expected red-capable regression.
        passed = passed and "failures=4" in compiler["legacy"]["log"] and "Ran 4 tests" in compiler["evidence"]["log"]
    output = {"classification": "internal_engineering_verification", "time_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(), "passed": passed, "test_count": result.testsRun,
        "failures": len(result.failures), "errors": len(result.errors), "skipped": len(result.skipped),
        "unit_test_log": stream.getvalue(), "legacy_schema_expected_failure_returncode": legacy.returncode,
        "legacy_schema_log": legacy.stdout + legacy.stderr,
        "corrected_schema_returncode": corrected.returncode, "corrected_schema_log": corrected.stdout + corrected.stderr,
        "compiler_regression": compiler,
        "sidecar_interface_integration": integration, "frozen_v2_unchanged": unchanged,
        "source_sha256": inventory, "model_calls": 0,
        "scope": "unit/regression tests and whole-corpus metadata adaptation; NOT semantic binding or execution accuracy"}
    args.output.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with args.output.open("x") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(output, handle, sort_keys=True, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps({"passed": passed, "test_count": result.testsRun, "frozen_v2_unchanged": unchanged,
        "sidecar_interface_integration": integration, "output": str(args.output)}, sort_keys=True))
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
