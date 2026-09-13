#!/usr/bin/env python3
"""Freeze the pinned Yao Interactive and BFCL missing-parameter pilots."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.adapters.bfcl import (  # noqa: E402
    BFCL_BENCHMARK,
    prepare_bfcl_missing_parameter,
)
from farm_r9.adapters.yao_interactive import (  # noqa: E402
    YAO_BENCHMARK,
    prepare_yao_interactive,
)
from farm_r9.artifact_io import (  # noqa: E402
    assert_no_secrets,
    canonical_json,
    ordered_ids_sha256,
    read_json,
    sha256_file,
    sha256_text,
    write_json_atomic,
    write_jsonl_atomic,
)


def _jsonl_sha256(rows: Sequence[Mapping[str, Any]]) -> str:
    return sha256_text("".join(f"{canonical_json(dict(row))}\n" for row in rows))


def _write_frozen_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    """Create an immutable JSONL artifact, or validate an identical prior run."""
    expected_sha256 = _jsonl_sha256(rows)
    if path.exists():
        actual_sha256 = sha256_file(path)
        if actual_sha256 != expected_sha256:
            raise RuntimeError(f"refusing to overwrite a different frozen artifact: {path}")
        return actual_sha256
    write_jsonl_atomic(path, rows)
    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(f"post-write checksum mismatch: {path}")
    return actual_sha256


def _write_frozen_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    assert_no_secrets(manifest)
    if path.exists():
        if read_json(path) != dict(manifest):
            raise RuntimeError(f"refusing to overwrite a different frozen manifest: {path}")
        return
    write_json_atomic(path, dict(manifest))


def _prepare_yao(converted_path: Path, output_root: Path) -> dict[str, Any]:
    cases, source_manifest = prepare_yao_interactive(converted_path, size=150)
    if len(cases) != 150:
        raise AssertionError(f"{YAO_BENCHMARK}: adapter did not return exactly 150 cases")
    assert_no_secrets(cases)

    case_path = output_root / "prepared" / YAO_BENCHMARK / "cases.jsonl"
    manifest_path = output_root / "manifests" / "samples" / f"{YAO_BENCHMARK}.json"
    case_sha256 = _write_frozen_jsonl(case_path, cases)
    manifest = dict(source_manifest)
    manifest.update({
        "schema_version": "round9-sample-manifest-v1",
        "case_file": str(case_path.relative_to(output_root)),
        "case_payload_sha256": case_sha256,
        "ordered_case_ids_sha256": ordered_ids_sha256(cases),
    })
    _write_frozen_manifest(manifest_path, manifest)
    return {
        "benchmark": YAO_BENCHMARK,
        "sample_size": len(cases),
        "case_file": str(case_path),
        "case_payload_sha256": case_sha256,
        "ordered_case_ids_sha256": manifest["ordered_case_ids_sha256"],
        "manifest_file": str(manifest_path),
    }


def _prepare_bfcl(
    test_path: Path,
    possible_answer_path: Path,
    output_root: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    tests, answers, source_manifest = prepare_bfcl_missing_parameter(
        test_path,
        possible_answer_path,
        size=150,
        seed=seed,
    )
    if len(tests) != 150 or len(answers) != 150:
        raise AssertionError(f"{BFCL_BENCHMARK}: adapter did not return two aligned 150-row files")

    category_root = output_root / "prepared" / BFCL_BENCHMARK
    test_output = category_root / "BFCL_v4_multi_turn_miss_param.json"
    answer_output = (
        category_root
        / "possible_answer"
        / "BFCL_v4_multi_turn_miss_param.json"
    )
    manifest_path = output_root / "manifests" / "samples" / f"{BFCL_BENCHMARK}.json"
    test_sha256 = _write_frozen_jsonl(test_output, tests)
    answer_sha256 = _write_frozen_jsonl(answer_output, answers)

    # The selected rows remain byte-content-compatible with the official evaluator.
    # Upstream mock credentials therefore remain in the private test artifact, while
    # this manifest and all console output contain provenance and counts only.
    manifest = dict(source_manifest)
    manifest.update({
        "schema_version": "round9-sample-manifest-v1",
        "test_file": str(test_output.relative_to(output_root)),
        "possible_answer_file": str(answer_output.relative_to(output_root)),
        "test_payload_sha256": test_sha256,
        "possible_answer_payload_sha256": answer_sha256,
        "ordered_case_ids_sha256": ordered_ids_sha256(tests, id_key="id"),
    })
    _write_frozen_manifest(manifest_path, manifest)
    return {
        "benchmark": BFCL_BENCHMARK,
        "evaluation_scope": manifest["evaluation_scope"],
        "sample_size": len(tests),
        "test_file": str(test_output),
        "test_payload_sha256": test_sha256,
        "possible_answer_file": str(answer_output),
        "possible_answer_payload_sha256": answer_sha256,
        "ordered_case_ids_sha256": manifest["ordered_case_ids_sha256"],
        "manifest_file": str(manifest_path),
    }


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--yao-converted",
        required=True,
        type=Path,
        help="Inert JSON produced by convert_yao_official.py",
    )
    parser.add_argument(
        "--bfcl-test",
        required=True,
        type=Path,
        help="Pinned BFCL_v4_multi_turn_miss_param.json prompt file",
    )
    parser.add_argument(
        "--bfcl-possible-answer",
        required=True,
        type=Path,
        help="Pinned BFCL possible-answer file for the same category",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=RUN_ROOT,
        help="Round9 run directory receiving prepared/ and manifests/ (default: script root)",
    )
    parser.add_argument("--seed", type=int, default=9052026)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.resolve()
    summaries = {
        YAO_BENCHMARK: _prepare_yao(args.yao_converted.resolve(), output_root),
        BFCL_BENCHMARK: _prepare_bfcl(
            args.bfcl_test.resolve(),
            args.bfcl_possible_answer.resolve(),
            output_root,
            seed=args.seed,
        ),
    }
    # Deliberately print only paths, hashes, counts, and the partial-evaluation label.
    print(json.dumps(summaries, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
