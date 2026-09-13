#!/usr/bin/env python3
"""Verify pinned BFCL imports, registration, and tool compilation without I/O calls."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.artifact_io import sha256_file, sha256_text  # noqa: E402
from farm_r9.bfcl_runner import (  # noqa: E402
    ArmPolicy,
    BFCLArm,
    build_official_handler_class,
    load_hydrated_sample,
    make_bfcl_handler_identity,
    register_official_model,
    validate_frozen_sample,
    verify_official_checkout,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", required=True, type=Path)
    parser.add_argument("--sample-test", required=True, type=Path)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--registry-name", required=True)
    parser.add_argument("--arm", required=True, choices=[value.value for value in BFCLArm])
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-physical-calls-per-turn", type=int, default=4)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commit = verify_official_checkout(args.bfcl_root)
    sample_rows, _ = validate_frozen_sample(args.sample_test, args.sample_manifest)
    cases = load_hydrated_sample(args.bfcl_root, sample_rows)
    arm = BFCLArm(args.arm)
    policy = ArmPolicy(arm, args.max_physical_calls_per_turn)

    class StartupOnlyTransport:
        model = args.model

        def chat(self, **kwargs):
            raise AssertionError("startup preflight must never perform inference")

    handler_class = build_official_handler_class(
        transport=StartupOnlyTransport(),
        policy=policy,
    )
    register_official_model(
        bfcl_root=args.bfcl_root,
        registry_name=args.registry_name,
        actual_model=args.model,
        handler_class=handler_class,
    )
    handler = handler_class(
        model_name=make_bfcl_handler_identity(
            registry_name=args.registry_name,
            arm=arm,
            provider_model=args.model,
        ),
        temperature=args.temperature,
        registry_name=args.registry_name,
        is_fc_model=True,
    )
    first_case = cases[0]
    inference_data = handler._pre_query_processing_FC({}, first_case)
    inference_data = handler._compile_tools(inference_data, first_case)

    from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING

    registered = MODEL_CONFIG_MAPPING.get(args.registry_name)
    if registered is None or registered.model_handler is not handler_class:
        raise RuntimeError("BFCL custom model registration did not persist")
    if registered.model_name != args.model:
        raise RuntimeError("BFCL registration changed the provider model identity")
    if not isinstance(inference_data.get("tools"), list) or not inference_data["tools"]:
        raise RuntimeError("BFCL official tool compilation produced no tools")

    # This CLI has no output path, transport credentials, limiter, cache, journal,
    # result root, or score root.  Its only output is this aggregate startup proof.
    print(
        json.dumps(
            {
                "status": "pass",
                "mode": "startup_import_registration_preflight",
                "network_requests": 0,
                "result_artifacts_written": 0,
                "official_commit": commit,
                "sample_n": len(cases),
                "schemas_hydrated_n": sum(bool(case.get("function")) for case in cases),
                "first_case_compiled_tool_count": len(inference_data["tools"]),
                "sample_manifest_sha256": sha256_file(args.sample_manifest),
                "model_sha256": sha256_text(args.model),
                "registry_name": args.registry_name,
                "arm": arm.value,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
