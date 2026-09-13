#!/usr/bin/env python3
"""Run one frozen BFCL V4 Missing Parameters arm through Ollama Cloud."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.bfcl_runner import (  # noqa: E402
    ArmPolicy,
    BFCLArm,
    BFCLCloudTransport,
    build_official_handler_class,
    ensure_run_binding,
    generate_official_results,
    load_hydrated_sample,
    make_bfcl_handler_identity,
    register_official_model,
    run_official_partial_evaluator,
    validate_frozen_sample,
    verify_official_checkout,
    write_protocol_summary,
)
from farm_r9.cloud_limiter import OllamaCloudLimiter  # noqa: E402


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bfcl-root", required=True, type=Path)
    parser.add_argument(
        "--sample-test",
        type=Path,
        default=RUN_ROOT
        / "prepared"
        / "bfcl_v4_multi_turn_miss_param"
        / "BFCL_v4_multi_turn_miss_param.json",
    )
    parser.add_argument(
        "--sample-manifest",
        type=Path,
        default=RUN_ROOT
        / "manifests"
        / "samples"
        / "bfcl_v4_multi_turn_miss_param.json",
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--registry-name",
        required=True,
        help="Credential-free BFCL registry label without underscore or slash",
    )
    parser.add_argument("--arm", required=True, choices=[value.value for value in BFCLArm])
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--max-physical-calls-per-turn", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=9052026)
    parser.add_argument("--think", default="low")
    parser.add_argument("--timeout-seconds", type=float, default=240.0)
    parser.add_argument("--max-transport-attempts", type=int, default=2)
    parser.add_argument("--host-env", default="OLLAMA_CLOUD_HOST")
    parser.add_argument("--api-key-env", default="OLLAMA_API_KEY")
    parser.add_argument(
        "--shared-lock-directory",
        required=True,
        type=Path,
        help="The same absolute directory must be supplied to every Cloud worker",
    )
    parser.add_argument("--result-root", required=True, type=Path)
    parser.add_argument("--score-root", required=True, type=Path)
    parser.add_argument("--journal", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--summary", required=True, type=Path)
    parser.add_argument(
        "--mode", choices=("generate", "evaluate", "both"), default="both"
    )
    return parser.parse_args(argv)


def _required_environment(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"required environment variable is unset: {name}")
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    commit = verify_official_checkout(args.bfcl_root)
    sample_rows, sample_manifest = validate_frozen_sample(
        args.sample_test, args.sample_manifest
    )
    cases = load_hydrated_sample(args.bfcl_root, sample_rows)

    arm = BFCLArm(args.arm)
    policy = ArmPolicy(arm, args.max_physical_calls_per_turn)
    if isinstance(args.workers, bool) or not 1 <= args.workers <= 3:
        raise ValueError("BFCL workers must be in 1..3")
    thinking_mode = None if args.think.casefold() == "none" else args.think
    ensure_run_binding(
        args.result_root / args.registry_name / "RUN_BINDING.json",
        registry_name=args.registry_name,
        arm=arm,
        model=args.model,
        sample_sha256=sample_manifest["test_payload_sha256"],
        official_commit=commit,
        temperature=args.temperature,
        seed=args.seed,
        thinking_mode=thinking_mode,
        max_physical_calls_per_turn=args.max_physical_calls_per_turn,
        workers=args.workers,
    )
    if args.mode == "evaluate":
        class EvaluationOnlyTransport:
            model = args.model

            def chat(self, **kwargs):
                raise RuntimeError("evaluation-only transport cannot perform inference")

        transport = EvaluationOnlyTransport()
    else:
        limiter = OllamaCloudLimiter(args.shared_lock_directory, max_concurrent=3)
        transport = BFCLCloudTransport(
            host=_required_environment(args.host_env),
            api_key=_required_environment(args.api_key_env),
            model=args.model,
            limiter=limiter,
            journal_path=args.journal,
            cache_directory=args.cache_dir,
            timeout_seconds=args.timeout_seconds,
            max_transport_attempts=args.max_transport_attempts,
            temperature=args.temperature,
            seed=args.seed,
            think=thinking_mode,
        )
    handler_class = build_official_handler_class(transport=transport, policy=policy)
    register_official_model(
        bfcl_root=args.bfcl_root,
        registry_name=args.registry_name,
        actual_model=args.model,
        handler_class=handler_class,
    )
    handler = handler_class(
        # The transport above preserves the exact provider model ID.  BFCL's
        # executable mock backend needs a separate colon-free Python identity.
        model_name=make_bfcl_handler_identity(
            registry_name=args.registry_name,
            arm=arm,
            provider_model=args.model,
        ),
        temperature=args.temperature,
        registry_name=args.registry_name,
        is_fc_model=True,
    )

    generation: dict[str, object]
    evaluation: dict[str, object]
    if args.mode in {"generate", "both"}:
        generation = generate_official_results(
            handler=handler,
            cases=cases,
            result_path=(
                args.result_root
                / args.registry_name
                / "multi_turn"
                / "BFCL_v4_multi_turn_miss_param_result.json"
            ),
            workers=args.workers,
        )
    else:
        generation = {"status": "not_requested"}

    if args.mode in {"evaluate", "both"}:
        evaluation = run_official_partial_evaluator(
            result_root=args.result_root,
            score_root=args.score_root,
            registry_name=args.registry_name,
        )
    else:
        evaluation = {"status": "not_requested"}

    write_protocol_summary(
        args.summary,
        generation=generation,
        evaluation=evaluation,
        arm=arm,
        model=args.model,
        registry_name=args.registry_name,
        official_commit=commit,
    )
    # The terminal receives only aggregate protocol fields; no raw BFCL text,
    # state, tool arguments, model output, or provider credential is printed.
    print(
        json.dumps(
            {
                "arm": arm.value,
                "evaluation_scope": "partial:150_of_200",
                "leaderboard_comparable": False,
                "generation": generation,
                "evaluation": evaluation,
                "summary": str(args.summary),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
