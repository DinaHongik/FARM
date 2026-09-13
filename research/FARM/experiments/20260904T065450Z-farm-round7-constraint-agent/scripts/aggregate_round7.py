#!/usr/bin/env python3
"""Aggregate Round 7 result JSON without reading examples or predictions.

Only ``EXPERIMENT_MATRIX.json`` and the eight preregistered files under
``results/`` are eligible inputs.  The script never opens records, attempts,
predictions, dataset splits, or the reserved confirmation window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "farm_round7_result_matrix_summary_v1"
SAFE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
ALPHA = 0.05


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _integer(value: Any, field: str) -> int:
    result = _number(value, field)
    if not result.is_integer():
        raise ValueError(f"{field} must be an integer")
    return int(result)


def _mapping(value: Any, field: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    return value


def _path_get(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for component in path.split("."):
        current = _mapping(current, path).get(component)
    if current is None:
        raise ValueError(f"result lacks {path}")
    return current


def holm_adjust(
    hypotheses: Sequence[tuple[str, float]], *, alpha: float = ALPHA
) -> dict[str, dict[str, Any]]:
    """Return Holm-adjusted p-values and step-down decisions.

    Ties are ordered by hypothesis ID, making serialized reports deterministic.
    """

    if not hypotheses:
        raise ValueError("Holm correction needs at least one hypothesis")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie inside (0,1)")
    ids = [name for name, _ in hypotheses]
    if len(ids) != len(set(ids)):
        raise ValueError("hypothesis IDs must be unique")
    checked: list[tuple[str, float]] = []
    for name, raw in hypotheses:
        p_value = _number(raw, f"p_value[{name}]")
        if not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p_value[{name}] must lie inside [0,1]")
        checked.append((name, p_value))
    ordered = sorted(checked, key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_adjusted = 0.0
    rejection_open = True
    output: dict[str, dict[str, Any]] = {}
    for index, (name, raw) in enumerate(ordered):
        multiplier = family_size - index
        running_adjusted = max(running_adjusted, min(1.0, raw * multiplier))
        critical = alpha / multiplier
        rejected = rejection_open and raw <= critical
        if not rejected:
            rejection_open = False
        output[name] = {
            "rank": index + 1,
            "raw_p": raw,
            "multiplier": multiplier,
            "holm_adjusted_p": running_adjusted,
            "step_down_critical_value": critical,
            "reject_at_0_05": rejected,
        }
    return output


def _safe_result_path(results_root: Path, experiment_id: str) -> Path:
    if not isinstance(experiment_id, str) or not SAFE_ID.fullmatch(experiment_id):
        raise ValueError(f"unsafe experiment ID: {experiment_id!r}")
    path = results_root / f"{experiment_id}.json"
    if path.is_symlink():
        raise ValueError(f"result symlinks are prohibited: {path}")
    if path.parent.resolve() != results_root.resolve():
        raise ValueError("result path escaped the result directory")
    return path


def _load_expected_result(results_root: Path, experiment_id: str) -> Mapping[str, Any] | None:
    path = _safe_result_path(results_root, experiment_id)
    if not path.exists():
        return None
    value = _read_json(path)
    return _mapping(value, str(path))


def _gate(observed: float | None, threshold: float, operator: str) -> dict[str, Any]:
    if observed is None:
        passed = False
    elif operator == ">=":
        passed = observed >= threshold
    elif operator == "<=":
        passed = observed <= threshold
    else:
        raise ValueError(f"unsupported gate operator: {operator}")
    return {
        "observed": observed,
        "operator": operator,
        "threshold": threshold,
        "pass": passed,
    }


def _validate_result_binding(
    result: Mapping[str, Any], *, experiment_id: str, run_id: str, dataset_id: str
) -> Mapping[str, Any]:
    binding = _mapping(result.get("binding"), f"{experiment_id}.binding")
    if binding.get("run_id") != run_id:
        raise ValueError(f"{experiment_id}: binding run_id mismatch")
    if binding.get("dataset_id") != dataset_id:
        raise ValueError(f"{experiment_id}: binding dataset_id mismatch")
    return binding


def _agent_summary(
    experiment_id: str,
    expected_rows: int,
    result: Mapping[str, Any],
    gates: Mapping[str, Any],
    *,
    run_id: str,
    dataset_id: str,
) -> dict[str, Any]:
    if result.get("status") != "completed":
        raise ValueError(f"{experiment_id}: result is not completed")
    binding = _validate_result_binding(
        result, experiment_id=experiment_id, run_id=run_id, dataset_id=dataset_id
    )
    if binding.get("arm") != experiment_id:
        raise ValueError(f"{experiment_id}: binding arm mismatch")
    metrics = _mapping(result.get("metrics"), f"{experiment_id}.metrics")
    rows = _integer(metrics.get("rows"), f"{experiment_id}.metrics.rows")
    if rows != expected_rows:
        raise ValueError(f"{experiment_id}: expected {expected_rows} rows, observed {rows}")
    joint = _mapping(metrics.get("function_joint"), f"{experiment_id}.function_joint")
    primary = _mapping(metrics.get("primary_comparison"), f"{experiment_id}.primary")
    protocol = _mapping(metrics.get("protocol"), f"{experiment_id}.protocol")
    bootstrap = _mapping(
        primary.get("family_cluster_paired_bootstrap"), f"{experiment_id}.bootstrap"
    )
    ci95 = bootstrap.get("ci95")
    if not isinstance(ci95, Sequence) or isinstance(ci95, (str, bytes)) or len(ci95) != 2:
        raise ValueError(f"{experiment_id}: invalid family bootstrap interval")
    raw_p = _number(primary.get("exact_two_sided_p"), f"{experiment_id}.raw_p")
    valid_rate = _integer(protocol.get("valid_rows"), f"{experiment_id}.valid_rows") / rows
    observed = {
        "rows": rows,
        "baseline_hits": _integer(joint.get("baseline_hits"), f"{experiment_id}.baseline_hits"),
        "baseline_rate": _number(joint.get("baseline_rate"), f"{experiment_id}.baseline_rate"),
        "agent_hits": _integer(joint.get("agent_hits"), f"{experiment_id}.agent_hits"),
        "agent_rate": _number(joint.get("agent_rate"), f"{experiment_id}.agent_rate"),
        "absolute_delta": _number(joint.get("absolute_delta"), f"{experiment_id}.delta"),
        "recoveries": _integer(primary.get("recoveries"), f"{experiment_id}.recoveries"),
        "regressions": _integer(primary.get("regressions"), f"{experiment_id}.regressions"),
        "net": _integer(primary.get("net"), f"{experiment_id}.net"),
        "raw_mcnemar_p": raw_p,
        "family_bootstrap_delta": _number(bootstrap.get("delta"), f"{experiment_id}.bootstrap.delta"),
        "family_bootstrap_ci95": [
            _number(ci95[0], f"{experiment_id}.bootstrap.ci95[0]"),
            _number(ci95[1], f"{experiment_id}.bootstrap.ci95[1]"),
        ],
        "accepted_override_precision": (
            None
            if primary.get("accepted_override_precision") is None
            else _number(
                primary.get("accepted_override_precision"),
                f"{experiment_id}.override_precision",
            )
        ),
        "baseline_correct_retention": (
            None
            if primary.get("baseline_correct_retention") is None
            else _number(
                primary.get("baseline_correct_retention"),
                f"{experiment_id}.retention",
            )
        ),
        "logical_model_calls_per_case": _number(
            protocol.get("logical_calls_per_case"), f"{experiment_id}.calls_per_case"
        ),
        "tokens_per_case": _number(
            protocol.get("tokens_per_case"), f"{experiment_id}.tokens_per_case"
        ),
        "protocol_valid_rate": valid_rate,
    }
    checks = {
        "absolute_delta": _gate(
            observed["absolute_delta"],
            _number(gates.get("minimum_absolute_delta"), "minimum_absolute_delta"),
            ">=",
        ),
        "baseline_correct_retention": _gate(
            observed["baseline_correct_retention"],
            _number(gates.get("minimum_baseline_correct_retention"), "minimum_baseline_correct_retention"),
            ">=",
        ),
        "accepted_override_precision": _gate(
            observed["accepted_override_precision"],
            _number(gates.get("minimum_override_precision"), "minimum_override_precision"),
            ">=",
        ),
        "logical_model_calls_per_case": _gate(
            observed["logical_model_calls_per_case"],
            _number(gates.get("maximum_logical_model_calls_per_case"), "maximum_logical_model_calls_per_case"),
            "<=",
        ),
        "tokens_per_case": _gate(
            observed["tokens_per_case"],
            _number(gates.get("maximum_tokens_per_case"), "maximum_tokens_per_case"),
            "<=",
        ),
        "protocol_valid_rate": _gate(
            observed["protocol_valid_rate"],
            _number(gates.get("minimum_protocol_valid_rate"), "minimum_protocol_valid_rate"),
            ">=",
        ),
    }
    return {"observed": observed, "gate_checks": checks}


def _router_summary(
    experiment_id: str,
    expected_fold: int,
    result: Mapping[str, Any],
    *,
    run_id: str,
    dataset_id: str,
) -> dict[str, Any]:
    if result.get("status") != "completed":
        raise ValueError(f"{experiment_id}: result is not completed")
    binding = _validate_result_binding(
        result, experiment_id=experiment_id, run_id=run_id, dataset_id=dataset_id
    )
    if binding.get("experiment_id") != experiment_id:
        raise ValueError(f"{experiment_id}: binding experiment mismatch")
    config = _mapping(binding.get("config"), f"{experiment_id}.binding.config")
    if _integer(config.get("holdout_fold"), f"{experiment_id}.holdout_fold") != expected_fold:
        raise ValueError(f"{experiment_id}: held-out fold mismatch")
    metrics = _mapping(result.get("metrics"), f"{experiment_id}.metrics")
    operational = _mapping(
        metrics.get("operational_four_class"), f"{experiment_id}.operational_four_class"
    )
    routing = _mapping(metrics.get("routing"), f"{experiment_id}.routing")
    sides = _mapping(metrics.get("side_routing"), f"{experiment_id}.side_routing")
    trigger = _mapping(sides.get("trigger"), f"{experiment_id}.side_routing.trigger")
    action = _mapping(sides.get("action"), f"{experiment_id}.side_routing.action")
    rows = _integer(metrics.get("rows"), f"{experiment_id}.metrics.rows")
    return {
        "heldout_fold": expected_fold,
        "heldout_rows": rows,
        "operational_accuracy": _number(operational.get("accuracy"), f"{experiment_id}.accuracy"),
        "operational_balanced_accuracy": _number(
            operational.get("balanced_accuracy"), f"{experiment_id}.balanced_accuracy"
        ),
        "operational_macro_f1": _number(operational.get("macro_f1"), f"{experiment_id}.macro_f1"),
        "route_precision": _number(routing.get("precision"), f"{experiment_id}.route_precision"),
        "route_recall": _number(routing.get("recall"), f"{experiment_id}.route_recall"),
        "route_f1": _number(routing.get("f1"), f"{experiment_id}.route_f1"),
        "trigger_side_f1": _number(trigger.get("f1"), f"{experiment_id}.trigger_f1"),
        "action_side_f1": _number(action.get("f1"), f"{experiment_id}.action_f1"),
    }


def _weighted_router_means(folds: Mapping[str, Mapping[str, Any]]) -> dict[str, float] | None:
    if not folds:
        return None
    total = sum(_integer(value["heldout_rows"], "heldout_rows") for value in folds.values())
    if total <= 0:
        raise ValueError("router held-out row count must be positive")
    fields = (
        "operational_accuracy",
        "operational_balanced_accuracy",
        "operational_macro_f1",
        "route_precision",
        "route_recall",
        "route_f1",
        "trigger_side_f1",
        "action_side_f1",
    )
    return {
        field: sum(
            _number(value[field], field) * _integer(value["heldout_rows"], "heldout_rows")
            for value in folds.values()
        )
        / total
        for field in fields
    }


def aggregate(run_root: Path) -> dict[str, Any]:
    run_root = run_root.resolve()
    matrix_path = run_root / "EXPERIMENT_MATRIX.json"
    matrix = _mapping(_read_json(matrix_path), str(matrix_path))
    run_id = matrix.get("run_id")
    if run_id != run_root.name:
        raise ValueError("run directory and matrix run_id differ")
    dataset_id = matrix.get("dataset_id")
    if not isinstance(dataset_id, str) or not dataset_id:
        raise ValueError("matrix lacks dataset_id")
    agent_specs = matrix.get("agent_discovery_arms")
    router_specs = matrix.get("gpu_router_folds")
    if not isinstance(agent_specs, list) or len(agent_specs) != 4:
        raise ValueError("matrix must preregister exactly four agent arms")
    if not isinstance(router_specs, list) or len(router_specs) != 4:
        raise ValueError("matrix must preregister exactly four router folds")
    results_root = run_root / "results"
    agent: dict[str, Any] = {}
    missing_agent: list[str] = []
    for raw_spec in agent_specs:
        spec = _mapping(raw_spec, "agent_discovery_arms[]")
        experiment_id = spec.get("id")
        path = _safe_result_path(results_root, experiment_id)
        result = _load_expected_result(results_root, experiment_id)
        if result is None:
            missing_agent.append(experiment_id)
            continue
        summary = _agent_summary(
            experiment_id,
            _integer(spec.get("scope"), f"{experiment_id}.scope"),
            result,
            _mapping(matrix.get("discovery_gates"), "discovery_gates"),
            run_id=run_id,
            dataset_id=dataset_id,
        )
        summary["result_path"] = str(path.relative_to(run_root))
        summary["result_sha256"] = _sha256_file(path)
        agent[experiment_id] = summary
    holm: dict[str, Any]
    if missing_agent:
        holm = {
            "status": "pending_all_four_preregistered_arms",
            "family_size": 4,
            "alpha": ALPHA,
            "missing": missing_agent,
            "by_arm": None,
        }
    else:
        adjusted = holm_adjust([
            (name, value["observed"]["raw_mcnemar_p"]) for name, value in agent.items()
        ])
        for name, correction in adjusted.items():
            check = {
                "observed": correction["holm_adjusted_p"],
                "operator": "<=",
                "threshold": ALPHA,
                "pass": correction["reject_at_0_05"],
            }
            agent[name]["gate_checks"]["holm_familywise_significance"] = check
            agent[name]["all_discovery_gates_pass"] = all(
                item["pass"] for item in agent[name]["gate_checks"].values()
            )
        holm = {
            "status": "complete",
            "family_size": 4,
            "alpha": ALPHA,
            "by_arm": adjusted,
        }
    for name in agent:
        if "all_discovery_gates_pass" not in agent[name]:
            agent[name]["all_discovery_gates_pass"] = None

    routers: dict[str, Any] = {}
    missing_routers: list[str] = []
    for raw_spec in router_specs:
        spec = _mapping(raw_spec, "gpu_router_folds[]")
        experiment_id = spec.get("id")
        path = _safe_result_path(results_root, experiment_id)
        result = _load_expected_result(results_root, experiment_id)
        if result is None:
            missing_routers.append(experiment_id)
            continue
        routers[experiment_id] = _router_summary(
            experiment_id,
            _integer(spec.get("holdout_fold"), f"{experiment_id}.holdout_fold"),
            result,
            run_id=run_id,
            dataset_id=dataset_id,
        ) | {
            "result_path": str(path.relative_to(run_root)),
            "result_sha256": _sha256_file(path),
        }
    total_router_rows = sum(value["heldout_rows"] for value in routers.values())
    router_complete = not missing_routers
    if router_complete and total_router_rows != 1145:
        raise ValueError(f"four router folds cover {total_router_rows} rows, expected 1145")

    return {
        "schema_version": SCHEMA_VERSION,
        "run_id": run_id,
        "dataset_id": dataset_id,
        "scientific_status": "development_only_no_sota_or_confirmation_claim",
        "input_policy": {
            "matrix": "EXPERIMENT_MATRIX.json",
            "eligible_results": [
                f"results/{spec['id']}.json" for spec in [*agent_specs, *router_specs]
            ],
            "records_read": False,
            "predictions_read": False,
            "dataset_rows_read": False,
            "reserved_confirmation_read": False,
            "test_split_read": False,
        },
        "agent_discovery": {
            "expected_arms": 4,
            "completed_arms": len(agent),
            "missing_arms": missing_agent,
            "arms": agent,
            "multiplicity": holm,
        },
        "side_router_cross_validation": {
            "expected_folds": 4,
            "completed_folds": len(routers),
            "missing_folds": missing_routers,
            "heldout_rows_observed": total_router_rows,
            "covers_all_1145_rows": router_complete and total_router_rows == 1145,
            "folds": routers,
            "heldout_row_weighted_fold_means": _weighted_router_means(routers),
            "aggregation_note": (
                "Weighted fold means are descriptive; macro-F1 and balanced accuracy are not "
                "reconstructed pooled metrics without the intentionally unread predictions."
            ),
        },
    }


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, replace: bool) -> None:
    if path.exists() and not replace:
        raise FileExistsError(f"refusing to overwrite {path}; pass --replace")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _validate_output_path(run_root: Path, output: Path) -> None:
    run_root = run_root.resolve()
    output = output.resolve()
    if output == run_root / "EXPERIMENT_MATRIX.json":
        raise ValueError("summary output cannot replace the experiment matrix")
    results_root = run_root / "results"
    if output == results_root or results_root in output.parents:
        raise ValueError("summary output cannot be written inside the immutable result directory")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    summary = aggregate(args.run_root)
    if args.output is None:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
    else:
        output = args.output.resolve()
        _validate_output_path(args.run_root, output)
        _write_json_atomic(output, summary, replace=args.replace)
        print(json.dumps({"status": "written", "output": str(output)}))


if __name__ == "__main__":
    main()


__all__ = ["aggregate", "holm_adjust"]
