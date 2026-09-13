from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from farm_r9.artifact_io import read_json, sha256_file
from farm_r9.sampling import stratified_sample


def _endpoint_alias_map(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        for identifier in [row.get("url"), *(row.get("equivalent_urls") or [])]:
            if isinstance(identifier, str) and identifier:
                result[identifier] = row
    return result


def _schema_profile(
    row: Mapping[str, Any],
    trigger_map: Mapping[str, Mapping[str, Any]],
    action_map: Mapping[str, Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    triggers = [trigger_map[url] for url in row["gold_trigger_urls"] if url in trigger_map]
    actions = [action_map[url] for url in row["gold_action_urls"] if url in action_map]
    trigger_fields = [field for endpoint in triggers for field in endpoint.get("input_fields", [])]
    action_fields = [field for endpoint in actions for field in endpoint.get("input_fields", [])]
    profile = {
        "multi_gold": bool(row.get("multi_gold")),
        "trigger_required": any(bool(field.get("required")) for field in trigger_fields),
        "action_required": any(bool(field.get("required")) for field in action_fields),
        "action_optional": any(not bool(field.get("required")) for field in action_fields),
        "schema_missing": len(triggers) != len(row["gold_trigger_urls"]) or len(actions) != len(row["gold_action_urls"]),
    }
    stratum = "|".join(f"{key}={int(value)}" for key, value in profile.items())
    return stratum, profile


def prepare_farm_v2(data_root: Path, *, size: int, seed: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest = read_json(data_root / "manifest.json")
    test_path = data_root / "splits" / "test.json"
    triggers_path = data_root / "corpus" / "triggers.json"
    actions_path = data_root / "corpus" / "actions.json"
    rows = read_json(test_path)
    triggers = read_json(triggers_path)
    actions = read_json(actions_path)
    trigger_map = _endpoint_alias_map(triggers)
    action_map = _endpoint_alias_map(actions)

    enriched = []
    for row in rows:
        stratum, profile = _schema_profile(row, trigger_map, action_map)
        enriched.append({"source": row, "case_id": f"farm:{row['group_id']}", "stratum": stratum, "profile": profile})
    selected, sample_manifest = stratified_sample(
        enriched,
        size=size,
        seed=seed,
        benchmark="farm_v2_test",
        id_of=lambda row: row["case_id"],
        stratum_of=lambda row: row["stratum"],
    )
    cases = []
    for item in selected:
        row = item["source"]
        cases.append({
            "schema_version": "round9-case-v1",
            "benchmark": "farm_v2_test",
            "case_id": item["case_id"],
            "stratum": item["stratum"],
            "input": {"query": row["query"]},
            "private_gold": {
                "trigger_ids": row["gold_trigger_urls"],
                "action_ids": row["gold_action_urls"],
                "valid_pairs": row["valid_pairs"],
                "channel_pairs": row["gold_channel_pairs"],
            },
            "audit": {
                "group_id": row["group_id"],
                "semantic_family_id": row["semantic_family_id"],
                "schema_profile": item["profile"],
                "multi_gold": bool(row.get("multi_gold")),
            },
        })
    sample_manifest.update({
        "dataset_id": manifest["dataset_id"],
        "source_files": {
            "manifest.json": sha256_file(data_root / "manifest.json"),
            "splits/test.json": sha256_file(test_path),
            "corpus/triggers.json": sha256_file(triggers_path),
            "corpus/actions.json": sha256_file(actions_path),
        },
        "gold_open_policy": "selection and scoring only; stripped from inference envelope",
    })
    return cases, sample_manifest
