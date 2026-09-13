#!/usr/bin/env python3
"""Build a private evidence sidecar; never overwrite frozen FARM-v2 artifacts."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
import sys

ROUND = Path(__file__).resolve().parents[1]
FARM = ROUND.parents[1]
sys.path.insert(0, str(FARM))
sys.path.insert(0, str(ROUND / "src"))
from farm.dataset_v2 import sha256_file, stable_json
from farm_arch.schema import canonical_schema
from audit_configuration_inputs import index_raw, exclusive


def build_record(row, side, versions):
    relevant = {}
    for url in (row["url"], *row.get("equivalent_urls", [])):
        relevant.update(versions.get((side, url), {}))
    if not relevant:
        raise ValueError("endpoint source evidence unavailable")
    fields, ingredients, conflicts = canonical_schema(list(relevant.values()), side, row["url"])
    by_slug = {f["slug"].casefold(): f for f in fields}
    original_slugs = {f["slug"].casefold() for f in row["input_fields"]}
    if not original_slugs <= by_slug.keys():
        raise ValueError("sidecar would lose a frozen input field")
    # Frozen input identities remain the interface. Source-only fields require
    # explicit reconciliation rather than quietly changing configuration tasks.
    return {"endpoint_id": row["url"], "side": side,
        "input_fields": [by_slug[f["slug"].casefold()] for f in row["input_fields"]],
        "source_only_input_fields": [f for f in fields if f["slug"].casefold() not in original_slugs],
        "ingredients": ingredients, "conflicts": conflicts,
        "validation_capabilities": {"remote_field_validation": "unverified", "dry_run": "unavailable", "execution": "unavailable"}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--farm-root", type=Path, default=FARM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.farm_root / "data/iftttt_dataset_full_trigger_action.json"
    v2 = args.farm_root / "data/v2"
    before = {str(p.relative_to(v2)): sha256_file(p) for p in sorted(v2.rglob("*")) if p.is_file()}
    versions, _, anomalies = index_raw(json.loads(raw_path.read_text()))
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    stats, hashes = {}, {}
    for side in ("trigger", "action"):
        corpus = json.loads((v2 / "corpus" / (side + "s.json")).read_text())
        records = [build_record(row, side, versions) for row in corpus]
        target = args.output / (side + "s.jsonl")
        with target.open("x") as handle:
            os.fchmod(handle.fileno(), 0o600)
            for record in records:
                handle.write(stable_json(record) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        fields = [f for r in records for f in r["input_fields"]]
        stats[side] = {"endpoints": len(records), "input_fields": len(fields),
            "with_help_text": sum(bool(f["help_text"]) for f in fields),
            "type_states": dict(Counter(f["type_state"] for f in fields)),
            "requiredness_states": dict(Counter(f["requiredness_state"] for f in fields)),
            "required_false": sum(f["required"] is False for f in fields),
            "source_only_fields": sum(len(r["source_only_input_fields"]) for r in records)}
        hashes[side] = sha256_file(target)
    after = {str(p.relative_to(v2)): sha256_file(p) for p in sorted(v2.rglob("*")) if p.is_file()}
    if before != after:
        raise ValueError("frozen v2 artifacts changed during sidecar build")
    report = {"classification": "confidential_internal_sidecar", "schema_version": "evidence-v1",
        "raw_source_sha256": sha256_file(raw_path), "frozen_v2_hashes": before,
        "output_sha256": hashes, "statistics": stats, "source_shape_anomalies": dict(anomalies),
        "v2_unchanged": True, "types_inferred": False, "model_calls": 0,
        "scope": "configuration metadata only; not a new benchmark or executable binding result"}
    exclusive(args.output / "manifest.json", report)
    print(json.dumps({"v2_unchanged": True, "statistics": stats, "output_sha256": hashes}, sort_keys=True))


if __name__ == "__main__":
    main()
