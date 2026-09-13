#!/usr/bin/env python3
"""Access-controlled aggregate audit of raw FARM metadata versus frozen v2.

No queries, endpoint IDs, labels, or field values are printed. No networking.
The output is internal audit material, not a release-shaped FARM table.
"""
from collections import Counter, defaultdict
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROUND = Path(__file__).resolve().parents[1]
FARM = ROUND.parents[1]
sys.path.insert(0, str(FARM))
from farm.dataset_v2 import canonical_schema, sha256_file, stable_json
from farm.render import parse_url


def exclusive(path, value):
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(path.parent, 0o700)
    with path.open("x") as handle:
        os.fchmod(handle.fileno(), 0o600)
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def index_raw(raw):
    versions = defaultdict(dict)
    raw_keys = {side: Counter() for side in ("trigger", "action")}
    anomalies = Counter()
    for service in raw:
        if not isinstance(service, dict) or not isinstance(service.get("applets"), list):
            anomalies["service_or_applet_collection_invalid"] += 1
            continue
        for applet in service["applets"]:
            if not isinstance(applet, dict):
                anomalies["applet_not_object"] += 1
                continue
            if not isinstance(applet.get("components"), list):
                anomalies["component_collection_not_list"] += 1
                continue
            for component in applet["components"]:
                if not isinstance(component, dict):
                    continue
                try:
                    channel, side, function = parse_url(component.get("url", ""))
                except ValueError:
                    continue
                if side not in raw_keys:
                    continue
                url = f"https://ifttt.com/{channel}/{side}s/{function}"
                api = component.get("api_info", {})
                if not isinstance(api, dict):
                    continue
                section = "Trigger fields" if side == "trigger" else "Action fields"
                entries = api.get(section, {})
                if isinstance(entries, dict):
                    for entry in entries.values():
                        if isinstance(entry, dict):
                            raw_keys[side].update(entry.keys())
                fingerprint = hashlib.sha256(stable_json(api).encode()).hexdigest()
                versions[(side, url)][fingerprint] = component
    return versions, raw_keys, anomalies


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--farm-root", type=Path, default=FARM)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw_path = args.farm_root / "data/iftttt_dataset_full_trigger_action.json"
    v2 = args.farm_root / "data/v2"
    versions, raw_keys, anomalies = index_raw(json.loads(raw_path.read_text()))
    report = {"classification": "confidential_internal_audit", "raw_sha256": sha256_file(raw_path),
              "v2_manifest_sha256": sha256_file(v2 / "manifest.json"),
              "source_shape_anomalies": dict(anomalies), "corpus": {}, "splits": {}}
    for side in ("trigger", "action"):
        corpus = json.loads((v2 / "corpus" / (side + "s.json")).read_text())
        fields = [f for row in corpus for f in row.get("input_fields", [])]
        counts = Counter()
        for row in corpus:
            relevant = {}
            for url in (row["url"], *row.get("equivalent_urls", [])):
                relevant.update(versions.get((side, url), {}))
            counts["endpoints_with_source"] += bool(relevant)
            source_by_slug = defaultdict(list)
            section = "Trigger fields" if side == "trigger" else "Action fields"
            for component in relevant.values():
                entries = component.get("api_info", {}).get(section, {})
                if not isinstance(entries, dict):
                    continue
                for label, entry in entries.items():
                    if isinstance(entry, dict):
                        identity = str(entry.get("Slug") or entry.get("Label") or label).strip().casefold()
                        source_by_slug[identity].append((label, entry))
            for field in row.get("input_fields", []):
                evidence = source_by_slug.get(field["slug"].strip().casefold(), [])
                counts["fields_with_source"] += bool(evidence)
                counts["source_explicit_type_but_v2_missing"] += bool(not field.get("type") and any(e.get("Type") for _, e in evidence))
                counts["source_help_but_v2_missing"] += bool(not field.get("help_text") and any(
                    e.get("Help text") or e.get("Description") or e.get("Helper text") for _, e in evidence))
                counts["required_true_optional_label_conflict"] += bool(field.get("required") is True and any(
                    re.search(r"\boptional\b", str(e.get("Label") or label), re.I) for label, e in evidence))
        report["corpus"][side] = {"endpoints": len(corpus), "input_fields": len(fields),
            "required_true": sum(f.get("required") is True for f in fields),
            "required_false": sum(f.get("required") is False for f in fields),
            "required_unknown": sum(f.get("required") is None for f in fields),
            "type_present": sum(bool(f.get("type")) for f in fields),
            "ingredients": sum(len(r.get("ingredients", [])) for r in corpus),
            "raw_field_attribute_occurrences": dict(raw_keys[side]), "source_join": dict(counts)}
    for name in ("encoder_train", "reranker_train", "dev", "test"):
        rows = json.loads((v2 / "splits" / (name + ".json")).read_text())
        split = {"groups": len(rows), "multi_gold_groups": sum(bool(r.get("multi_gold")) for r in rows)}
        for side in ("trigger", "action"):
            frequencies = Counter(u for r in rows for u in set(r["gold_" + side + "_urls"]))
            total = sum(frequencies.values())
            split[side] = {"distinct_gold_endpoints": len(frequencies), "endpoint_memberships": total,
                "singleton_endpoints": sum(n == 1 for n in frequencies.values()),
                "top10_endpoint_memberships": sum(sorted(frequencies.values(), reverse=True)[:10])}
        report["splits"][name] = split
    exclusive(args.output, report)
    print(json.dumps({"status": "complete", "output": str(args.output),
                      "corpus": {k: {n: v for n, v in values.items() if n != "raw_field_attribute_occurrences"}
                                 for k, values in report["corpus"].items()}, "splits": report["splits"]}, sort_keys=True))


if __name__ == "__main__":
    main()
