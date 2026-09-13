#!/usr/bin/env python3
"""Low-cost DGX monitoring: save counts every five minutes, never model requests."""
import argparse
import datetime
import json
import os
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--round9-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False, mode=0o700)
    definitions = {
        "bfcl_h21": ("results/bfcl-v4-missing-v3-official-horizon/raw/farm-r9-bfcl-dsv4-agent-v3h21/multi_turn/BFCL_v4_multi_turn_miss_param_result.json",
                     "results/bfcl-v4-missing-v3-official-horizon/summaries/farm-r9-bfcl-dsv4-agent-v3h21.json"),
        "yao_original_adaptive": ("results/yao_v2/deepseek/bounded_clarification_agent/records.jsonl",
                                  "results/yao_v2/deepseek/bounded_clarification_agent/aggregate.json"),
    }
    for arm in ("native_one_shot", "native_adaptive", "native_ask_all"):
        base = f"results/yao_native_astra_v3/{arm}"
        definitions["yao_" + arm] = (base + "/records.jsonl", base + "/aggregate.json")
    for arm in ("native_fixed_top10", "native_search_agent"):
        base = f"results/catalog_search_astra_v1/{arm}"
        definitions["catalog_" + arm] = (base + "/records.jsonl", base + "/aggregate.json")
    for _ in range(240):
        statuses = {}
        for name, (ledger_name, aggregate_name) in definitions.items():
            ledger = args.round9_root / ledger_name
            count, malformed = 0, 0
            if ledger.exists():
                with ledger.open() as handle:
                    for line in handle:
                        if not line.strip():
                            continue
                        try:
                            json.loads(line)
                            count += 1
                        except json.JSONDecodeError:
                            malformed += 1
            statuses[name] = {"records": count, "expected": 150,
                              "partial_write_lines": malformed,
                              "aggregate_exists": (args.round9_root / aggregate_name).is_file()}
        entry = {"time_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "runs": statuses}
        with (args.output / "progress.jsonl").open("a") as handle:
            os.fchmod(handle.fileno(), 0o600)
            handle.write(json.dumps(entry, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        if all(value["records"] == 150 and value["aggregate_exists"] for value in statuses.values()):
            with (args.output / "all_aggregates_present.json").open("x") as handle:
                os.fchmod(handle.fileno(), 0o600)
                json.dump(entry, handle, indent=2)
            return
        time.sleep(300)


if __name__ == "__main__":
    main()
