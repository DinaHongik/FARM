#!/usr/bin/env python3
"""Read PUBLIC response caches and emit structural counts, never response text."""
from __future__ import annotations
import argparse
import collections
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from farm_r9.ollama_client import parse_json_object


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path)
    args = parser.parse_args()
    counts = collections.Counter()
    reasons = collections.Counter()
    sizes = []
    tokens = []
    for path in sorted(args.cache.glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("data_classification") != "public":
            raise SystemExit("refusing non-public response cache")
        content = record.get("content", "")
        raw = record.get("raw_response", {})
        counts["responses"] += 1
        counts["empty_content"] += not bool(content.strip())
        counts["native_tool_calls"] += bool(record.get("tool_calls"))
        counts["fenced_content"] += content.strip().startswith("```")
        reasons[str(raw.get("done_reason"))] += 1
        sizes.append(len(content))
        tokens.append(record.get("completion_tokens", 0))
        try:
            _, strict = parse_json_object(content)
            counts["strict_json" if strict else "recoverable_json"] += 1
        except ValueError:
            counts["unparseable_content"] += 1
            message = raw.get("message", {})
            counts["unparseable_with_thinking"] += bool(message.get("thinking"))
            counts["unparseable_with_tool_calls"] += bool(message.get("tool_calls"))
    print(json.dumps({"counts": dict(counts), "done_reason_counts": dict(reasons),
                      "content_length_min": min(sizes, default=0),
                      "content_length_max": max(sizes, default=0),
                      "completion_tokens_max": max(tokens, default=0)}, indent=2))


if __name__ == "__main__":
    main()
