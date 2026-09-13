"""Evidence-preserving configuration schemas without rewriting retrieval v2.

The small canonical_schema interface retains the legacy tuple shape while
adding source evidence, help, and explicit unknown/conflicting type/requiredness.
No value types, defaults, field options, or connector capabilities are invented.
"""
from __future__ import annotations
from collections import defaultdict
import json
import re

from farm.dataset_v2 import canonical_schema as legacy_canonical_schema
from farm.dataset_v2 import clean_text, normalise_bool, stable_json, _entry_identity


def _observations(occurrences, section):
    grouped = defaultdict(dict)
    for component in occurrences:
        api = component.get("api_info")
        if not isinstance(api, dict) or not isinstance(api.get(section), dict):
            continue
        for label, entry in api[section].items():
            if not isinstance(entry, dict):
                continue
            evidence = {"source_label": label, "metadata": entry}
            # Identical copies do not become stronger evidence through repetition.
            grouped[_entry_identity(label, entry)][stable_json(evidence)] = evidence
    return {key: [json.loads(encoded) for encoded in sorted(rows)] for key, rows in grouped.items()}


def _nonempty_strings(rows, attributes):
    return sorted({str(row["metadata"][key]).strip() for row in rows for key in attributes
                   if isinstance(row["metadata"].get(key), str) and row["metadata"][key].strip()})


def canonical_schema(occurrences, kind, url):
    if kind not in ("trigger", "action", "query"):
        raise ValueError("unsupported endpoint kind")
    fields, ingredients, conflicts = legacy_canonical_schema(occurrences, kind, url)
    sources = _observations(occurrences, {"trigger": "Trigger fields", "action": "Action fields", "query": "Query fields"}[kind])
    ingredient_sources = _observations(occurrences, "Ingredients")
    for field in fields:
        rows = sources.get(_entry_identity(field["label"], {"Slug": field["slug"]}), [])
        types = _nonempty_strings(rows, ("Type", "Data type"))
        helpers = _nonempty_strings(rows, ("Helper text", "Help text", "Description"))
        required = sorted({value for row in rows
                           if (value := normalise_bool(row["metadata"].get("Required"))) is not None})
        optional_hint = any(re.search(r"\boptional\b", str(row["metadata"].get("Label") or row["source_label"]), re.I)
                            for row in rows)
        required_conflict = len(required) > 1 or (True in required and optional_hint)
        field.update({
            "type": types[0] if len(types) == 1 else "",
            "type_state": "known" if len(types) == 1 else "conflicting" if types else "unknown",
            "help_text": "\n".join(helpers), "help_text_variants": helpers,
            "required": required[0] if len(required) == 1 and not required_conflict else None,
            "requiredness_state": "conflicting" if required_conflict else "known" if required else "unknown",
            "optional_label_hint": optional_hint,
            "source_evidence": rows,
        })
        attributes = {}
        if required_conflict:
            attributes["required"] = {"explicit_values": required, "optional_label_hint": optional_hint}
        if len(types) > 1:
            attributes["type"] = types
        if attributes:
            conflicts.append({"url": url, "section": "configuration_evidence", "identity": field["slug"], "attributes": attributes})
        # Permission to have a default is not an actual default value.
        defaults = {stable_json(row["metadata"][key]) for row in rows for key in ("Default", "Default value")
                    if key in row["metadata"]}
        field["default_state"] = "known" if len(defaults) == 1 else "conflicting" if defaults else "unknown"
        field["default_values"] = [json.loads(v) for v in sorted(defaults)]
    for ingredient in ingredients:
        rows = ingredient_sources.get(_entry_identity(ingredient["label"], {"Slug": ingredient["slug"]}), [])
        types = _nonempty_strings(rows, ("Type",))
        ingredient.update({"type": types[0] if len(types) == 1 else "",
            "type_state": "known" if len(types) == 1 else "conflicting" if types else "unknown",
            "source_evidence": rows})
    # Preserve every conflict without order dependence or duplicate issue rows.
    conflicts = [json.loads(row) for row in sorted({stable_json(row) for row in conflicts})]
    return fields, ingredients, conflicts
