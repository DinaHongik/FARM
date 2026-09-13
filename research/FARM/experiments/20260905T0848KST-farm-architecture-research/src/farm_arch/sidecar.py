"""Adapter from frozen evidence sidecars to the local configuration interface.

Does not infer types from widgets, slug spelling or help text. Binding/auth
capabilities require a future trusted connector adapter; this scrape alone
cannot certify them. All source evidence contributes to the schema revision.
"""
import hashlib
import json
from .configuration import Endpoint, Field, Ingredient


def endpoint_from_sidecar(record: dict) -> Endpoint:
    if record.get("source_only_input_fields"):
        raise ValueError("unreconciled source-only fields")
    revision = hashlib.sha256(json.dumps(record, sort_keys=True, separators=(",", ":"),
        allow_nan=False, ensure_ascii=False).encode()).hexdigest()
    fields = tuple(Field(slug=f["slug"], required=f["required"] if f["requiredness_state"] == "known" else None,
        value_type=f["type"] if f["type_state"] == "known" else None,
        label=f.get("label", ""), help_text=f.get("help_text", "")) for f in record["input_fields"])
    ingredients = tuple(Ingredient(i["slug"], i["type"] if i["type_state"] == "known" else None)
        for i in record["ingredients"])
    return Endpoint(record["endpoint_id"], record["side"], revision, fields, ingredients)
