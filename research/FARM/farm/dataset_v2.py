"""Canonical data primitives for the reviewer-grade FARM Dataset v2.

The v2 pipeline deliberately keeps data construction separate from model code.  It
normalises queries with Unicode semantics, resolves scrape duplicates by stable
voting, and renders paired plain/schema views from one canonical function record.
"""
from __future__ import annotations

import hashlib
import html
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from farm.render import parse_url, pretty_channel


DATASET_VERSION = "2.0.0"
NORMALIZER = "unicode-nfkc-casefold-alnum-v1"
PARAPHRASE_THRESHOLD = 0.90
CORPUS_FILENAMES = {
    "trigger": "triggers.json",
    "action": "actions.json",
    "query": "queries.json",
}
STOPWORDS = {
    "a", "an", "the", "to", "of", "in", "on", "for", "and", "or", "if",
    "when", "your", "my", "with", "from", "at", "is", "it", "this", "that",
    "be", "by",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def clean_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def query_key(value: Any) -> str:
    """Order-preserving Unicode query identity.

    NFKC handles full-width/compatibility forms, casefold handles Unicode case,
    and all non letter/number/combining-mark characters become separators.  This
    preserves Korean/Japanese/Arabic/etc. instead of collapsing them to ``""``.
    """
    value = html.unescape(str(value or ""))
    value = unicodedata.normalize("NFKC", value).casefold()
    out: list[str] = []
    for ch in value:
        category = unicodedata.category(ch)
        if ch.isalnum() or category.startswith("M"):
            out.append(ch)
        else:
            out.append(" ")
    return " ".join("".join(out).split())


def content_tokens(value: Any) -> frozenset[str]:
    return frozenset(
        token for token in query_key(value).split()
        if len(token) > 2 and token not in STOPWORDS
    )


def query_flags(value: Any) -> list[str]:
    raw = clean_text(value)
    key = query_key(raw)
    flags: list[str] = []
    if len(key.split()) <= 2:
        flags.append("short_tokens")
    if len(raw) < 10:
        flags.append("short_chars")
    if len(raw) == 100 and raw[-1:].isalnum():
        flags.append("possibly_truncated")
    if re.search(r"<\s*[a-zA-Z][^>]*>", raw):
        flags.append("contains_html")
    if any(ord(ch) > 127 for ch in raw):
        flags.append("non_ascii")
    if not key:
        flags.append("no_unicode_alnum")
    return flags


def normalise_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    text = clean_text(value).casefold()
    if text in {"true", "yes", "1"}:
        return True
    if text in {"false", "no", "0"}:
        return False
    return None


def stable_mode(values: Iterable[Any]) -> Any:
    """Most frequent non-empty value, with a deterministic lexical tie break."""
    vals = [value for value in values if value not in (None, "")]
    if not vals:
        return None
    encoded = Counter(stable_json(value) for value in vals)
    winner = min(encoded, key=lambda key: (-encoded[key], key))
    return json.loads(winner)


def _identity(value: Any) -> str:
    return unicodedata.normalize("NFKC", clean_text(value)).casefold()


def _real_entries(value: Any) -> list[tuple[str, dict[str, Any]]]:
    if not isinstance(value, dict):
        return []
    return [
        (str(name), entry)
        for name, entry in value.items()
        if isinstance(entry, dict)
    ]


def _entry_identity(name: str, entry: dict[str, Any]) -> str:
    return _identity(entry.get("Slug") or entry.get("Label") or name)


def _entry_quality(name: str, entry: dict[str, Any]) -> tuple[int, int, int, str]:
    populated = sum(value not in (None, "") for value in entry.values())
    has_setter = int(bool(clean_text(entry.get("Filter code method"))))
    human_key = int("\n" not in name)
    return populated, has_setter, human_key, stable_json(entry)


def _dedup_occurrence_entries(entries: list[tuple[str, dict[str, Any]]]) -> list[tuple[str, dict[str, Any]]]:
    grouped: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    for name, entry in entries:
        identity = _entry_identity(name, entry)
        if identity:
            grouped[identity].append((name, entry))
    out = []
    for identity in sorted(grouped):
        out.append(max(grouped[identity], key=lambda item: _entry_quality(*item)))
    return out


def _attribute_conflicts(values: Iterable[Any]) -> list[Any]:
    unique = {stable_json(value): value for value in values if value not in (None, "")}
    return [unique[key] for key in sorted(unique)] if len(unique) > 1 else []


def canonical_schema(
    occurrences: list[dict[str, Any]], kind: str, url: str
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Return canonical input fields, ingredients and a conflict ledger.

    Each scrape occurrence contributes at most one entry per semantic slug, so an
    old/new duplicate inside one payload cannot outvote other applets.  Conflicts
    are resolved by stable mode and retained in the ledger rather than hidden.
    """
    if kind == "trigger":
        primary, fallback = "Trigger fields", "Action fields"
    elif kind == "action":
        primary, fallback = "Action fields", "Trigger fields"
    else:
        primary, fallback = "Query fields", ""
    field_groups: dict[str, list[tuple[str, dict[str, Any], str]]] = defaultdict(list)
    ingredient_groups: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    fallback_occurrences = 0

    for component in occurrences:
        api = component.get("api_info") if isinstance(component.get("api_info"), dict) else {}
        source = primary
        fields = _real_entries(api.get(primary))
        if not fields:
            other = _real_entries(api.get(fallback)) if fallback else []
            if other:
                fallback_occurrences += 1
        for name, entry in _dedup_occurrence_entries(fields):
            field_groups[_entry_identity(name, entry)].append((name, entry, source))
        ingredients = _dedup_occurrence_entries(_real_entries(api.get("Ingredients")))
        for name, entry in ingredients:
            ingredient_groups[_entry_identity(name, entry)].append((name, entry))

    conflicts: list[dict[str, Any]] = []
    fields_out: list[dict[str, Any]] = []
    for identity in sorted(field_groups):
        rows = field_groups[identity]
        names = [clean_text(entry.get("Label") or name) for name, entry, _ in rows]
        slugs = [clean_text(entry.get("Slug")) for _, entry, _ in rows]
        setters = [clean_text(entry.get("Filter code method")) for _, entry, _ in rows]
        required_values = [normalise_bool(entry.get("Required")) for _, entry, _ in rows]
        default_values = [normalise_bool(entry.get("Can have default value")) for _, entry, _ in rows]
        record = {
            "slug": stable_mode(slugs) or identity,
            "label": stable_mode(names) or stable_mode(slugs) or identity,
            "required": stable_mode([v for v in required_values if v is not None]),
            "can_have_default": stable_mode([v for v in default_values if v is not None]),
            "filter_code_method": stable_mode(setters) or "",
            "bindable": any(bool(value) for value in setters),
        }
        attr = {
            "label": _attribute_conflicts(names),
            "slug": _attribute_conflicts(slugs),
            "required": _attribute_conflicts([v for v in required_values if v is not None]),
            "can_have_default": _attribute_conflicts([v for v in default_values if v is not None]),
            "filter_code_method": _attribute_conflicts(setters),
        }
        attr = {key: value for key, value in attr.items() if value}
        if attr:
            conflicts.append({"url": url, "section": "input_field", "identity": identity, "attributes": attr})
        fields_out.append(record)

    ingredients_out: list[dict[str, Any]] = []
    for identity in sorted(ingredient_groups):
        rows = ingredient_groups[identity]
        names = [clean_text(name) for name, _ in rows]
        slugs = [clean_text(entry.get("Slug")) for _, entry in rows]
        filter_codes = [clean_text(entry.get("Filter code")) for _, entry in rows]
        types = [clean_text(entry.get("Type")) for _, entry in rows]
        examples = [clean_text(entry.get("Example")) for _, entry in rows]
        record = {
            "slug": stable_mode(slugs) or identity,
            "label": stable_mode(names) or stable_mode(slugs) or identity,
            "filter_code": stable_mode(filter_codes) or "",
            "type": stable_mode(types) or "",
            "example": stable_mode(examples) or "",
        }
        attr = {
            "label": _attribute_conflicts(names),
            "slug": _attribute_conflicts(slugs),
            "filter_code": _attribute_conflicts(filter_codes),
            "type": _attribute_conflicts(types),
            "example": _attribute_conflicts(examples),
        }
        attr = {key: value for key, value in attr.items() if value}
        if attr:
            conflicts.append({"url": url, "section": "ingredient", "identity": identity, "attributes": attr})
        ingredients_out.append(record)

    if fallback_occurrences:
        conflicts.append({
            "url": url,
            "section": "schema_shape",
            "identity": kind,
            "attributes": {"fallback_occurrences": fallback_occurrences, "fallback_key": fallback},
        })
    return fields_out, ingredients_out, conflicts


def render_plain(record: dict[str, Any]) -> str:
    function_name = record.get("function_name") or record.get("service_name")
    parts = [
        f"channel: {pretty_channel(record['channel'])} | function: {clean_text(function_name)}"
    ]
    if clean_text(record.get("description")):
        parts.append(clean_text(record["description"]))
    return "\n".join(parts)


def _field_annotation(field: dict[str, Any]) -> str:
    flags: list[str] = []
    if field.get("required") is True:
        flags.append("required")
    elif field.get("required") is False:
        flags.append("optional")
    if field.get("bindable"):
        flags.append("ingredient-bindable")
    elif field.get("filter_code_method") == "":
        flags.append("no-setter-metadata")
    return f"{field['label']} [{'; '.join(flags)}]" if flags else str(field["label"])


def render_schema(record: dict[str, Any]) -> str:
    parts = [render_plain(record)]
    fields = record.get("input_fields") or []
    if fields:
        noun = {"trigger": "Trigger fields", "action": "Action fields", "query": "Query fields"}[record["kind"]]
        parts.append(f"{noun}: " + ", ".join(_field_annotation(field) for field in fields) + ".")
    ingredients = record.get("ingredients") or []
    if ingredients:
        rendered = [
            f"{item['slug']} ({item['type']})" if item.get("type") else str(item["slug"])
            for item in ingredients
        ]
        parts.append("Provides: " + ", ".join(rendered) + ".")
    return "\n".join(parts)


def component_quality(component: dict[str, Any]) -> tuple[int, int, int, str]:
    api = component.get("api_info") if isinstance(component.get("api_info"), dict) else {}
    entries = 0
    setters = 0
    for key in ("Trigger fields", "Action fields", "Ingredients"):
        for _, entry in _real_entries(api.get(key)):
            entries += 1
            setters += int(bool(clean_text(entry.get("Filter code method"))))
    return int(bool(api)), entries, setters, stable_json(component)


def applet_quality(applet: dict[str, Any]) -> tuple[int, int, int, int, str]:
    components = applet.get("components")
    component_rows = components if isinstance(components, list) else []
    valid_urls = 0
    api_rows = 0
    for component in component_rows:
        if not isinstance(component, dict):
            continue
        try:
            parse_url(component.get("url"))
            valid_urls += 1
        except ValueError:
            pass
        api_rows += int(isinstance(component.get("api_info"), dict))
    return (
        int(isinstance(components, list)),
        int(bool(clean_text(applet.get("description")))),
        valid_urls,
        api_rows,
        stable_json(applet),
    )


def group_id(query: str) -> str:
    return "q_" + sha256_bytes(query.encode("utf-8"))[:20]


def family_id(group_ids: Iterable[str]) -> str:
    value = "\n".join(sorted(group_ids))
    return "f_" + sha256_bytes(value.encode("utf-8"))[:20]


def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
