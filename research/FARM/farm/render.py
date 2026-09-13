"""The single document renderer.

Every consumer imports from here: the corpus builder, the training-pair builder,
the hard-negative miner, the indexer and the evaluator. There is exactly one
rendering of a function, so a training positive is byte-identical to the document
the retriever must return.

The previous pipeline had two renderings. Positives were produced by applying
rag/indexer.py:extract_trigger_text to the raw applet component, which carries no
'category' key, while the index applied the same function to a corpus row, which
always does. The two strings therefore never matched: 0 of 14,180 training
positives equalled any indexed document.
"""
from __future__ import annotations

import re
from typing import Any, Dict, Iterable

_URL = re.compile(r"ifttt\.com/([^/]+)/(triggers|actions|queries)/([^/?#]+)")

# api_info uses these sentinels in place of a dict when a function has no fields.
_PLACEHOLDER_KEYS = {"status"}


def parse_url(url: str) -> tuple[str, str, str]:
    """(channel_slug, kind, function_slug). Raises on anything unparseable."""
    m = _URL.search(str(url or ""))
    if not m:
        raise ValueError(f"unparseable component url: {url!r}")
    channel, kind, fn = m.group(1).lower(), m.group(2), m.group(3).lower()
    return channel, {"triggers": "trigger", "actions": "action", "queries": "query"}[kind], fn


def pretty_channel(slug: str) -> str:
    """weebly -> weebly, google_sheets -> google sheets, 5_minute_crafts -> 5 minute crafts.

    Derived from the url, never from a Filter code. The filter code spells the same
    channel differently ('5MinuteCrafts' vs '5_minute_crafts'), and it is absent
    entirely from 47.4% of actions.
    """
    return re.sub(r"[_\-]+", " ", str(slug or "").strip().lower()).strip()


def _clean_fields(api_info: Dict[str, Any], key: str) -> Dict[str, Dict[str, Any]]:
    """Field entries that are real dicts. Drops the 'No fields for this trigger'
    style placeholders, which a previous generator turned into a fabricated
    required field literally named 'status'."""
    raw = (api_info or {}).get(key) or {}
    if not isinstance(raw, dict):
        return {}
    return {n: v for n, v in raw.items()
            if isinstance(v, dict) and n not in _PLACEHOLDER_KEYS}


def _dedup(fields: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """The scrape emits every field TWICE: once keyed by its display label
    ("Which appliance?") and once by a widget-shaped key ("Appliancelist\nDropdown
    list"). 110 such groups exist in the action corpus and the `Slug` is identical
    in all 110, so the Slug is a safe identity. Left in, it double-counts required
    fields, which made a correctly-bound applet look like it had omitted exactly
    half its fields.

    Where the two copies disagree, keep the one WITHOUT a newline in its key: in 7
    of the 110 groups the widget-shaped copy has dropped 'Filter code method',
    which decides whether a field can accept a trigger ingredient at all.
    """
    best: Dict[str, tuple] = {}
    for name, v in fields.items():
        key = str(v.get("Slug") or "").strip().lower() or str(v.get("Label") or name).strip().lower()
        rank = (("\n" in name), not bool(str(v.get("Filter code method") or "").strip()))
        if key not in best or rank < best[key][0]:
            best[key] = (rank, name, v)
    return {n: v for _, n, v in best.values()}


def field_dict(api_info: Dict[str, Any], kind: str) -> Dict[str, Dict[str, Any]]:
    """Input fields, reading whichever key is actually present, deduplicated.

    11 urls carry api_info of the wrong shape (8 triggers hold 'Action fields',
    3 actions hold 'Ingredients'), so keying strictly on `kind` silently yields an
    empty field list for them.
    """
    primary = "Trigger fields" if kind == "trigger" else "Action fields"
    other = "Action fields" if kind == "trigger" else "Trigger fields"
    return _dedup(_clean_fields(api_info, primary) or _clean_fields(api_info, other))


def ingredient_dict(api_info: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return _clean_fields(api_info, "Ingredients")


def filter_code_tail(entry: Dict[str, Any]) -> str:
    """Channel.method.Tail -> Tail. The tail is the discriminative name; the Slug
    is often a catalogue-wide generic ('created_at' appears in 40.5% of triggers)."""
    code = entry.get("Filter code") or entry.get("Filter code method") or ""
    parts = str(code).split(".")
    return parts[-1].strip() if len(parts) > 1 else ""


def render(rec: Dict[str, Any]) -> str:
    """Render a normalised corpus record. `rec` must carry url, service_name,
    description, api_info; channel/category are optional and derived if absent."""
    channel, kind, _ = parse_url(rec["url"])
    api = rec.get("api_info") or {}
    name = str(rec.get("service_name") or "").strip()
    desc = str(rec.get("description") or "").strip()
    category = str(rec.get("category") or "").strip()

    head = [f"channel: {pretty_channel(channel)}"]
    if category:
        head.append(f"category: {category}")
    head.append(f"function: {name}")
    parts = [" | ".join(head)]
    if desc:
        parts.append(desc)

    fields = field_dict(api, kind)
    if fields:
        labels = [str(v.get("Label") or n).strip() for n, v in fields.items()]
        noun = "Trigger fields" if kind == "trigger" else "Action fields"
        parts.append(f"{noun}: " + ", ".join(labels) + ".")

    ings = ingredient_dict(api)
    if ings:
        out = []
        for n, v in ings.items():
            tail = filter_code_tail(v)
            slug = str(v.get("Slug") or n).strip()
            label = f"{tail}/{slug}" if tail and tail != slug else slug
            typ = str(v.get("Type") or "").strip()
            out.append(f"{label} ({typ})" if typ else label)
        parts.append("Provides: " + ", ".join(out) + ".")

    return "\n".join(parts)


def required_fields(rec: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Fields whose Required is the string 'true'. The raw data stores it as a
    string, so a plain truthiness test would mark 'false' as required."""
    _, kind, _ = parse_url(rec["url"])
    return {n: v for n, v in field_dict(rec.get("api_info") or {}, kind).items()
            if str(v.get("Required", "")).strip().lower() == "true"}


def bindable(entry: Dict[str, Any]) -> bool:
    """A field can accept a trigger ingredient only if it has a Filter code method.
    Fields without one are device/account pickers ('Which lights?') that no
    ingredient can fill; 38.0% of required action fields are of this kind."""
    return bool(str(entry.get("Filter code method") or "").strip())
