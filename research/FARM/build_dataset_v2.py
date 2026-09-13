"""Build the immutable, multi-gold FARM Dataset v2.

Dataset v2 is emitted beside the legacy artifacts and never reads their derived
files.  Its only data input is the raw scrape.  The output is deterministic under
raw service/applet reordering and uses four disjoint semantic-cluster splits:
70% encoder train, 10% reranker train, 10% dev, 10% locked test.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median
from typing import Any

from farm.dataset_v2 import (
    CORPUS_FILENAMES,
    DATASET_VERSION,
    NORMALIZER,
    PARAPHRASE_THRESHOLD,
    applet_quality,
    canonical_schema,
    clean_text,
    component_quality,
    content_tokens,
    family_id,
    group_id,
    jaccard,
    query_flags,
    query_key,
    render_plain,
    render_schema,
    sha256_bytes,
    sha256_file,
    stable_json,
    stable_mode,
)
from farm.render import parse_url


ROOT = Path(__file__).resolve().parent
DEFAULT_RAW = ROOT / "data/iftttt_dataset_full_trigger_action.json"
DEFAULT_OUT = ROOT / "data/v2"
SEED = 42
SPLIT_RATIOS = {
    "encoder_train": 0.70,
    "reranker_train": 0.10,
    "dev": 0.10,
    "test": 0.10,
}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=1) + "\n",
        encoding="utf-8",
    )


def canonical_applet_url(value: Any) -> str:
    match = re.search(r"ifttt\.com/(applets|connections)/([^/?#]+)", str(value or ""), re.I)
    return f"https://ifttt.com/{match.group(1).casefold()}/{match.group(2)}" if match else ""


def canonical_component_url(value: Any) -> str:
    channel, kind, function = parse_url(str(value or ""))
    plural = {"trigger": "triggers", "action": "actions", "query": "queries"}[kind]
    return f"https://ifttt.com/{channel}/{plural}/{function}"


def best_component(versions: list[dict[str, Any]]) -> dict[str, Any]:
    selected = max(versions, key=component_quality)
    result = json.loads(json.dumps(selected))
    result["url"] = canonical_component_url(result.get("url"))
    return result


def structure_signature(applet: dict[str, Any]) -> tuple[tuple[str, str], ...] | None:
    components = applet.get("components")
    if not isinstance(components, list):
        return None
    out = []
    for component in components:
        if not isinstance(component, dict):
            continue
        try:
            url = canonical_component_url(component.get("url"))
        except ValueError:
            continue
        out.append((clean_text(component.get("label")), url))
    return tuple(sorted(out))


def collect_raw(raw: list[Any]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Canonicalise applet occurrences without relying on input order."""
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    missing_url_occurrences = 0
    invalid_applet_objects = 0
    service_categories: dict[str, set[str]] = defaultdict(set)
    service_names: dict[str, list[str]] = defaultdict(list)
    scrape_timestamps = []
    coverage_rows = []

    for service in raw:
        if not isinstance(service, dict):
            continue
        service_url = clean_text(service.get("service_url"))
        channel = ""
        if "ifttt.com/" in service_url:
            channel = service_url.rstrip("/").rsplit("/", 1)[-1].casefold()
        if channel:
            service_names[channel].append(clean_text(service.get("service_name")))
            for category in service.get("categories") or []:
                if clean_text(category):
                    service_categories[channel].add(clean_text(category))
        if clean_text(service.get("scraped_at")):
            scrape_timestamps.append(clean_text(service.get("scraped_at")))
        if isinstance(service.get("applets_scraped"), int) and isinstance(service.get("total_applets_available"), int):
            coverage_rows.append({
                "service": clean_text(service.get("service_name")),
                "scraped": service["applets_scraped"],
                "available": service["total_applets_available"],
            })
        applets = service.get("applets")
        if not isinstance(applets, list):
            continue
        for applet in applets:
            if not isinstance(applet, dict):
                invalid_applet_objects += 1
                continue
            url = canonical_applet_url(applet.get("applet_url"))
            if not url:
                missing_url_occurrences += 1
                continue
            occurrences[url].append(applet)

    canonical: dict[str, dict[str, Any]] = {}
    duplicate_structure_conflicts = []
    metadata_conflicts = Counter()
    for url in sorted(occurrences):
        versions = occurrences[url]
        signatures = {signature for item in versions if (signature := structure_signature(item)) is not None}
        if len(signatures) > 1:
            duplicate_structure_conflicts.append({
                "applet_url": url,
                "signatures": [[list(pair) for pair in signature] for signature in sorted(signatures)],
            })
        for key in (
            "description", "additional_description", "service_applet_name", "related_services",
            "trigger_categories", "action_categories", "user_count",
        ):
            values = {stable_json(item.get(key)) for item in versions}
            if len(values) > 1:
                metadata_conflicts[key] += 1

        selected = json.loads(json.dumps(max(versions, key=applet_quality)))
        selected["applet_url"] = url
        selected["n_occurrences"] = len(versions)
        selected["all_user_counts"] = sorted({clean_text(item.get("user_count")) for item in versions if clean_text(item.get("user_count"))})

        # Preserve the selected structure, but use the richest deterministic copy
        # of each component from every duplicate host listing.
        all_components: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in versions:
            if not isinstance(item.get("components"), list):
                continue
            for component in item["components"]:
                if not isinstance(component, dict):
                    continue
                try:
                    component_url = canonical_component_url(component.get("url"))
                except ValueError:
                    continue
                all_components[(clean_text(component.get("label")), component_url)].append(component)
        if isinstance(selected.get("components"), list):
            enriched = []
            for component in selected["components"]:
                if not isinstance(component, dict):
                    continue
                try:
                    component_url = canonical_component_url(component.get("url"))
                except ValueError:
                    enriched.append(component)
                    continue
                key = (clean_text(component.get("label")), component_url)
                enriched.append(best_component(all_components.get(key) or [component]))
            selected["components"] = enriched
        canonical[url] = selected

    audit = {
        "raw_service_records": sum(isinstance(service, dict) for service in raw),
        "raw_applet_occurrences": sum(len(items) for items in occurrences.values()) + missing_url_occurrences,
        "unique_applet_urls": len(occurrences),
        "duplicate_occurrences": sum(len(items) - 1 for items in occurrences.values()),
        "missing_url_occurrences": missing_url_occurrences,
        "invalid_applet_objects": invalid_applet_objects,
        "duplicate_structure_conflicts": duplicate_structure_conflicts,
        "metadata_conflict_urls": dict(sorted(metadata_conflicts.items())),
        "service_categories": {key: sorted(value) for key, value in sorted(service_categories.items())},
        "service_names": {key: stable_mode(value) or key for key, value in sorted(service_names.items())},
        "provenance": {
            "timestamped_service_records": len(scrape_timestamps),
            "scrape_timestamp_min": min(scrape_timestamps) if scrape_timestamps else None,
            "scrape_timestamp_max": max(scrape_timestamps) if scrape_timestamps else None,
            "coverage_metadata_records": len(coverage_rows),
            "coverage_incomplete_records": sum(row["scraped"] < row["available"] for row in coverage_rows),
            "coverage_rows": sorted(coverage_rows, key=lambda row: row["service"]),
        },
    }
    return canonical, audit


def build_corpus(
    applets: dict[str, dict[str, Any]], raw_audit: dict[str, Any]
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]], list[dict[str, Any]]]:
    occurrences: dict[str, list[dict[str, Any]]] = defaultdict(list)
    api_presence: Counter[str] = Counter()
    for applet_url in sorted(applets):
        applet = applets[applet_url]
        per_url: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for component in applet.get("components") or []:
            if not isinstance(component, dict):
                continue
            try:
                url = canonical_component_url(component.get("url"))
            except ValueError:
                continue
            per_url[url].append(component)
        for url in sorted(per_url):
            selected = best_component(per_url[url])
            occurrences[url].append(selected)
            api_presence[url] += int(isinstance(selected.get("api_info"), dict))

    by_kind: dict[str, list[dict[str, Any]]] = defaultdict(list)
    all_conflicts: list[dict[str, Any]] = []
    for url in sorted(occurrences):
        rows = occurrences[url]
        channel, kind, function_slug = parse_url(url)
        fields, ingredients, conflicts = canonical_schema(rows, kind, url)
        names = [clean_text(row.get("service_name")) for row in rows]
        descriptions = [clean_text(row.get("description")) for row in rows]
        titles = [clean_text(row.get("title")) for row in rows]
        critical_conflict = any(
            set(item.get("attributes", {})) & {"slug", "type", "required", "can_have_default"}
            for item in conflicts
        )
        record = {
            "url": url,
            "kind": kind,
            "channel": channel,
            "function_slug": function_slug,
            "function_name": stable_mode(names) or function_slug,
            "description": stable_mode(descriptions) or "",
            "title": stable_mode(titles) or "",
            "categories": raw_audit["service_categories"].get(channel, []),
            "channel_display": raw_audit["service_names"].get(channel, channel),
            "input_fields": fields,
            "ingredients": ingredients,
            "schema_status": (
                "critical_conflict_resolved" if critical_conflict
                else "metadata_variants_resolved" if conflicts
                else "present" if api_presence[url]
                else "missing_api_info"
            ),
            "n_unique_applets": len(rows),
            "api_info_coverage": {"present": api_presence[url], "total": len(rows)},
        }
        record["text_plain"] = render_plain(record)
        record["text_schema"] = render_schema(record)
        all_conflicts.extend(conflicts)
        by_kind[kind].append(record)

    # Declare semantic/document aliases instead of forcing impossible tie grading.
    aliases: list[dict[str, Any]] = []
    for kind, records in by_kind.items():
        fingerprints: dict[str, list[str]] = defaultdict(list)
        for record in records:
            payload = {
                "channel": record["channel"],
                "function_name": query_key(record["function_name"]),
                "description": query_key(record["description"]),
                "input_fields": record["input_fields"],
                "ingredients": record["ingredients"],
            }
            fingerprints[stable_json(payload)].append(record["url"])
        equivalent: dict[str, list[str]] = {}
        for urls in fingerprints.values():
            if len(urls) > 1:
                urls = sorted(urls)
                aliases.append({"kind": kind, "urls": urls})
                for url in urls:
                    equivalent[url] = urls
        for record in records:
            record["equivalent_urls"] = equivalent.get(record["url"], [record["url"]])
        records.sort(key=lambda record: record["url"])
    return dict(by_kind), sorted(all_conflicts, key=stable_json), sorted(aliases, key=stable_json)


def classify_applet(applet: dict[str, Any], structure_conflict: bool) -> tuple[str | None, dict[str, Any] | None]:
    if structure_conflict:
        return "conflicting_duplicate_structure", None
    components = applet.get("components")
    if not isinstance(components, list):
        return "components_not_list", None
    description = str(applet.get("description") or "").strip()
    if not description:
        return "missing_description", None

    parsed = []
    for component in components:
        if not isinstance(component, dict):
            continue
        try:
            url = canonical_component_url(component.get("url"))
            channel, kind, _ = parse_url(url)
        except ValueError:
            continue
        parsed.append({"label": clean_text(component.get("label")), "url": url, "channel": channel, "kind": kind})
    if any(item["label"] == "And" for item in parsed):
        return "contains_and", None
    if any(item["label"] == "With" for item in parsed):
        return "contains_with", None
    triggers = [item for item in parsed if item["label"] == "If" and item["kind"] == "trigger"]
    actions = [item for item in parsed if item["label"] == "Then" and item["kind"] == "action"]
    if len(triggers) != 1 or len(actions) != 1:
        return "not_exactly_one_if_then", None
    chosen_urls = {triggers[0]["url"], actions[0]["url"]}
    if any(item["url"] not in chosen_urls for item in parsed):
        return "extra_component", None
    normalised = query_key(description)
    if not normalised:
        return "no_unicode_alnum_query", None
    return None, {
        "applet_url": applet["applet_url"],
        "query": description,
        "query_norm": normalised,
        "additional_description": str(applet.get("additional_description") or "").strip(),
        "trigger_url": triggers[0]["url"],
        "action_url": actions[0]["url"],
        "trigger_channel": triggers[0]["channel"],
        "action_channel": actions[0]["channel"],
        "user_counts": applet.get("all_user_counts") or [],
        "quality_flags": query_flags(description),
    }


class UnionFind:
    def __init__(self, keys: list[str]):
        self.parent = {key: key for key in keys}

    def find(self, key: str) -> str:
        while self.parent[key] != key:
            self.parent[key] = self.parent[self.parent[key]]
            key = self.parent[key]
        return key

    def union(self, left: str, right: str) -> None:
        a, b = self.find(left), self.find(right)
        if a == b:
            return
        low, high = sorted((a, b))
        self.parent[high] = low


def expand_alias_pairs(
    pairs: set[tuple[str, str]], alias_map: dict[str, list[str]]
) -> set[tuple[str, str]]:
    out = set()
    for trigger, action in pairs:
        for trigger_alias in alias_map.get(trigger, [trigger]):
            for action_alias in alias_map.get(action, [action]):
                out.add((trigger_alias, action_alias))
    return out


def make_groups(
    included: list[dict[str, Any]], aliases: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    alias_map = {url: item["urls"] for item in aliases for url in item["urls"]}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in included:
        grouped[row["query_norm"]].append(row)

    groups: dict[str, dict[str, Any]] = {}
    for normalised in sorted(grouped):
        rows = sorted(grouped[normalised], key=lambda row: (row["applet_url"], row["query"]))
        observed = {(row["trigger_url"], row["action_url"]) for row in rows}
        valid = expand_alias_pairs(observed, alias_map)
        variants = sorted({row["query"] for row in rows})
        additional = sorted({
            str(row["additional_description"]).strip() for row in rows
            if str(row.get("additional_description") or "").strip()
            and query_key(row["additional_description"]) != normalised
        })
        representative = stable_mode([row["query"] for row in rows]) or variants[0]
        gid = group_id(normalised)
        valid_pairs = [{"trigger_url": trigger, "action_url": action} for trigger, action in sorted(valid)]
        observed_pairs = [{"trigger_url": trigger, "action_url": action} for trigger, action in sorted(observed)]
        trigger_urls = sorted({trigger for trigger, _ in valid})
        action_urls = sorted({action for _, action in valid})
        channel_pairs = sorted({
            (parse_url(trigger)[0], parse_url(action)[0]) for trigger, action in valid
        })
        primary = valid_pairs[0]
        groups[gid] = {
            "group_id": gid,
            "query": representative,
            "query_norm": normalised,
            "query_variants": variants,
            "additional_descriptions": additional,
            "source_applet_urls": sorted({row["applet_url"] for row in rows}),
            "observed_pairs": observed_pairs,
            "valid_pairs": valid_pairs,
            "gold_trigger_urls": trigger_urls,
            "gold_action_urls": action_urls,
            "gold_channel_pairs": [
                {"trigger_channel": trigger, "action_channel": action}
                for trigger, action in channel_pairs
            ],
            "trigger_url": primary["trigger_url"],
            "action_url": primary["action_url"],
            "trigger_channel": parse_url(primary["trigger_url"])[0],
            "action_channel": parse_url(primary["action_url"])[0],
            "applet_url": sorted({row["applet_url"] for row in rows})[0],
            "quality_flags": sorted({flag for row in rows for flag in row["quality_flags"]}),
            "multi_gold": len(valid_pairs) > 1,
            "examples": rows,
        }

    ids = sorted(groups)
    union = UnionFind(ids)
    paraphrase_links: list[dict[str, Any]] = []
    hard_edges: list[dict[str, Any]] = []

    def side_compatible(left: dict[str, Any], right: dict[str, Any]) -> bool:
        return (
            set(left["gold_trigger_urls"]) == set(right["gold_trigger_urls"])
            or set(left["gold_action_urls"]) == set(right["gold_action_urls"])
        )

    # Exact additional-description links. Generic/conflicting shared descriptions
    # are logged but deliberately do not merge labels.
    text_index: dict[str, set[str]] = defaultdict(set)
    for gid, group in groups.items():
        text_index[group["query_norm"]].add(gid)
        for description in group["additional_descriptions"]:
            key = query_key(description)
            if key:
                text_index[key].add(gid)
    for key in sorted(text_index):
        members = sorted(text_index[key])
        for pos, left_id in enumerate(members):
            for right_id in members[pos + 1:]:
                left, right = groups[left_id], groups[right_id]
                edge = {"left": left_id, "right": right_id, "reason": "shared_primary_or_additional", "text_norm": key}
                if side_compatible(left, right):
                    union.union(left_id, right_id)
                    paraphrase_links.append(edge)
                else:
                    hard_edges.append(edge)

    # Exhaustive inverted-index near comparison. No top-N cap and no deletion.
    token_sets = {gid: content_tokens(groups[gid]["query_norm"]) for gid in ids}
    inverted: dict[str, set[str]] = defaultdict(set)
    for gid, tokens in token_sets.items():
        for token in tokens:
            inverted[token].add(gid)
    seen_pairs = set()
    for left_id in ids:
        left_tokens = token_sets[left_id]
        if len(left_tokens) < 3:
            continue
        candidates = set().union(*(inverted[token] for token in left_tokens)) if left_tokens else set()
        for right_id in sorted(candidate for candidate in candidates if candidate > left_id):
            pair_key = (left_id, right_id)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            right_tokens = token_sets[right_id]
            if len(right_tokens) < 3:
                continue
            similarity = jaccard(left_tokens, right_tokens)
            if similarity < PARAPHRASE_THRESHOLD:
                continue
            edge = {
                "left": left_id,
                "right": right_id,
                "reason": "token_jaccard",
                "similarity": round(similarity, 12),
            }
            if side_compatible(groups[left_id], groups[right_id]):
                union.union(left_id, right_id)
                paraphrase_links.append(edge)
            else:
                hard_edges.append(edge)

    families: dict[str, list[str]] = defaultdict(list)
    for gid in ids:
        families[union.find(gid)].append(gid)
    for members in families.values():
        fid = family_id(members)
        for gid in members:
            groups[gid]["semantic_family_id"] = fid
    return [groups[gid] for gid in ids], sorted(paraphrase_links, key=stable_json), sorted(hard_edges, key=stable_json)


def assign_splits(groups: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    families: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for group in groups:
        families[group["semantic_family_id"]].append(group)
    ranked = sorted(
        families,
        key=lambda fid: sha256_bytes(f"{SEED}:{fid}".encode("utf-8")),
    )
    total = len(groups)
    targets = {
        "test": round(total * SPLIT_RATIOS["test"]),
        "dev": round(total * SPLIT_RATIOS["dev"]),
        "reranker_train": round(total * SPLIT_RATIOS["reranker_train"]),
    }
    output: dict[str, list[dict[str, Any]]] = {key: [] for key in SPLIT_RATIOS}
    role_order = ["test", "dev", "reranker_train"]
    role_index = 0
    for fid in ranked:
        members = families[fid]
        while role_index < len(role_order) and len(output[role_order[role_index]]) >= targets[role_order[role_index]]:
            role_index += 1
        role = role_order[role_index] if role_index < len(role_order) else "encoder_train"
        output[role].extend(members)
    for role in output:
        output[role].sort(key=lambda group: group["group_id"])
        for group in output[role]:
            group["split"] = role
    return output


def build_pairs(
    encoder_groups: list[dict[str, Any]], corpus: dict[str, list[dict[str, Any]]]
) -> dict[str, list[dict[str, Any]]]:
    lookup = {
        kind: {record["url"]: record for record in corpus[kind]}
        for kind in ("trigger", "action")
    }
    outputs: dict[str, list[dict[str, Any]]] = {}
    for kind in ("trigger", "action"):
        for view in ("plain", "schema"):
            rows = []
            for group in encoder_groups:
                labels = group[f"gold_{kind}_urls"]
                for label in labels:
                    record = lookup[kind][label]
                    rows.append({
                        "group_id": group["group_id"],
                        "anchor": group["query"],
                        "positive": record[f"text_{view}"],
                        "label_url": label,
                        "valid_label_urls": labels,
                        "source_applet_urls": group["source_applet_urls"],
                        "view": view,
                    })
            rows.sort(key=lambda row: (row["group_id"], row["label_url"]))
            outputs[f"{kind}_encoder_train_{view}"] = rows
    return outputs


def distribution_summary(counter: Counter[str]) -> dict[str, Any]:
    counts = sorted(counter.values())
    total = sum(counts)
    classes = len(counts)
    if not counts:
        return {"classes": 0, "incidences": 0}
    gini_numerator = sum((2 * index - classes - 1) * value for index, value in enumerate(counts, start=1))
    gini = gini_numerator / (classes * total) if total else 0.0
    probabilities = [value / total for value in counts]
    entropy = -sum(value * math.log(value) for value in probabilities if value)
    descending = sorted(counts, reverse=True)
    return {
        "classes": classes,
        "incidences": total,
        "singleton_classes": sum(value == 1 for value in counts),
        "doubleton_classes": sum(value == 2 for value in counts),
        "median_frequency": median(counts),
        "max_frequency": max(counts),
        "top1_share": descending[0] / total,
        "top10_share": sum(descending[:10]) / total,
        "top50_share": sum(descending[:50]) / total,
        "gini": gini,
        "normalised_entropy": entropy / math.log(classes) if classes > 1 else 1.0,
        "effective_classes": math.exp(entropy),
    }


def build_statistics(
    splits: dict[str, list[dict[str, Any]]], corpus: dict[str, list[dict[str, Any]]]
) -> dict[str, Any]:
    lookup = {
        kind: {record["url"]: record for record in corpus[kind]}
        for kind in ("trigger", "action")
    }
    output: dict[str, Any] = {"splits": {}}
    encoder = splits["encoder_train"]
    seen_trigger = {url for group in encoder for url in group["gold_trigger_urls"]}
    seen_action = {url for group in encoder for url in group["gold_action_urls"]}
    seen_pairs = {
        (pair["trigger_url"], pair["action_url"])
        for group in encoder for pair in group["valid_pairs"]
    }

    for role, groups in splits.items():
        trigger_counts = Counter(url for group in groups for url in group["gold_trigger_urls"])
        action_counts = Counter(url for group in groups for url in group["gold_action_urls"])
        pair_counts = Counter(
            (pair["trigger_url"], pair["action_url"])
            for group in groups for pair in group["valid_pairs"]
        )
        token_lengths = sorted(len(group["query_norm"].split()) for group in groups)
        both_explicit = neither_explicit = 0
        unseen_trigger_rows = unseen_action_rows = unseen_pair_rows = 0
        for group in groups:
            query = f" {group['query_norm']} "
            trigger_channels = {row["trigger_channel"] for row in group["gold_channel_pairs"]}
            action_channels = {row["action_channel"] for row in group["gold_channel_pairs"]}
            trigger_named = any(f" {query_key(channel)} " in query for channel in trigger_channels)
            action_named = any(f" {query_key(channel)} " in query for channel in action_channels)
            both_explicit += trigger_named and action_named
            neither_explicit += not trigger_named and not action_named
            unseen_trigger_rows += any(url not in seen_trigger for url in group["gold_trigger_urls"])
            unseen_action_rows += any(url not in seen_action for url in group["gold_action_urls"])
            unseen_pair_rows += all(
                (pair["trigger_url"], pair["action_url"]) not in seen_pairs
                for pair in group["valid_pairs"]
            )
        output["splits"][role] = {
            "groups": len(groups),
            "source_applets": sum(len(group["source_applet_urls"]) for group in groups),
            "multi_gold_groups": sum(len(group["valid_pairs"]) > 1 for group in groups),
            "query_tokens": {
                "min": min(token_lengths),
                "median": median(token_lengths),
                "max": max(token_lengths),
                "at_most_2": sum(value <= 2 for value in token_lengths),
            },
            "service_name_signal": {
                "both_explicit": both_explicit,
                "neither_explicit": neither_explicit,
            },
            "trigger_functions": distribution_summary(trigger_counts),
            "action_functions": distribution_summary(action_counts),
            "function_pairs": distribution_summary(pair_counts),
            "relative_to_encoder_train": {
                "rows_with_unseen_trigger": unseen_trigger_rows if role != "encoder_train" else 0,
                "rows_with_unseen_action": unseen_action_rows if role != "encoder_train" else 0,
                "rows_with_all_pairs_unseen": unseen_pair_rows if role != "encoder_train" else 0,
            },
        }
    output["corpus"] = {
        kind: {
            "functions": len(records),
            "channels": len({record["channel"] for record in records}),
            "schema_informative": sum(record["text_plain"] != record["text_schema"] for record in records),
            "missing_api_info": sum(record["schema_status"] == "missing_api_info" for record in records),
            "critical_conflicts_resolved": sum(record["schema_status"] == "critical_conflict_resolved" for record in records),
        }
        for kind, records in corpus.items()
    }
    return output


def git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def artifact_info(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8")) if path.suffix == ".json" else None
    if isinstance(value, list):
        rows = len(value)
    elif isinstance(value, dict):
        rows = len(value)
    else:
        rows = None
    return {"sha256": sha256_file(path), "bytes": path.stat().st_size, "rows": rows}


def build(raw_path: Path, out: Path, replace: bool = False) -> dict[str, Any]:
    raw_path = raw_path.resolve()
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise TypeError("raw dataset must be a JSON list")

    applets, raw_audit = collect_raw(raw)
    conflict_urls = {item["applet_url"] for item in raw_audit["duplicate_structure_conflicts"]}
    corpus_by_kind, schema_conflicts, aliases = build_corpus(applets, raw_audit)

    included: list[dict[str, Any]] = []
    excluded = []
    disposition = Counter()
    for applet_url in sorted(applets):
        reason, record = classify_applet(applets[applet_url], applet_url in conflict_urls)
        if reason:
            disposition[reason] += 1
            excluded.append({"applet_url": applet_url, "reason": reason})
        else:
            disposition["included"] += 1
            included.append(record)
    if sum(disposition.values()) != len(applets):
        raise AssertionError("terminal applet disposition does not reconcile")

    groups, paraphrase_links, hard_edges = make_groups(included, aliases)
    splits = assign_splits(groups)
    pairs = build_pairs(splits["encoder_train"], corpus_by_kind)
    statistics = build_statistics(splits, corpus_by_kind)

    config = {
        "dataset_version": DATASET_VERSION,
        "normalizer": NORMALIZER,
        "scope": "exactly one If trigger and one Then action; no And; no With",
        "split_seed": SEED,
        "split_ratios": SPLIT_RATIOS,
        "paraphrase": {
            "threshold": PARAPHRASE_THRESHOLD,
            "minimum_content_tokens": 3,
            "link_policy": "equal trigger set OR equal action set; never delete different-label edges",
        },
        "query_source": "description only; additional_description metadata is not a baseline anchor",
        "documents": {
            "plain": "channel + function name + function description",
            "schema": "plain prefix + canonical input fields/ingredients",
            "excluded": ["category", "title", "related_services", "popularity", "examples", "full filter-code paths"],
        },
        "schema_conflict_policy": "stable mode over one semantic entry per unique applet; record all variants",
    }

    builder_path = Path(__file__).resolve()
    module_path = ROOT / "farm/dataset_v2.py"
    raw_sha = sha256_file(raw_path)
    recipe = {
        "raw_sha256": raw_sha,
        "builder_sha256": sha256_file(builder_path),
        "module_sha256": sha256_file(module_path),
        "config": config,
    }
    recipe_id = sha256_bytes(stable_json(recipe).encode("utf-8"))

    out_parent = out.resolve().parent
    out_parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".v2-build-", dir=out_parent))
    try:
        for kind, filename in CORPUS_FILENAMES.items():
            write_json(temporary / "corpus" / filename, corpus_by_kind.get(kind, []))
        for role, rows in splits.items():
            write_json(temporary / f"splits/{role}.json", rows)
        for name, rows in pairs.items():
            write_json(temporary / f"pairs/{name}.json", rows)
        write_json(temporary / "audit/excluded_applets.json", excluded)
        write_json(temporary / "audit/schema_conflicts.json", schema_conflicts)
        write_json(temporary / "audit/alias_sets.json", aliases)
        write_json(temporary / "audit/paraphrase_links.json", paraphrase_links)
        write_json(temporary / "audit/different_label_hard_edges.json", hard_edges)
        write_json(temporary / "audit/statistics.json", statistics)

        artifact_paths = sorted(
            path for path in temporary.rglob("*") if path.is_file() and path.name not in {"manifest.json", "checksums.sha256"}
        )
        artifacts = {str(path.relative_to(temporary)): artifact_info(path) for path in artifact_paths}
        content_id = sha256_bytes(stable_json({"recipe_id": recipe_id, "artifacts": artifacts}).encode("utf-8"))
        checksums = "".join(f"{info['sha256']}  {name}\n" for name, info in sorted(artifacts.items()))
        (temporary / "checksums.sha256").write_text(checksums, encoding="utf-8")
        artifacts["checksums.sha256"] = {
            "sha256": sha256_file(temporary / "checksums.sha256"),
            "bytes": (temporary / "checksums.sha256").stat().st_size,
            "rows": len(artifact_paths),
        }

        manifest = {
            "dataset_version": DATASET_VERSION,
            "dataset_id": content_id,
            "recipe_id": recipe_id,
            "source": {
                "path": "data/iftttt_dataset_full_trigger_action.json",
                "sha256": raw_sha,
                "bytes": raw_path.stat().st_size,
                "repository_head": git_head(),
            },
            "code": {
                "builder": {"path": "build_dataset_v2.py", "sha256": recipe["builder_sha256"]},
                "module": {"path": "farm/dataset_v2.py", "sha256": recipe["module_sha256"]},
            },
            "config": config,
            "counts": {
                **{key: raw_audit[key] for key in (
                    "raw_service_records", "raw_applet_occurrences", "unique_applet_urls",
                    "duplicate_occurrences", "missing_url_occurrences", "invalid_applet_objects",
                )},
                "applet_disposition": dict(sorted(disposition.items())),
                "included_applets": len(included),
                "query_groups": len(groups),
                "observed_multi_gold_groups": sum(len(group["observed_pairs"]) > 1 for group in groups),
                "observed_multi_gold_applets": sum(
                    len(group["source_applet_urls"]) for group in groups if len(group["observed_pairs"]) > 1
                ),
                "multi_gold_groups": sum(group["multi_gold"] for group in groups),
                "multi_gold_applets": sum(len(group["source_applet_urls"]) for group in groups if group["multi_gold"]),
                "semantic_families": len({group["semantic_family_id"] for group in groups}),
                "split_groups": {role: len(rows) for role, rows in splits.items()},
                "corpus": {kind: len(rows) for kind, rows in corpus_by_kind.items()},
                "schema_conflicts": len(schema_conflicts),
                "alias_sets": len(aliases),
                "paraphrase_links": len(paraphrase_links),
                "different_label_hard_edges": len(hard_edges),
                "pair_rows": {name: len(rows) for name, rows in pairs.items()},
            },
            "provenance": raw_audit["provenance"],
            "metadata_conflict_urls": raw_audit["metadata_conflict_urls"],
            "artifacts": dict(sorted(artifacts.items())),
        }
        write_json(temporary / "manifest.json", manifest)

        target = out.resolve()
        if target.exists():
            existing_manifest = target / "manifest.json"
            if existing_manifest.exists() and json.loads(existing_manifest.read_text())["dataset_id"] == content_id:
                shutil.rmtree(temporary)
                return json.loads(existing_manifest.read_text())
            if not replace:
                raise FileExistsError(f"{target} exists with different content; pass --replace explicitly")
            backup = target.with_name(target.name + ".previous")
            if backup.exists():
                raise FileExistsError(f"refusing to overwrite existing backup {backup}")
            os.replace(target, backup)
        os.replace(temporary, target)
        return manifest
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    manifest = build(args.raw, args.out, replace=args.replace)
    print(json.dumps({
        "dataset_id": manifest["dataset_id"],
        "counts": manifest["counts"],
        "out": str(args.out),
    }, indent=2))


if __name__ == "__main__":
    main()
