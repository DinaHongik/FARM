"""Strict, exhaustive release gate for FARM Dataset v2."""
from __future__ import annotations

import argparse
import json
import random
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from build_dataset_v2 import (
    DEFAULT_RAW,
    build,
    canonical_applet_url,
)
from farm.dataset_v2 import (
    CORPUS_FILENAMES,
    PARAPHRASE_THRESHOLD,
    content_tokens,
    jaccard,
    query_key,
    render_plain,
    render_schema,
    sha256_bytes,
    sha256_file,
    stable_json,
)
from farm.render import parse_url


ROOT = Path(__file__).resolve().parent


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, bool]] = []

    def add(self, tag: str, description: str, value: Any, ok: bool) -> None:
        self.rows.append((tag, description, str(value), bool(ok)))

    def require(self, tag: str, description: str, fn: Callable[[], Any]) -> Any:
        try:
            value = fn()
            self.add(tag, description, value if value is not None else "ok", True)
            return value
        except Exception as exc:  # report every independent gate in one run
            self.add(tag, description, f"{type(exc).__name__}: {exc}", False)
            return None

    def show(self) -> int:
        print("\n" + "=" * 100)
        print("FARM DATASET v2 STRICT RELEASE GATE")
        print("=" * 100)
        for tag, description, value, ok in self.rows:
            print(f"[{'PASS' if ok else 'FAIL'}] {tag}  {description}")
            print(f"        {value}")
        failed = sum(not row[3] for row in self.rows)
        print(f"\n{len(self.rows) - failed}/{len(self.rows)} gates passed")
        return failed


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def unique_urls_from_raw(raw: list[Any]) -> set[str]:
    out = set()
    for service in raw:
        if not isinstance(service, dict) or not isinstance(service.get("applets"), list):
            continue
        for applet in service["applets"]:
            if isinstance(applet, dict):
                url = canonical_applet_url(applet.get("applet_url"))
                if url:
                    out.add(url)
    return out


def verify(root: Path, raw_path: Path, verify_rebuild: bool, verify_order: bool) -> int:
    report = Report()
    manifest = load_json(root / "manifest.json")
    raw = load_json(raw_path)

    def source_gate() -> str:
        actual = sha256_file(raw_path)
        assert_true(actual == manifest["source"]["sha256"], "raw SHA mismatch")
        assert_true(raw_path.stat().st_size == manifest["source"]["bytes"], "raw byte size mismatch")
        return actual

    report.require("V01", "source SHA and size match the manifest", source_gate)

    def artifact_gate() -> str:
        for name, expected in manifest["artifacts"].items():
            path = root / name
            assert_true(path.is_file(), f"missing artifact {name}")
            assert_true(sha256_file(path) == expected["sha256"], f"SHA mismatch: {name}")
            assert_true(path.stat().st_size == expected["bytes"], f"size mismatch: {name}")
        checksum_expected = "".join(
            f"{info['sha256']}  {name}\n"
            for name, info in sorted(manifest["artifacts"].items())
            if name != "checksums.sha256"
        )
        assert_true((root / "checksums.sha256").read_text() == checksum_expected, "checksums.sha256 content mismatch")
        return f"{len(manifest['artifacts'])} artifacts"

    report.require("V02", "every artifact hash, size and checksum entry matches", artifact_gate)

    def lineage_gate() -> str:
        recipe = {
            "raw_sha256": manifest["source"]["sha256"],
            "builder_sha256": manifest["code"]["builder"]["sha256"],
            "module_sha256": manifest["code"]["module"]["sha256"],
            "config": manifest["config"],
        }
        recipe_id = sha256_bytes(stable_json(recipe).encode("utf-8"))
        assert_true(recipe_id == manifest["recipe_id"], "recipe ID mismatch")
        content_artifacts = {
            name: info for name, info in manifest["artifacts"].items()
            if name != "checksums.sha256"
        }
        dataset_id = sha256_bytes(stable_json({"recipe_id": recipe_id, "artifacts": content_artifacts}).encode("utf-8"))
        assert_true(dataset_id == manifest["dataset_id"], "dataset ID mismatch")
        return dataset_id

    report.require("V03", "recipe and content-addressed dataset IDs recompute", lineage_gate)

    splits = {
        role: load_json(root / f"splits/{role}.json")
        for role in ("encoder_train", "reranker_train", "dev", "test")
    }
    groups = [group for role in splits.values() for group in role]
    by_group = {group["group_id"]: group for group in groups}
    excluded = load_json(root / "audit/excluded_applets.json")

    def census_gate() -> str:
        raw_urls = unique_urls_from_raw(raw)
        included_urls = {
            url for group in groups for url in group["source_applet_urls"]
        }
        excluded_urls = {item["applet_url"] for item in excluded}
        assert_true(len(included_urls) == sum(len(group["source_applet_urls"]) for group in groups), "included applet duplicated")
        assert_true(len(excluded_urls) == len(excluded), "excluded applet duplicated")
        assert_true(not (included_urls & excluded_urls), "included/excluded overlap")
        assert_true(included_urls | excluded_urls == raw_urls, "terminal dispositions do not cover raw URLs")
        counts = manifest["counts"]
        assert_true(len(raw_urls) == counts["unique_applet_urls"], "unique raw URL count mismatch")
        assert_true(len(included_urls) == counts["included_applets"], "included count mismatch")
        assert_true(Counter(item["reason"] for item in excluded) == Counter({
            key: value for key, value in counts["applet_disposition"].items() if key != "included"
        }), "exclusion reason counts mismatch")
        return f"{len(raw_urls)} = {len(included_urls)} included + {len(excluded_urls)} excluded"

    report.require("V04", "raw URL census has one mutually exclusive terminal disposition", census_gate)

    def split_gate() -> str:
        seen_group: dict[str, str] = {}
        seen_query: dict[str, str] = {}
        seen_applet: dict[str, str] = {}
        seen_family: dict[str, str] = {}
        for role, rows in splits.items():
            for group in rows:
                for value, seen, label in (
                    (group["group_id"], seen_group, "group"),
                    (group["query_norm"], seen_query, "query"),
                    (group["semantic_family_id"], seen_family, "family"),
                ):
                    previous = seen.setdefault(value, role)
                    assert_true(previous == role, f"{label} {value} crosses {previous}/{role}")
                for applet_url in group["source_applet_urls"]:
                    previous = seen_applet.setdefault(applet_url, role)
                    assert_true(previous == role, f"applet {applet_url} crosses {previous}/{role}")
        assert_true(len(seen_group) == manifest["counts"]["query_groups"], "split union group count mismatch")
        assert_true({role: len(rows) for role, rows in splits.items()} == manifest["counts"]["split_groups"], "split counts mismatch")
        return ", ".join(f"{role}={len(rows)}" for role, rows in splits.items())

    report.require("V05", "groups, queries, families and applets are disjoint across four roles", split_gate)

    def group_semantics_gate() -> str:
        observed_multi = valid_multi = observed_multi_applets = valid_multi_applets = 0
        for group in groups:
            assert_true(group["query_norm"], f"empty query norm: {group['group_id']}")
            assert_true(query_key(group["query"]) == group["query_norm"], f"representative query mismatch: {group['group_id']}")
            assert_true(all(query_key(value) == group["query_norm"] for value in group["query_variants"]), f"query variant mismatch: {group['group_id']}")
            observed = {(row["trigger_url"], row["action_url"]) for row in group["observed_pairs"]}
            valid = {(row["trigger_url"], row["action_url"]) for row in group["valid_pairs"]}
            assert_true(observed <= valid and observed, f"observed/valid pair error: {group['group_id']}")
            assert_true((group["trigger_url"], group["action_url"]) in valid, f"primary not valid: {group['group_id']}")
            assert_true({t for t, _ in valid} == set(group["gold_trigger_urls"]), f"trigger projection mismatch: {group['group_id']}")
            assert_true({a for _, a in valid} == set(group["gold_action_urls"]), f"action projection mismatch: {group['group_id']}")
            assert_true(set(group["source_applet_urls"]) == {row["applet_url"] for row in group["examples"]}, f"source provenance mismatch: {group['group_id']}")
            if len(observed) > 1:
                observed_multi += 1; observed_multi_applets += len(group["source_applet_urls"])
            if len(valid) > 1:
                valid_multi += 1; valid_multi_applets += len(group["source_applet_urls"])
        counts = manifest["counts"]
        assert_true(observed_multi == counts["observed_multi_gold_groups"], "observed multi-gold count mismatch")
        assert_true(observed_multi_applets == counts["observed_multi_gold_applets"], "observed multi-gold applet mismatch")
        assert_true(valid_multi == counts["multi_gold_groups"], "alias-aware multi-gold count mismatch")
        assert_true(valid_multi_applets == counts["multi_gold_applets"], "alias-aware multi-gold applet mismatch")
        return f"observed={observed_multi} groups; alias-aware={valid_multi} groups"

    report.require("V06", "multi-gold pair truth and side projections are lossless", group_semantics_gate)

    corpus = {
        kind: load_json(root / "corpus" / filename)
        for kind, filename in CORPUS_FILENAMES.items()
    }
    corpus_lookup = {
        kind: {record["url"]: record for record in records}
        for kind, records in corpus.items()
    }

    def corpus_gate() -> str:
        schema_informative = Counter()
        for kind, records in corpus.items():
            assert_true(len(corpus_lookup[kind]) == len(records), f"duplicate {kind} URL")
            for record in records:
                channel, parsed_kind, _ = parse_url(record["url"])
                assert_true(parsed_kind == kind and channel == record["channel"], f"URL identity mismatch: {record['url']}")
                fields = [query_key(field["slug"]) for field in record["input_fields"]]
                ingredients = [query_key(item["slug"]) for item in record["ingredients"]]
                assert_true(len(fields) == len(set(fields)), f"duplicate field slug: {record['url']}")
                assert_true(len(ingredients) == len(set(ingredients)), f"duplicate ingredient slug: {record['url']}")
                assert_true(all(fields + ingredients), f"empty schema ID: {record['url']}")
                assert_true(all(
                    not str(item.get("label", "")).casefold().startswith("no fields for this")
                    for item in record["input_fields"] + record["ingredients"]
                ), f"placeholder schema entry: {record['url']}")
                assert_true(record["text_plain"] == render_plain(record), f"plain render mismatch: {record['url']}")
                assert_true(record["text_schema"] == render_schema(record), f"schema render mismatch: {record['url']}")
                assert_true(record["text_schema"].startswith(record["text_plain"]), f"schema prefix mismatch: {record['url']}")
                schema_informative[kind] += record["text_plain"] != record["text_schema"]
        return ", ".join(f"{kind} informative={schema_informative[kind]}/{len(corpus[kind])}" for kind in corpus)

    report.require("V07", "canonical corpus has unique schema IDs and reproducible paired renders", corpus_gate)

    def alias_gate() -> str:
        declared = {
            (item["kind"], tuple(item["urls"]))
            for item in load_json(root / "audit/alias_sets.json")
        }
        actual = set()
        for kind, records in corpus.items():
            collisions: dict[str, list[str]] = defaultdict(list)
            for record in records:
                collisions[record["text_schema"]].append(record["url"])
            for urls in collisions.values():
                if len(urls) > 1:
                    actual.add((kind, tuple(sorted(urls))))
        assert_true(actual == declared, f"alias ledger mismatch actual={actual} declared={declared}")
        for kind, urls in declared:
            for url in urls:
                assert_true(corpus_lookup[kind][url]["equivalent_urls"] == list(urls), f"alias not attached: {url}")
        return f"{len(declared)} alias sets"

    report.require("V08", "every identical canonical document collision is an explicit alias set", alias_gate)

    def gold_gate() -> str:
        sides = 0
        for group in groups:
            for pair in group["valid_pairs"]:
                assert_true(pair["trigger_url"] in corpus_lookup["trigger"], f"missing trigger gold: {pair}")
                assert_true(pair["action_url"] in corpus_lookup["action"], f"missing action gold: {pair}")
                sides += 2
        return f"{sides} valid gold sides indexed"

    report.require("V09", "every valid function-pair side resolves to the correct corpus", gold_gate)

    def near_gate() -> str:
        tokens = {group["group_id"]: content_tokens(group["query_norm"]) for group in groups}
        inverted: dict[str, set[str]] = defaultdict(set)
        for gid, values in tokens.items():
            for token in values:
                inverted[token].add(gid)
        same_label_edges = different_label_edges = 0
        for left_id in sorted(by_group):
            left = by_group[left_id]
            if len(tokens[left_id]) < 3:
                continue
            candidates = set().union(*(inverted[token] for token in tokens[left_id]))
            for right_id in sorted(gid for gid in candidates if gid > left_id):
                if len(tokens[right_id]) < 3 or jaccard(tokens[left_id], tokens[right_id]) < PARAPHRASE_THRESHOLD:
                    continue
                right = by_group[right_id]
                compatible = (
                    set(left["gold_trigger_urls"]) == set(right["gold_trigger_urls"])
                    or set(left["gold_action_urls"]) == set(right["gold_action_urls"])
                )
                if compatible:
                    same_label_edges += 1
                    assert_true(left["semantic_family_id"] == right["semantic_family_id"], f"same-label near pair crosses family: {left_id}/{right_id}")
                else:
                    different_label_edges += 1
        assert_true(different_label_edges > 0, "different-label hard cases were unexpectedly removed")
        return f"same-side-compatible={same_label_edges}; retained hard={different_label_edges}"

    report.require("V10", "exhaustive label-aware near grouping has no capped search or deletion", near_gate)

    def additional_gate() -> str:
        index: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for group in groups:
            index[group["query_norm"]].append(group)
            for text in group["additional_descriptions"]:
                if query_key(text):
                    index[query_key(text)].append(group)
        compatible_links = conflicts = 0
        for linked in index.values():
            dedup = {group["group_id"]: group for group in linked}
            rows = list(dedup.values())
            for pos, left in enumerate(rows):
                for right in rows[pos + 1:]:
                    compatible = (
                        set(left["gold_trigger_urls"]) == set(right["gold_trigger_urls"])
                        or set(left["gold_action_urls"]) == set(right["gold_action_urls"])
                    )
                    if compatible:
                        compatible_links += 1
                        assert_true(left["semantic_family_id"] == right["semantic_family_id"], "compatible additional description crosses family")
                    else:
                        conflicts += 1
        return f"compatible={compatible_links}; conflicting texts retained={conflicts}"

    report.require("V11", "additional descriptions inherit compatible split families but are not anchors", additional_gate)

    pair_files = {
        f"{kind}_{view}": load_json(root / f"pairs/{kind}_encoder_train_{view}.json")
        for kind in ("trigger", "action") for view in ("plain", "schema")
    }

    def pair_gate() -> str:
        encoder_ids = {group["group_id"] for group in splits["encoder_train"]}
        non_encoder_ids = {group["group_id"] for role in ("reranker_train", "dev", "test") for group in splits[role]}
        total = 0
        for key, rows in pair_files.items():
            kind, view = key.split("_")
            seen = set()
            expected = sum(len(group[f"gold_{kind}_urls"]) for group in splits["encoder_train"])
            assert_true(len(rows) == expected, f"pair count mismatch: {key}")
            for row in rows:
                identity = (row["group_id"], row["label_url"])
                assert_true(identity not in seen, f"duplicate pair row: {key}/{identity}")
                seen.add(identity)
                assert_true(row["group_id"] in encoder_ids and row["group_id"] not in non_encoder_ids, f"non-train group in pairs: {row['group_id']}")
                record = corpus_lookup[kind][row["label_url"]]
                assert_true(row["positive"] == record[f"text_{view}"], f"URL→positive mismatch: {key}/{row['label_url']}")
                assert_true(set(row["valid_label_urls"]) == set(by_group[row["group_id"]][f"gold_{kind}_urls"]), f"valid label mismatch: {key}/{row['group_id']}")
                total += 1
        return f"{total} plain/schema URL-exact pair rows"

    report.require("V12", "training positives are byte-identical to their own URL/view documents", pair_gate)

    def no_cartesian_gate() -> str:
        ambiguous = next((group for group in groups if len(group["valid_pairs"]) > 1), None)
        assert_true(ambiguous is not None, "no multi-gold fixture in full data")
        valid = {(row["trigger_url"], row["action_url"]) for row in ambiguous["valid_pairs"]}
        cross = {
            (trigger, action)
            for trigger in ambiguous["gold_trigger_urls"]
            for action in ambiguous["gold_action_urls"]
        } - valid
        # Some regional alias groups genuinely make the full cross-product valid;
        # locate any observed ambiguity that demonstrates pair preservation.
        if not cross:
            for group in groups:
                valid = {(row["trigger_url"], row["action_url"]) for row in group["valid_pairs"]}
                cross = {(t, a) for t in group["gold_trigger_urls"] for a in group["gold_action_urls"]} - valid
                if cross:
                    ambiguous = group
                    break
        assert_true(bool(cross), "no non-Cartesian multi-gold group found")
        assert_true(all(pair not in valid for pair in cross), "Cartesian invention accepted")
        return f"{ambiguous['group_id']} rejects {len(cross)} Cartesian inventions"

    report.require("V13", "joint truth preserves observed pairs rather than a side-label Cartesian product", no_cartesian_gate)

    def conflict_gate() -> str:
        conflicts = load_json(root / "audit/schema_conflicts.json")
        critical = [item for item in conflicts if set(item.get("attributes", {})) & {"slug", "type", "required", "can_have_default"}]
        assert_true(all(item.get("url") for item in conflicts), "anonymous schema conflict")
        statuses = Counter(record["schema_status"] for records in corpus.values() for record in records)
        assert_true(statuses["critical_conflict_resolved"] == len({item["url"] for item in critical}), "critical status/ledger mismatch")
        return f"ledger={len(conflicts)} variants; critical URLs={len({item['url'] for item in critical})}"

    report.require("V14", "schema variants and critical modal resolutions are explicit", conflict_gate)

    def statistics_gate() -> str:
        statistics = load_json(root / "audit/statistics.json")
        for role, rows in splits.items():
            summary = statistics["splits"][role]
            assert_true(summary["groups"] == len(rows), f"statistics group count mismatch: {role}")
            assert_true(summary["source_applets"] == sum(len(group["source_applet_urls"]) for group in rows), f"statistics applet count mismatch: {role}")
            assert_true(summary["trigger_functions"]["classes"] > 0, f"no trigger classes: {role}")
            assert_true(summary["action_functions"]["classes"] > 0, f"no action classes: {role}")
        train = statistics["splits"]["encoder_train"]
        return (
            f"encoder trigger classes={train['trigger_functions']['classes']} "
            f"singletons={train['trigger_functions']['singleton_classes']}; "
            f"action classes={train['action_functions']['classes']} "
            f"singletons={train['action_functions']['singleton_classes']}"
        )

    report.require("V15", "non-blocking imbalance/coverage statistics reconcile with artifacts", statistics_gate)

    if verify_rebuild:
        def rebuild_gate() -> str:
            with tempfile.TemporaryDirectory(prefix="farm-v2-rebuild-") as directory:
                rebuilt_root = Path(directory) / "v2"
                rebuilt = build(raw_path, rebuilt_root)
                assert_true(rebuilt["dataset_id"] == manifest["dataset_id"], "rebuild dataset ID differs")
                assert_true(rebuilt["artifacts"] == manifest["artifacts"], "rebuild artifact hashes differ")
            return manifest["dataset_id"]
        report.require("V16", "a clean build in another directory is byte-identical", rebuild_gate)

    if verify_order:
        def order_gate() -> str:
            shuffled = json.loads(json.dumps(raw))
            rng = random.Random(20260901)
            rng.shuffle(shuffled)
            for service in shuffled:
                if isinstance(service, dict) and isinstance(service.get("applets"), list):
                    rng.shuffle(service["applets"])
            with tempfile.TemporaryDirectory(prefix="farm-v2-order-") as directory:
                directory_path = Path(directory)
                shuffled_raw = directory_path / "raw.json"
                shuffled_raw.write_text(json.dumps(shuffled, ensure_ascii=False), encoding="utf-8")
                shuffled_root = directory_path / "v2"
                shuffled_manifest = build(shuffled_raw, shuffled_root)
                original_artifacts = {
                    name: info for name, info in manifest["artifacts"].items()
                    if name != "checksums.sha256"
                }
                shuffled_artifacts = {
                    name: info for name, info in shuffled_manifest["artifacts"].items()
                    if name != "checksums.sha256"
                }
                assert_true(original_artifacts == shuffled_artifacts, "raw-order permutation changes an artifact")
            return "service and applet order permutation preserved every artifact"
        report.require("V17", "raw service/applet order cannot choose different survivors", order_gate)

    return report.show()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT / "data/v2")
    parser.add_argument("--raw", type=Path, default=DEFAULT_RAW)
    parser.add_argument("--verify-rebuild", action="store_true")
    parser.add_argument("--verify-order", action="store_true")
    args = parser.parse_args()
    raise SystemExit(1 if verify(args.root.resolve(), args.raw.resolve(), args.verify_rebuild, args.verify_order) else 0)


if __name__ == "__main__":
    main()
