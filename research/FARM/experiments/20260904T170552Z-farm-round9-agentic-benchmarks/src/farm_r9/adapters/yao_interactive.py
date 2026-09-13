"""JSON-only adapter for the pinned Yao et al. Interactive IFTTT artifact."""
from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Any

from farm_r9.artifact_io import read_json, sha256_file, sha256_text


YAO_REPOSITORY_COMMIT = "cd190229b0b6f237fd3534a297d138c2d834168d"
YAO_PICKLE_SHA256 = "3d47f7047808085c75826d90997f05498fae4747fd3d78e6130e93742806a535"
YAO_CONVERTED_SCHEMA = "yao-interactive-converted-v1"
YAO_BENCHMARK = "interactive_ifttt"
YAO_POPULATION_SIZE = 3870
YAO_SAMPLE_SIZE = 150
YAO_TAG_COUNTS = {"CI": 727, "VI-1": 453, "VI-2": 818, "VI-3": 272, "VI-4": 1600}
YAO_SAMPLE_QUOTAS = {"CI": 28, "VI-1": 18, "VI-2": 32, "VI-3": 10, "VI-4": 62}
_ANSWER_HASH_NAMESPACE = "farm-round9-answer-v1"
_SAMPLE_HASH_NAMESPACE = "farm-round9-yao-v1"
_COMPONENTS = (
    "trigger_channel",
    "trigger_function",
    "action_channel",
    "action_function",
)
_CONSTRAINT_KEYS = {
    "trigger_functions_for_service": ("trigger_function",),
    "trigger_services_for_function": ("trigger_channel",),
    "action_functions_for_service": ("action_function",),
    "action_services_for_function": ("action_channel",),
}
_LABEL_TYPES = ("trigger_chans", "trigger_funcs", "action_chans", "action_funcs")
_LABEL_TYPE_TO_COMPONENT = dict(zip(_LABEL_TYPES, _COMPONENTS))


def _text(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if not isinstance(value, str):
        raise ValueError(f"expected text, got {type(value).__name__}")
    # Python-2 byte strings are loaded with latin-1 by the restricted converter.
    # Recover UTF-8 where that round trip is valid; native Unicode is retained.
    try:
        return value.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return value


def _label_classes(labeler: Any) -> list[str]:
    values = getattr(labeler, "classes_", labeler)
    try:
        result = [_text(value) for value in values]
    except TypeError as error:
        raise ValueError("Yao label encoder has no iterable classes_") from error
    if not result or len(result) != len(set(result)):
        raise ValueError("Yao label classes must be nonempty and unique")
    return result


def _string_keyed_pool(pool: Any) -> dict[str, Any]:
    if not isinstance(pool, dict):
        raise ValueError("Yao simulator pool must be a dictionary")
    result: dict[str, Any] = {}
    for key, value in pool.items():
        text_key = _text(key)
        if text_key in result:
            raise ValueError(f"duplicate decoded Yao simulator key: {text_key}")
        result[text_key] = value
    return result


def _decode_answer_options(
    raw_options: Any,
    id_to_word: dict[int, str],
) -> list[str]:
    if not isinstance(raw_options, list) or not raw_options:
        raise ValueError("Yao simulator endpoint has no answer options")
    decoded: list[str] = []
    for option in raw_options:
        if not isinstance(option, (list, tuple)):
            raise ValueError("Yao simulator answer must be a token-id sequence")
        try:
            # The pinned artifact uses one out-of-vocabulary sentinel (78122)
            # which is intentionally absent from ``word_ids``.  Preserve its
            # position as inert text rather than dropping or guessing a token.
            tokens = [id_to_word.get(int(token_id), "<UNK>") for token_id in option]
        except (TypeError, ValueError) as error:
            raise ValueError("Yao simulator answer has a non-integer token ID") from error
        answer = " ".join(tokens).strip()
        if not answer:
            raise ValueError("Yao simulator answer decodes to empty text")
        decoded.append(answer)
    return decoded


def _decode_constraint(
    mapping: Any,
    key_id: int,
    value_labels: list[str],
) -> list[str]:
    if not isinstance(mapping, dict) or key_id not in mapping:
        raise ValueError(f"Yao endpoint constraint has no key {key_id}")
    try:
        decoded = [value_labels[int(value)] for value in mapping[key_id]]
    except (IndexError, KeyError, TypeError, ValueError) as error:
        raise ValueError("Yao endpoint constraint references an unknown label") from error
    if not decoded:
        raise ValueError("Yao endpoint constraint list is empty")
    return decoded


def convert_loaded_yao_data(
    data: Any,
    *,
    source_sha256: str,
    require_official_shape: bool = True,
) -> dict[str, Any]:
    """Convert an already restricted-unpickled object into inert JSON data."""
    if source_sha256 != YAO_PICKLE_SHA256:
        raise ValueError("Yao source SHA-256 does not match the pinned official artifact")
    if not isinstance(data, dict):
        raise ValueError("Yao pickle root must be a dictionary")
    required = {
        "label_types", "num_labels", "labelers", "word_ids", "train", "dev",
        "sample_dev", "test", "user_answers", "chnl_fn_constraints",
    }
    if required - set(data):
        raise ValueError(f"Yao pickle is missing keys: {sorted(required - set(data))}")
    if tuple(data["label_types"]) != _LABEL_TYPES:
        raise ValueError("Yao label_types order changed")
    if not isinstance(data["labelers"], dict):
        raise ValueError("Yao labelers must be a dictionary")
    catalogs = {
        _LABEL_TYPE_TO_COMPONENT[label_type]: _label_classes(data["labelers"][label_type])
        for label_type in _LABEL_TYPES
    }
    num_labels = [int(value) for value in data["num_labels"]]
    if num_labels != [len(catalogs[_LABEL_TYPE_TO_COMPONENT[name]]) for name in _LABEL_TYPES]:
        raise ValueError("Yao num_labels disagrees with label encoder classes")

    if not isinstance(data["word_ids"], dict):
        raise ValueError("Yao word_ids must be a dictionary")
    id_to_word: dict[int, str] = {}
    for word, token_id_value in data["word_ids"].items():
        token_id = int(token_id_value)
        if token_id in id_to_word:
            raise ValueError(f"duplicate Yao vocabulary ID: {token_id}")
        id_to_word[token_id] = _text(word)

    raw_answer_pools = data["user_answers"]
    raw_constraints = data["chnl_fn_constraints"]
    if not isinstance(raw_answer_pools, list) or len(raw_answer_pools) != 4:
        raise ValueError("Yao user_answers must contain four component pools")
    if not isinstance(raw_constraints, list) or len(raw_constraints) != 4:
        raise ValueError("Yao chnl_fn_constraints must contain four mappings")
    answer_pools = [_string_keyed_pool(pool) for pool in raw_answer_pools]

    train_ids = {str(row.get("recipe_id")) for row in data["train"]}
    dev_ids = {str(row.get("recipe_id")) for row in data["dev"]}
    converted_rows: list[dict[str, Any]] = []
    seen_test_ids: set[str] = set()
    for source_index, row in enumerate(data["test"]):
        if not isinstance(row, dict):
            raise ValueError(f"Yao test row {source_index} is not a dictionary")
        recipe_id = _text(row["recipe_id"])
        if recipe_id in seen_test_ids or recipe_id in train_ids or recipe_id in dev_ids:
            raise ValueError(f"Yao test recipe ID is duplicate or leaked: {recipe_id}")
        seen_test_ids.add(recipe_id)
        labels = [int(value) for value in row["labels"]]
        if len(labels) != 4:
            raise ValueError(f"Yao test row {source_index} does not have four labels")
        gold_values = [_text(value) for value in row["label_names"]]
        if len(gold_values) != 4:
            raise ValueError(f"Yao test row {source_index} does not have four label names")
        for component_index, component in enumerate(_COMPONENTS):
            try:
                decoded_label = catalogs[component][labels[component_index]]
            except IndexError as error:
                raise ValueError(f"Yao test row {source_index} has out-of-range label") from error
            if decoded_label != gold_values[component_index]:
                raise ValueError(f"Yao test row {source_index} label/name mismatch")
        gold = dict(zip(_COMPONENTS, gold_values))
        tags = row.get("tags")
        if not isinstance(tags, list) or not tags:
            raise ValueError(f"Yao test row {source_index} has invalid ambiguity tag")
        decoded_tags = [_text(tag) for tag in tags]
        primary_tags = [tag for tag in decoded_tags if tag in YAO_TAG_COUNTS]
        if len(primary_tags) != 1:
            raise ValueError(f"Yao test row {source_index} has ambiguous primary tag: {decoded_tags}")
        words = [_text(value) for value in row["words"]]
        if not words:
            raise ValueError(f"Yao test row {source_index} has empty request")

        answer_keys = (
            gold["trigger_channel"],
            f"{gold['trigger_channel'].lower().strip()}.{gold['trigger_function'].lower().strip()}",
            gold["action_channel"],
            f"{gold['action_channel'].lower().strip()}.{gold['action_function'].lower().strip()}",
        )
        simulator_answers = {
            component: _decode_answer_options(answer_pools[index][answer_keys[index]], id_to_word)
            for index, component in enumerate(_COMPONENTS)
        }
        constraints = {
            "trigger_functions_for_service": _decode_constraint(
                raw_constraints[0], labels[0], catalogs["trigger_function"]
            ),
            "trigger_services_for_function": _decode_constraint(
                raw_constraints[1], labels[1], catalogs["trigger_channel"]
            ),
            "action_functions_for_service": _decode_constraint(
                raw_constraints[2], labels[2], catalogs["action_function"]
            ),
            "action_services_for_function": _decode_constraint(
                raw_constraints[3], labels[3], catalogs["action_channel"]
            ),
        }
        pseudo = [int(value) for value in row["pseudo_ask_labels"]]
        converted_row = {
            "case_id": f"yao:{recipe_id}",
            "recipe_id": recipe_id,
            "request": " ".join(words),
            "words": words,
            "gold": gold,
            "ambiguity_tag": primary_tags[0],
            "source_tags": decoded_tags,
            "pseudo_ask_labels": pseudo,
            "simulator_answers": simulator_answers,
            "valid_endpoint_constraints": constraints,
            "source_index": source_index,
        }
        _validate_converted_case(converted_row, source_index)
        converted_rows.append(converted_row)

    if require_official_shape:
        split_sizes = {
            "train": len(data["train"]),
            "dev": len(data["dev"]),
            "sample_dev": len(data["sample_dev"]),
            "test": len(data["test"]),
        }
        expected_split_sizes = {
            "train": 233076, "dev": 58209, "sample_dev": 1000, "test": 3870,
        }
        if split_sizes != expected_split_sizes:
            raise ValueError(f"Yao official split sizes changed: {split_sizes}")
        if len(id_to_word) != 78122:
            raise ValueError(f"Yao official vocabulary size changed: {len(id_to_word)}")
        pool_sizes = [len(pool) for pool in answer_pools]
        if pool_sizes != [251, 975, 218, 550]:
            raise ValueError(f"Yao official simulator pool sizes changed: {pool_sizes}")
        observed_tags = Counter(row["ambiguity_tag"] for row in converted_rows)
        if dict(observed_tags) != YAO_TAG_COUNTS:
            raise ValueError(f"Yao official tag counts changed: {dict(observed_tags)}")

    return {
        "schema_version": YAO_CONVERTED_SCHEMA,
        "source": {
            "repository_commit": YAO_REPOSITORY_COMMIT,
            "pickle_sha256": source_sha256,
        },
        "catalog": catalogs,
        "audit": {
            "split_sizes": {
                "train": len(data["train"]),
                "dev": len(data["dev"]),
                "sample_dev": len(data["sample_dev"]),
                "test": len(data["test"]),
            },
            "vocabulary_size": len(id_to_word),
            "answer_pool_sizes": [len(pool) for pool in answer_pools],
        },
        "test": converted_rows,
    }


def freeze_simulator_answer(
    *,
    recipe_id: str,
    component_index: int,
    answer_options: list[str],
    ask_ordinal: int,
) -> dict[str, Any]:
    """Choose one official simulator answer with the preregistered hash rule."""
    if component_index not in range(4):
        raise ValueError("Yao component index must be in [0, 3]")
    if ask_ordinal < 0:
        raise ValueError("Yao ask ordinal cannot be negative")
    if not answer_options or not all(isinstance(value, str) for value in answer_options):
        raise ValueError("Yao simulator answer options must be a nonempty string list")
    payload = "\0".join(
        (_ANSWER_HASH_NAMESPACE, recipe_id, str(component_index), str(ask_ordinal))
    )
    answer_index = int(hashlib.sha256(payload.encode("utf-8")).hexdigest(), 16) % len(
        answer_options
    )
    return {"answer_index": answer_index, "answer": answer_options[answer_index]}


def prepare_yao_interactive(
    converted_path: Path,
    *,
    size: int = YAO_SAMPLE_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare worker cases exclusively from inert converted JSON."""
    if converted_path.suffix.casefold() in {".pkl", ".pickle"}:
        raise ValueError("Yao experiment adapter requires converted JSON, never pickle")
    if size != YAO_SAMPLE_SIZE:
        raise ValueError(f"Yao partial protocol requires exactly {YAO_SAMPLE_SIZE} cases")
    converted = read_json(converted_path)
    if not isinstance(converted, dict) or converted.get("schema_version") != YAO_CONVERTED_SCHEMA:
        raise ValueError(f"Yao converted artifact must use {YAO_CONVERTED_SCHEMA}")
    source = converted.get("source")
    if not isinstance(source, dict):
        raise ValueError("Yao converted artifact is missing source provenance")
    if source.get("repository_commit") != YAO_REPOSITORY_COMMIT:
        raise ValueError("Yao converted artifact repository commit does not match pinned source")
    if source.get("pickle_sha256") != YAO_PICKLE_SHA256:
        raise ValueError("Yao converted artifact source SHA-256 does not match pinned pickle")
    rows = converted.get("test")
    if not isinstance(rows, list) or len(rows) != YAO_POPULATION_SIZE:
        raise ValueError(f"Yao converted test must contain exactly {YAO_POPULATION_SIZE} rows")

    seen_ids: set[str] = set()
    seen_source_indices: set[int] = set()
    buckets: dict[str, list[dict[str, Any]]] = {tag: [] for tag in YAO_TAG_COUNTS}
    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Yao converted row {row_index} is not an object")
        recipe_id = row.get("recipe_id")
        case_id = row.get("case_id")
        tag = row.get("ambiguity_tag")
        source_index = row.get("source_index")
        if not isinstance(recipe_id, str) or not recipe_id:
            raise ValueError(f"Yao converted row {row_index} has invalid recipe_id")
        if case_id != f"yao:{recipe_id}" or case_id in seen_ids:
            raise ValueError(f"Yao converted row {row_index} has invalid or duplicate case_id")
        if tag not in buckets:
            raise ValueError(f"Yao converted row {row_index} has unknown ambiguity tag")
        if not isinstance(source_index, int) or source_index in seen_source_indices:
            raise ValueError(f"Yao converted row {row_index} has invalid source_index")
        seen_ids.add(case_id)
        seen_source_indices.add(source_index)
        _validate_converted_case(row, row_index)
        buckets[str(tag)].append(row)

    observed_counts = {tag: len(buckets[tag]) for tag in YAO_TAG_COUNTS}
    if observed_counts != YAO_TAG_COUNTS:
        raise ValueError(f"Yao ambiguity-stratum population changed: {observed_counts}")

    selected_rows: list[tuple[dict[str, Any], str]] = []
    for tag in YAO_TAG_COUNTS:
        ranked = sorted(
            buckets[tag],
            key=lambda row: (
                _selection_hash(str(row["recipe_id"])),
                str(row["recipe_id"]),
            ),
        )
        selected_rows.extend(
            (row, _selection_hash(str(row["recipe_id"])))
            for row in ranked[: YAO_SAMPLE_QUOTAS[tag]]
        )

    cases = [_worker_case(row, selection_hash) for row, selection_hash in selected_rows]
    sample_strata = Counter(row["stratum"] for row in cases)
    manifest = {
        "benchmark": YAO_BENCHMARK,
        "sampling": "yao-ambiguity-stratified-sha256-v1",
        "sample_namespace": _SAMPLE_HASH_NAMESPACE,
        "answer_namespace": _ANSWER_HASH_NAMESPACE,
        "population_size": YAO_POPULATION_SIZE,
        "sample_size": len(cases),
        "population_strata": dict(YAO_TAG_COUNTS),
        "sample_strata": dict(sample_strata),
        "quotas": dict(YAO_SAMPLE_QUOTAS),
        "ordered_case_ids": [row["case_id"] for row in cases],
        "ordered_case_ids_sha256": sha256_text(
            "".join(f"{row['case_id']}\n" for row in cases)
        ),
        "source_commit": YAO_REPOSITORY_COMMIT,
        "source_artifact_sha256": YAO_PICKLE_SHA256,
        "converted_file": {
            "name": converted_path.name,
            "sha256": sha256_file(converted_path),
        },
        "evaluation_scope": "partial:150_of_3870",
        "worker_input_format": "inert-json-only",
    }
    return cases, manifest


def _selection_hash(recipe_id: str) -> str:
    payload = "\0".join((_SAMPLE_HASH_NAMESPACE, YAO_PICKLE_SHA256, recipe_id))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_converted_case(row: dict[str, Any], row_index: int) -> None:
    if not isinstance(row.get("request"), str) or not row["request"]:
        raise ValueError(f"Yao converted row {row_index} has invalid request")
    gold = row.get("gold")
    if not isinstance(gold, dict) or set(gold) != set(_COMPONENTS):
        raise ValueError(f"Yao converted row {row_index} has invalid gold endpoints")
    if not all(isinstance(gold[name], str) and gold[name] for name in _COMPONENTS):
        raise ValueError(f"Yao converted row {row_index} has empty gold endpoint")
    pseudo = row.get("pseudo_ask_labels")
    if not isinstance(pseudo, list) or len(pseudo) != 4 or any(value not in {0, 1} for value in pseudo):
        raise ValueError(f"Yao converted row {row_index} has invalid pseudo_ask_labels")
    answers = row.get("simulator_answers")
    if not isinstance(answers, dict) or set(answers) != set(_COMPONENTS):
        raise ValueError(f"Yao converted row {row_index} has invalid simulator answers")
    if any(
        not isinstance(answers[name], list)
        or not answers[name]
        or not all(isinstance(value, str) and value for value in answers[name])
        for name in _COMPONENTS
    ):
        raise ValueError(f"Yao converted row {row_index} has empty simulator answer pool")
    constraints = row.get("valid_endpoint_constraints")
    if not isinstance(constraints, dict) or set(constraints) != set(_CONSTRAINT_KEYS):
        raise ValueError(f"Yao converted row {row_index} has invalid endpoint constraints")
    for constraint_name, (gold_name,) in _CONSTRAINT_KEYS.items():
        values = constraints[constraint_name]
        if not isinstance(values, list) or gold[gold_name] not in values:
            raise ValueError(
                f"Yao converted row {row_index} constraint {constraint_name} excludes gold"
            )


def _worker_case(row: dict[str, Any], selection_hash: str) -> dict[str, Any]:
    frozen_answers = {}
    for component_index, component in enumerate(_COMPONENTS):
        frozen_answers[component] = freeze_simulator_answer(
            recipe_id=row["recipe_id"],
            component_index=component_index,
            answer_options=row["simulator_answers"][component],
            ask_ordinal=0,
        )
    return {
        "schema_version": "round9-case-v1",
        "benchmark": YAO_BENCHMARK,
        "case_id": row["case_id"],
        "stratum": row["ambiguity_tag"],
        "input": {"query": row["request"]},
        "simulator": {
            "answer_options": row["simulator_answers"],
            "frozen_answers": frozen_answers,
            "max_asks_per_component": 1,
        },
        "private_gold": {
            **row["gold"],
            "pseudo_ask_labels": row["pseudo_ask_labels"],
            "valid_endpoint_constraints": row["valid_endpoint_constraints"],
        },
        "audit": {
            "source_index": row["source_index"],
            "source_artifact_sha256": YAO_PICKLE_SHA256,
            "selection_sha256": selection_hash,
        },
    }
