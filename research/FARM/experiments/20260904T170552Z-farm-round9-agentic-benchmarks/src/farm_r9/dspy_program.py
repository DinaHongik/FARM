"""Genuine DSPy configuration module and fail-closed optimization gate."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from farm_r9.artifact_io import read_json, sha256_file


class DSPyUnavailable(RuntimeError):
    pass


class InvalidDSPyArtifact(ValueError):
    pass


try:  # DSPy is pinned in the DGX Round-8 dependency environment.
    import dspy
except ImportError:  # pragma: no cover - exercised on lightweight workstations
    dspy = None


if dspy is not None:
    class AppletConfigurationSignature(dspy.Signature):
        """Produce a grounded ask-or-commit JSON action for fixed endpoints."""

        instruction = dspy.InputField(desc="Strict configuration policy and allowed source types")
        grounding_text = dspy.InputField(desc="Text whose exact spans may be cited as query_literal")
        selected_trigger_json = dspy.InputField(desc="Fixed selected trigger schema and ingredients")
        selected_action_json = dspy.InputField(desc="Fixed selected action schema")
        observations_json = dspy.InputField(desc="Supplied resources, secret aliases, and prior clarification answers")
        action_json = dspy.OutputField(desc="One strict AgentAction JSON object; no prose")


    class AppletConfigurationProgram(dspy.Module):
        def __init__(self) -> None:
            super().__init__()
            self.configure = dspy.Predict(AppletConfigurationSignature)

        def forward(
            self, *, instruction: str, grounding_text: str,
            selected_trigger_json: str, selected_action_json: str,
            observations_json: str,
        ) -> Any:
            return self.configure(
                instruction=instruction, grounding_text=grounding_text,
                selected_trigger_json=selected_trigger_json,
                selected_action_json=selected_action_json,
                observations_json=observations_json,
            )
else:
    class AppletConfigurationSignature:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise DSPyUnavailable("dspy is not installed")


    class AppletConfigurationProgram:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise DSPyUnavailable("dspy is not installed")


_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_PUBLIC_TRAIN_SOURCES = {"recipegen", "interactive_ifttt", "bfcl_v4", "targe"}


def validate_optimizer_manifest(manifest: Mapping[str, Any], *, manifest_path: Path | None = None) -> dict[str, Any]:
    """Validate proof that a DSPy optimizer actually ran on public train data."""
    required = {
        "schema_version", "status", "optimizer", "optimizer_executed",
        "training_source", "training_classification", "train_count",
        "training_case_ids_sha256", "heldout_overlap_count",
        "program_state_path", "program_state_sha256",
    }
    if set(manifest) != required:
        raise InvalidDSPyArtifact("optimizer manifest has missing or unexpected fields")
    if manifest["schema_version"] != "round9-dspy-optimizer-v1" or manifest["status"] != "optimized":
        raise InvalidDSPyArtifact("artifact is not an optimized Round9 DSPy program")
    if manifest["optimizer_executed"] is not True or not isinstance(manifest["optimizer"], str) or not manifest["optimizer"]:
        raise InvalidDSPyArtifact("optimizer execution proof is absent")
    if manifest["training_classification"] != "public" or manifest["training_source"] not in _PUBLIC_TRAIN_SOURCES:
        raise InvalidDSPyArtifact("DSPy optimization must use registered public training data")
    if type(manifest["train_count"]) is not int or manifest["train_count"] <= 0:
        raise InvalidDSPyArtifact("train_count must be positive")
    if manifest["heldout_overlap_count"] != 0:
        raise InvalidDSPyArtifact("training and held-out case IDs overlap")
    for key in ("training_case_ids_sha256", "program_state_sha256"):
        if not isinstance(manifest[key], str) or not _SHA256.fullmatch(manifest[key]):
            raise InvalidDSPyArtifact(f"{key} is not a SHA-256 digest")
    state_path = Path(str(manifest["program_state_path"]))
    if state_path.is_absolute() or ".." in state_path.parts:
        raise InvalidDSPyArtifact("program_state_path must be relative and contained")
    if manifest_path is not None:
        resolved = manifest_path.parent / state_path
        if not resolved.is_file() or sha256_file(resolved) != manifest["program_state_sha256"]:
            raise InvalidDSPyArtifact("DSPy state file is absent or hash-mismatched")
    return dict(manifest)


def load_optimized_program(manifest_path: Path) -> Any:
    """Load a program that may truthfully be labeled ``dspy_compiled_agent``."""
    manifest = validate_optimizer_manifest(read_json(manifest_path), manifest_path=manifest_path)
    if dspy is None:
        raise DSPyUnavailable("dspy is not installed")
    program = AppletConfigurationProgram()
    program.load(str(manifest_path.parent / manifest["program_state_path"]))
    return program


def exact_action_json_metric(example: Any, prediction: Any, trace: Any = None) -> float:
    """Exact canonical JSON metric for genuinely supervised public records."""
    del trace
    try:
        expected = json.loads(str(example.action_json))
        actual = json.loads(str(prediction.action_json))
    except (AttributeError, TypeError, json.JSONDecodeError):
        return 0.0
    return float(expected == actual)

