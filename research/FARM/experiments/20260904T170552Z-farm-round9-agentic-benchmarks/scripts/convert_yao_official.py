#!/usr/bin/env python3
"""One-way, hash-pinned conversion of the official Yao Python-2 pickle."""
from __future__ import annotations

import argparse
import builtins
import copyreg
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


RUN_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN_ROOT / "src"))

from farm_r9.adapters.yao_interactive import (
    YAO_PICKLE_SHA256,
    convert_loaded_yao_data,
)
from farm_r9.artifact_io import assert_no_secrets, sha256_file, write_json_atomic


class _LabelEncoderProxy:
    """State-only stand-in; conversion needs only the serialized ``classes_``."""


class _RestrictedYaoUnpickler(pickle.Unpickler):
    """Permit only globals present in the hash-pinned official artifact."""

    _STDLIB_GLOBALS = {
        ("__builtin__", "list"): builtins.list,
        ("__builtin__", "object"): builtins.object,
        ("collections", "defaultdict"): defaultdict,
        ("copy_reg", "_reconstructor"): copyreg._reconstructor,
        ("sklearn.preprocessing.label", "LabelEncoder"): _LabelEncoderProxy,
    }
    _NUMPY_GLOBALS = {
        ("numpy", "dtype"): "dtype",
        ("numpy", "ndarray"): "ndarray",
        ("numpy.core.multiarray", "_reconstruct"): "_reconstruct",
        ("numpy.core.multiarray", "scalar"): "scalar",
    }

    def find_class(self, module: str, name: str) -> Any:
        key = (module, name)
        if key in self._STDLIB_GLOBALS:
            return self._STDLIB_GLOBALS[key]
        if key in self._NUMPY_GLOBALS:
            try:
                import numpy
            except ImportError as error:
                raise RuntimeError(
                    "NumPy is required only for the one-time official Yao conversion"
                ) from error
            attribute = self._NUMPY_GLOBALS[key]
            if attribute in {"dtype", "ndarray"}:
                return getattr(numpy, attribute)
            multiarray = getattr(getattr(numpy, "_core", numpy.core), "multiarray")
            return getattr(multiarray, attribute)
        raise pickle.UnpicklingError(f"forbidden global in Yao pickle: {module}.{name}")


def load_verified_pickle(
    source: Path,
    *,
    expected_sha256: str = YAO_PICKLE_SHA256,
) -> Any:
    """Hash before opening with pickle, then deserialize through a strict whitelist."""
    actual_sha256 = sha256_file(source)
    if actual_sha256 != expected_sha256:
        raise ValueError(
            f"{source.name}: SHA-256 mismatch: expected {expected_sha256}, got {actual_sha256}"
        )
    with source.open("rb") as handle:
        return _RestrictedYaoUnpickler(handle, encoding="latin1", errors="strict").load()


def convert_official_pickle(source: Path, destination: Path) -> dict[str, Any]:
    if destination.suffix.casefold() != ".json":
        raise ValueError("Yao converted destination must be a .json file")
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite existing converted artifact: {destination}")
    loaded = load_verified_pickle(source)
    converted = convert_loaded_yao_data(
        loaded,
        source_sha256=YAO_PICKLE_SHA256,
        require_official_shape=True,
    )
    assert_no_secrets(converted)
    write_json_atomic(destination, converted)
    return {
        "destination": str(destination),
        "source_sha256": YAO_PICKLE_SHA256,
        "converted_sha256": sha256_file(destination),
        "test_cases": len(converted["test"]),
        "format": converted["schema_version"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pickle", required=True, type=Path, help="Expanded official data pickle")
    parser.add_argument("--output", required=True, type=Path, help="New inert JSON destination")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = convert_official_pickle(args.pickle.resolve(), args.output.resolve())
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
