#!/usr/bin/env python3
"""Stage an explicit, hash-pinned Round 9 public-release allowlist."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence


ROUND9_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND9_ROOT / "src"))

from farm_r9.privacy import DataClassification, DataSource  # noqa: E402
from farm_r9.release_staging import (  # noqa: E402
    ReleaseArtifactKind,
    ReleaseStagingError,
    stage_public_release,
)


def _arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument(
        "--artifact-kind",
        required=True,
        choices=[item.value for item in ReleaseArtifactKind],
    )
    parser.add_argument(
        "--benchmark-source",
        required=True,
        choices=[item.value for item in DataSource],
    )
    parser.add_argument(
        "--classification",
        required=True,
        choices=[item.value for item in DataClassification],
    )
    parser.add_argument(
        "--license-clearance",
        type=Path,
        help="required only for public_trace packages",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _arguments(argv)
    try:
        result = stage_public_release(
            manifest_path=args.manifest,
            source_root=args.source_root,
            output_directory=args.output_directory,
            expected_kind=ReleaseArtifactKind(args.artifact_kind),
            expected_source=DataSource(args.benchmark_source),
            expected_classification=DataClassification(args.classification),
            license_clearance_path=args.license_clearance,
        )
    except (OSError, ReleaseStagingError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    print(
        f"staged {result.files_staged} {result.artifact_kind.value} artifact(s) "
        f"({result.bytes_staged} bytes)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
