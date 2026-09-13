from __future__ import annotations

import hashlib
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any


ROUND9_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND9_ROOT / "src"))

from farm_r9.privacy import DataClassification, DataSource  # noqa: E402
from farm_r9.release_staging import (  # noqa: E402
    LICENSE_MANIFEST_SCHEMA,
    RELEASE_MANIFEST_SCHEMA,
    ReleaseArtifactKind,
    ReleaseStagingError,
    stage_public_release,
)


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def _aggregate(label: str, *, confidential: bool = True) -> dict[str, Any]:
    aggregate = {
        "n": 150,
        "raw_numerators": {"success": 123},
        "raw_denominators": {"success": 150},
        "percentages": {"success": 82.0},
        "confidence_intervals": {"success": {"low": 75.0, "high": 88.0, "level": 95.0}},
        "hashes": {"source_artifact_sha256": "a" * 64},
    }
    if not confidential:
        aggregate["benchmark_label"] = label
    return aggregate


def _release_manifest(
    *,
    kind: str,
    source: str,
    classification: str,
    label: str,
    source_name: str,
    destination_name: str,
    payload: bytes,
) -> dict[str, Any]:
    return {
        "schema_version": RELEASE_MANIFEST_SCHEMA,
        "artifact_kind": kind,
        "benchmark_source": source,
        "data_classification": classification,
        "benchmark_label": label,
        "files": [
            {
                "source": source_name,
                "destination": destination_name,
                "sha256": hashlib.sha256(payload).hexdigest(),
                "size_bytes": len(payload),
            }
        ],
    }


def _license_manifest(
    *,
    release_manifest_payload: bytes,
    source: str,
    label: str,
) -> dict[str, Any]:
    return {
        "schema_version": LICENSE_MANIFEST_SCHEMA,
        "benchmark_source": source,
        "data_classification": "public",
        "benchmark_label": label,
        "release_manifest_sha256": hashlib.sha256(release_manifest_payload).hexdigest(),
        "scope": "public_benchmark_traces",
        "decision": "approved",
        "trace_redistribution_permitted": True,
        "license_expression": "Apache-2.0",
        "evidence_uri": "https://example.org/upstream/LICENSE",
        "evidence_sha256": "b" * 64,
        "checked_by": "release-reviewer",
        "checked_at_utc": "2026-01-01T00:00:00Z",
    }


class ReleaseStagingTests(unittest.TestCase):
    def _stage_aggregate(
        self,
        root: Path,
        *,
        source: DataSource = DataSource.FARM_V2,
        classification: DataClassification = DataClassification.CONFIDENTIAL,
        label: str = "farm_v2_test",
    ) -> tuple[Path, Path, bytes]:
        source_root = root / "source"
        source_root.mkdir()
        payload = _json_bytes(
            _aggregate(
                label,
                confidential=classification is DataClassification.CONFIDENTIAL,
            )
        )
        _write(source_root / "aggregate.json", payload)
        manifest_value = _release_manifest(
            kind="aggregate_only",
            source=source.value,
            classification=classification.value,
            label=label,
            source_name="aggregate.json",
            destination_name="tables/aggregate.json",
            payload=payload,
        )
        manifest = root / "release-manifest.json"
        _write(manifest, _json_bytes(manifest_value))
        output = root / "staged"
        result = stage_public_release(
            manifest_path=manifest,
            source_root=source_root,
            output_directory=output,
            expected_kind=ReleaseArtifactKind.AGGREGATE_ONLY,
            expected_source=source,
            expected_classification=classification,
        )
        self.assertEqual(result.files_staged, 1)
        return output, source_root, payload

    def _trace_fixture(
        self,
        root: Path,
        *,
        source: str = "bfcl_v4",
        classification: str = "public",
        label: str = "bfcl_v4_multi_turn_miss_param",
        trace: Any | None = None,
    ) -> tuple[Path, Path, Path, Path]:
        source_root = root / "source"
        source_root.mkdir()
        trace_value = (
            trace
            if trace is not None
            else {
                "benchmark_source": source,
                "data_classification": classification,
                "records": [
                    {
                        "id": "public-case-1",
                        "result": [{"tool": "weather", "arguments": {"city": "Seoul"}}],
                        "authorization": "[REDACTED]",
                    }
                ],
            }
        )
        payload = _json_bytes(trace_value)
        _write(source_root / "trace.json", payload)
        release_value = _release_manifest(
            kind="public_trace",
            source=source,
            classification=classification,
            label=label,
            source_name="trace.json",
            destination_name="traces/trace.json",
            payload=payload,
        )
        release_payload = _json_bytes(release_value)
        manifest = root / "release-manifest.json"
        license_path = root / "license-clearance.json"
        _write(manifest, release_payload)
        _write(
            license_path,
            _json_bytes(
                _license_manifest(
                    release_manifest_payload=release_payload,
                    source=source,
                    label=label,
                )
            ),
        )
        return source_root, manifest, license_path, root / "staged"

    def test_stages_confidential_farm_aggregate_only_after_strict_revalidation(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, source_root, payload = self._stage_aggregate(Path(directory))

            destination = output / "tables" / "aggregate.json"
            self.assertEqual(destination.read_bytes(), payload)
            self.assertEqual(stat.S_IMODE(output.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(destination.parent.stat().st_mode), 0o700)
            self.assertEqual(stat.S_IMODE(destination.stat().st_mode), 0o600)
            self.assertEqual(
                sorted(
                    path.relative_to(output).as_posix()
                    for path in output.rglob("*")
                    if path.is_file()
                ),
                ["tables/aggregate.json"],
            )
            self.assertFalse((output / "release-manifest.json").exists())
            self.assertTrue((source_root / "aggregate.json").exists())

    def test_stages_public_upstream_aggregate_without_trace_license(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output, _, _ = self._stage_aggregate(
                Path(directory),
                source=DataSource.RECIPEGEN,
                classification=DataClassification.PUBLIC,
                label="recipegen_gold",
            )
            self.assertTrue((output / "tables" / "aggregate.json").is_file())

    def test_aggregate_lane_rejects_content_fields_and_creates_no_output(self) -> None:
        for forbidden in ("query", "schemas", "case_ids", "raw_case_id"):
            with (
                self.subTest(forbidden=forbidden),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_root = root / "source"
                source_root.mkdir()
                aggregate = _aggregate("farm_v2_test")
                aggregate[forbidden] = "private-value"
                payload = _json_bytes(aggregate)
                _write(source_root / "aggregate.json", payload)
                manifest = root / "manifest.json"
                _write(
                    manifest,
                    _json_bytes(
                        _release_manifest(
                            kind="aggregate_only",
                            source="farm_v2",
                            classification="confidential",
                            label="farm_v2_test",
                            source_name="aggregate.json",
                            destination_name="aggregate.json",
                            payload=payload,
                        )
                    ),
                )
                output = root / "staged"
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.AGGREGATE_ONLY,
                        expected_source=DataSource.FARM_V2,
                        expected_classification=DataClassification.CONFIDENTIAL,
                    )
                self.assertFalse(output.exists())

    def test_aggregate_requires_a_whole_artifact_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            value = _aggregate("farm_v2_test")
            value.pop("hashes")
            payload = _json_bytes(value)
            _write(source_root / "aggregate.json", payload)
            manifest = root / "manifest.json"
            _write(
                manifest,
                _json_bytes(
                    _release_manifest(
                        kind="aggregate_only",
                        source="farm_v2",
                        classification="confidential",
                        label="farm_v2_test",
                        source_name="aggregate.json",
                        destination_name="aggregate.json",
                        payload=payload,
                    )
                ),
            )
            with self.assertRaisesRegex(ReleaseStagingError, "hashes|whole-artifact"):
                stage_public_release(
                    manifest_path=manifest,
                    source_root=source_root,
                    output_directory=root / "staged",
                    expected_kind=ReleaseArtifactKind.AGGREGATE_ONLY,
                    expected_source=DataSource.FARM_V2,
                    expected_classification=DataClassification.CONFIDENTIAL,
                )

    def test_stages_license_cleared_public_trace_and_nothing_else(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, manifest, license_path, output = self._trace_fixture(root)
            _write(
                source_root / "not-allowlisted.json",
                _json_bytes({"private": "not copied"}),
            )

            result = stage_public_release(
                manifest_path=manifest,
                source_root=source_root,
                output_directory=output,
                expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                expected_source=DataSource.BFCL_V4,
                expected_classification=DataClassification.PUBLIC,
                license_clearance_path=license_path,
            )

            self.assertEqual(result.files_staged, 1)
            self.assertEqual(
                sorted(
                    path.relative_to(output).as_posix()
                    for path in output.rglob("*")
                    if path.is_file()
                ),
                ["traces/trace.json"],
            )
            self.assertEqual(
                stat.S_IMODE((output / "traces" / "trace.json").stat().st_mode), 0o600
            )

    def test_farm_trace_is_categorically_refused(self) -> None:
        for classification in ("confidential", "public"):
            with (
                self.subTest(classification=classification),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(
                    root,
                    source="farm_v2",
                    classification=classification,
                    label="farm_v2_test",
                )
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.FARM_V2,
                        expected_classification=(
                            DataClassification.CONFIDENTIAL
                            if classification == "confidential"
                            else DataClassification.PUBLIC
                        ),
                        license_clearance_path=license_path,
                    )
                self.assertFalse(output.exists())

    def test_trace_requires_affirmative_hash_bound_license_clearance(self) -> None:
        mutations = {
            "missing": None,
            "denied": ("decision", "denied"),
            "no_permission": ("trace_redistribution_permitted", False),
            "wrong_hash": ("release_manifest_sha256", "c" * 64),
            "wrong_source": ("benchmark_source", "recipegen"),
            "wrong_scope": ("scope", "aggregate_only"),
            "unresolved_license": ("license_expression", "TBD"),
        }
        for name, mutation in mutations.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(root)
                selected_license: Path | None = license_path
                if mutation is None:
                    selected_license = None
                else:
                    value = json.loads(license_path.read_text(encoding="utf-8"))
                    value[mutation[0]] = mutation[1]
                    _write(license_path, _json_bytes(value))
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=selected_license,
                    )
                self.assertFalse(output.exists())

    def test_trace_rejects_credentials_confidential_markers_and_mixed_lineage(
        self,
    ) -> None:
        bad_traces = {
            "credential": {
                "benchmark_source": "bfcl_v4",
                "data_classification": "public",
                "authorization": "Bearer abcdefghijklmnopqrstuvwxyz",
            },
            "confidential": {
                "benchmark_source": "bfcl_v4",
                "data_classification": "confidential",
            },
            "farm": {"benchmark_source": "farm_v2", "data_classification": "public"},
            "mixed": {"benchmark_source": "recipegen", "data_classification": "public"},
            "private_id": {
                "benchmark_source": "bfcl_v4",
                "data_classification": "public",
                "private_ids": ["case-1"],
            },
        }
        for name, trace in bad_traces.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(
                    root, trace=trace
                )
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=license_path,
                    )
                self.assertFalse(output.exists())

    def test_hash_size_and_duplicate_json_checks_are_fail_closed(self) -> None:
        for name in ("hash", "size", "duplicate_json"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(root)
                manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
                if name == "hash":
                    manifest_value["files"][0]["sha256"] = "0" * 64
                    release_payload = _json_bytes(manifest_value)
                    _write(manifest, release_payload)
                elif name == "size":
                    manifest_value["files"][0]["size_bytes"] += 1
                    release_payload = _json_bytes(manifest_value)
                    _write(manifest, release_payload)
                else:
                    payload = (
                        b'{"benchmark_source":"bfcl_v4","benchmark_source":"farm_v2"}\n'
                    )
                    _write(source_root / "trace.json", payload)
                    manifest_value["files"][0]["sha256"] = hashlib.sha256(
                        payload
                    ).hexdigest()
                    manifest_value["files"][0]["size_bytes"] = len(payload)
                    release_payload = _json_bytes(manifest_value)
                    _write(manifest, release_payload)
                _write(
                    license_path,
                    _json_bytes(
                        _license_manifest(
                            release_manifest_payload=release_payload,
                            source="bfcl_v4",
                            label="bfcl_v4_multi_turn_miss_param",
                        )
                    ),
                )
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=license_path,
                    )
                self.assertFalse(output.exists())

    def test_rejects_source_or_destination_traversal_and_unsafe_paths(self) -> None:
        unsafe_paths = (
            "../trace.json",
            "/tmp/trace.json",
            "nested\\..\\trace.json",
            ".hidden.json",
        )
        for field, unsafe in (
            (field, unsafe)
            for field in ("source", "destination")
            for unsafe in unsafe_paths
        ):
            with (
                self.subTest(field=field, unsafe=unsafe),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(root)
                value = json.loads(manifest.read_text(encoding="utf-8"))
                value["files"][0][field] = unsafe
                release_payload = _json_bytes(value)
                _write(manifest, release_payload)
                _write(
                    license_path,
                    _json_bytes(
                        _license_manifest(
                            release_manifest_payload=release_payload,
                            source="bfcl_v4",
                            label="bfcl_v4_multi_turn_miss_param",
                        )
                    ),
                )
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=license_path,
                    )
                self.assertFalse(output.exists())

    def test_rejects_duplicate_sources_or_destinations(self) -> None:
        for duplicated_field in ("source", "destination"):
            with (
                self.subTest(duplicated_field=duplicated_field),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(root)
                first_payload = (source_root / "trace.json").read_bytes()
                _write(source_root / "trace2.json", first_payload)
                value = json.loads(manifest.read_text(encoding="utf-8"))
                second = dict(value["files"][0])
                second["source"] = "trace2.json"
                second["destination"] = "traces/trace2.json"
                second[duplicated_field] = value["files"][0][duplicated_field]
                value["files"].append(second)
                release_payload = _json_bytes(value)
                _write(manifest, release_payload)
                _write(
                    license_path,
                    _json_bytes(
                        _license_manifest(
                            release_manifest_payload=release_payload,
                            source="bfcl_v4",
                            label="bfcl_v4_multi_turn_miss_param",
                        )
                    ),
                )
                with self.assertRaisesRegex(ReleaseStagingError, "duplicate"):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=license_path,
                    )
                self.assertFalse(output.exists())

    def test_rejects_source_and_control_manifest_symlinks(self) -> None:
        for target in ("source", "manifest", "license"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_root, manifest, license_path, output = self._trace_fixture(root)
                if target == "source":
                    real = source_root / "real.json"
                    (source_root / "trace.json").rename(real)
                    (source_root / "trace.json").symlink_to(real.name)
                elif target == "manifest":
                    real = root / "real-manifest.json"
                    manifest.rename(real)
                    manifest.symlink_to(real.name)
                else:
                    real = root / "real-license.json"
                    license_path.rename(real)
                    license_path.symlink_to(real.name)
                with self.assertRaises(ReleaseStagingError):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=source_root,
                        output_directory=output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=license_path,
                    )
                self.assertFalse(output.exists())

    def test_rejects_symlinked_source_root_or_output_parent(self) -> None:
        for target in ("source_root", "output_parent"):
            with (
                self.subTest(target=target),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                source_root, manifest, license_path, default_output = (
                    self._trace_fixture(root)
                )
                selected_source = source_root
                selected_output = default_output
                if target == "source_root":
                    selected_source = root / "source-link"
                    selected_source.symlink_to(
                        source_root.name, target_is_directory=True
                    )
                else:
                    real_parent = root / "real-output-parent"
                    real_parent.mkdir()
                    linked_parent = root / "output-parent-link"
                    linked_parent.symlink_to(
                        real_parent.name,
                        target_is_directory=True,
                    )
                    selected_output = linked_parent / "staged"
                with self.assertRaisesRegex(ReleaseStagingError, "symbolic links"):
                    stage_public_release(
                        manifest_path=manifest,
                        source_root=selected_source,
                        output_directory=selected_output,
                        expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                        expected_source=DataSource.BFCL_V4,
                        expected_classification=DataClassification.PUBLIC,
                        license_clearance_path=license_path,
                    )
                self.assertFalse(default_output.exists())

    def test_refuses_existing_output_and_explicit_declaration_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, manifest, license_path, output = self._trace_fixture(root)
            output.mkdir()
            marker = output / "keep.txt"
            marker.write_text("untouched", encoding="utf-8")
            with self.assertRaisesRegex(ReleaseStagingError, "overwrite"):
                stage_public_release(
                    manifest_path=manifest,
                    source_root=source_root,
                    output_directory=output,
                    expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                    expected_source=DataSource.BFCL_V4,
                    expected_classification=DataClassification.PUBLIC,
                    license_clearance_path=license_path,
                )
            self.assertEqual(marker.read_text(encoding="utf-8"), "untouched")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root, manifest, license_path, output = self._trace_fixture(root)
            with self.assertRaisesRegex(ReleaseStagingError, "caller declarations"):
                stage_public_release(
                    manifest_path=manifest,
                    source_root=source_root,
                    output_directory=output,
                    expected_kind=ReleaseArtifactKind.PUBLIC_TRACE,
                    expected_source=DataSource.RECIPEGEN,
                    expected_classification=DataClassification.PUBLIC,
                    license_clearance_path=license_path,
                )
            self.assertFalse(output.exists())

    def test_control_manifests_cannot_be_allowlisted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            root_source = root / "source"
            root_source.mkdir()
            aggregate = _json_bytes(_aggregate("farm_v2_test"))
            manifest = root_source / "release-manifest.json"
            manifest_value = _release_manifest(
                kind="aggregate_only",
                source="farm_v2",
                classification="confidential",
                label="farm_v2_test",
                source_name="release-manifest.json",
                destination_name="aggregate.json",
                payload=aggregate,
            )
            _write(manifest, _json_bytes(manifest_value))
            # Re-pin the self-referential file to its actual current bytes.  It
            # still cannot pass because control files are categorically denied.
            value = json.loads(manifest.read_text(encoding="utf-8"))
            current = manifest.read_bytes()
            value["files"][0]["sha256"] = hashlib.sha256(current).hexdigest()
            value["files"][0]["size_bytes"] = len(current)
            _write(manifest, _json_bytes(value))
            with self.assertRaisesRegex(ReleaseStagingError, "control manifests"):
                stage_public_release(
                    manifest_path=manifest,
                    source_root=root_source,
                    output_directory=root / "staged",
                    expected_kind=ReleaseArtifactKind.AGGREGATE_ONLY,
                    expected_source=DataSource.FARM_V2,
                    expected_classification=DataClassification.CONFIDENTIAL,
                )

    def test_cli_requires_explicit_kind_source_and_classification(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_root = root / "source"
            source_root.mkdir()
            payload = _json_bytes(_aggregate("farm_v2_test"))
            _write(source_root / "aggregate.json", payload)
            manifest = root / "manifest.json"
            _write(
                manifest,
                _json_bytes(
                    _release_manifest(
                        kind="aggregate_only",
                        source="farm_v2",
                        classification="confidential",
                        label="farm_v2_test",
                        source_name="aggregate.json",
                        destination_name="aggregate.json",
                        payload=payload,
                    )
                ),
            )
            output = root / "staged"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROUND9_ROOT / "scripts" / "stage_public_release.py"),
                    "--manifest",
                    str(manifest),
                    "--source-root",
                    str(source_root),
                    "--output-directory",
                    str(output),
                    "--artifact-kind",
                    "aggregate_only",
                    "--benchmark-source",
                    "farm_v2",
                    "--classification",
                    "confidential",
                ],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("staged 1 aggregate_only", completed.stdout)
            self.assertTrue((output / "aggregate.json").is_file())


if __name__ == "__main__":
    unittest.main()
