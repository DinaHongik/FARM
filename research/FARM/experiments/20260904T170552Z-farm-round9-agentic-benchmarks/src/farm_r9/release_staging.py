"""Fail-closed staging for deliberately reviewed Round 9 release artifacts.

This module has two deliberately separate lanes:

* ``aggregate_only`` accepts a registered benchmark aggregate only after the
  existing strict public-aggregate validator reproduces the input exactly.
  FARM is accepted only in this lane and only as confidential source lineage.
* ``public_trace`` accepts only registered public upstream benchmarks, requires
  a separate affirmative license-clearance manifest bound to the byte-exact
  release manifest, and requires traces to have already been credential-
  redacted.

Neither control manifest is copied into the staged package.  Consequently, the
output directory contains exactly the explicitly allowlisted artifacts and no
local source paths or reviewer identities.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path, PurePosixPath
from typing import Any, NoReturn
from urllib.parse import urlsplit

from farm_r9.privacy import (
    DataClassification,
    DataSource,
    PrivacyViolation,
    export_public_aggregate,
    redact_public_trace,
)


RELEASE_MANIFEST_SCHEMA = "farm-r9-public-release-manifest-v1"
LICENSE_MANIFEST_SCHEMA = "farm-r9-license-clearance-v1"


class ReleaseArtifactKind(str, Enum):
    """The two non-mixable release package types."""

    AGGREGATE_ONLY = "aggregate_only"
    PUBLIC_TRACE = "public_trace"


class ReleaseStagingError(ValueError):
    """Raised before or during staging when a release boundary is violated."""


@dataclass(frozen=True)
class StagingResult:
    """Non-content summary of one completed staging operation."""

    artifact_kind: ReleaseArtifactKind
    benchmark_source: DataSource
    files_staged: int
    bytes_staged: int
    output_directory: Path


@dataclass(frozen=True)
class _ManifestFile:
    source_parts: tuple[str, ...]
    destination_parts: tuple[str, ...]
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class _ReleaseManifest:
    artifact_kind: ReleaseArtifactKind
    benchmark_source: DataSource
    classification: DataClassification
    benchmark_label: str
    files: tuple[_ManifestFile, ...]


@dataclass(frozen=True)
class _PreparedArtifact:
    destination_parts: tuple[str, ...]
    content: bytes


_RELEASE_MANIFEST_FIELDS = {
    "schema_version",
    "artifact_kind",
    "benchmark_source",
    "data_classification",
    "benchmark_label",
    "files",
}
_FILE_FIELDS = {"source", "destination", "sha256", "size_bytes"}
_LICENSE_MANIFEST_FIELDS = {
    "schema_version",
    "benchmark_source",
    "data_classification",
    "benchmark_label",
    "release_manifest_sha256",
    "scope",
    "decision",
    "trace_redistribution_permitted",
    "license_expression",
    "evidence_uri",
    "evidence_sha256",
    "checked_by",
    "checked_at_utc",
}
_SOURCE_CLASSIFICATION = {
    DataSource.FARM_V2: DataClassification.CONFIDENTIAL,
    DataSource.RECIPEGEN: DataClassification.PUBLIC,
    DataSource.INTERACTIVE_IFTTT: DataClassification.PUBLIC,
    DataSource.BFCL_V4: DataClassification.PUBLIC,
    DataSource.TARGE: DataClassification.PUBLIC,
}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_LABEL = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,127}\Z")
_REVIEWER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.@-]{1,127}\Z")
_LICENSE_EXPRESSION = re.compile(r"[A-Za-z0-9][A-Za-z0-9.+() -]{0,127}\Z")
_PATH_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TRACE_SUFFIXES = {".json", ".jsonl"}
_AGGREGATE_SUFFIXES = {".json"}
_SENSITIVE_PATH_MARKERS = {
    ".env",
    "credential",
    "credentials",
    "private_key",
    "secret",
    "secrets",
    "token",
    "tokens",
}
_CLASSIFICATION_KEYS = {
    "dataclassification",
    "sourceclassification",
    "trainingclassification",
}
_SOURCE_KEYS = {
    "benchmarksource",
    "datasource",
    "datasetsource",
    "datasetid",
    "benchmark",
    "benchmarklabel",
}
_PRIVATE_IDENTIFIER_MARKERS = {"privateid", "privateids", "rawcaseid", "rawcaseids"}


def _refuse(reason: str) -> NoReturn:
    raise ReleaseStagingError(f"release staging refused: {reason}")


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, child in pairs:
        if key in value:
            _refuse("JSON contains a duplicate object key")
        value[key] = child
    return value


def _reject_json_constant(_: str) -> NoReturn:
    _refuse("JSON contains a non-finite number")


def _strict_json_loads(payload: bytes, *, expect_jsonl: bool) -> Any:
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        _refuse("artifact is not valid UTF-8")

    def parse_one(value: str) -> Any:
        try:
            return json.loads(
                value,
                object_pairs_hook=_strict_object,
                parse_constant=_reject_json_constant,
            )
        except ReleaseStagingError:
            raise
        except (json.JSONDecodeError, TypeError, ValueError):
            _refuse("artifact is not strict JSON")

    if not expect_jsonl:
        return parse_one(text)

    rows: list[Any] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        row = parse_one(line)
        if not isinstance(row, Mapping):
            _refuse("each JSONL trace record must be an object")
        rows.append(row)
    if not rows:
        _refuse("JSONL trace must contain at least one record")
    return rows


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _assert_no_symlink_components(
    path: Path, *, allow_missing_final: bool = False
) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    parts = absolute.parts[1:]
    for index, part in enumerate(parts):
        current = current / part
        try:
            status = os.lstat(current)
        except FileNotFoundError:
            if allow_missing_final and index == len(parts) - 1:
                return
            _refuse("a required path component does not exist")
        if stat.S_ISLNK(status.st_mode):
            _refuse("symbolic links are forbidden")


def _read_regular_file(path: Path) -> bytes:
    absolute = _absolute(path)
    _assert_no_symlink_components(absolute)
    flags = os.O_RDONLY
    if hasattr(os, "O_CLOEXEC"):
        flags |= os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(absolute, flags)
    except OSError:
        _refuse("unable to open a required regular file")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _refuse("allowlisted artifacts must be regular files")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    ) != (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    ):
        _refuse("source artifact changed while it was being read")
    try:
        pathname_status = os.lstat(absolute)
    except OSError:
        _refuse("source artifact changed while it was being read")
    if (pathname_status.st_dev, pathname_status.st_ino) != (after.st_dev, after.st_ino):
        _refuse("source artifact changed while it was being read")
    return b"".join(chunks)


def _parse_control_json(payload: bytes, *, name: str) -> Mapping[str, Any]:
    value = _strict_json_loads(payload, expect_jsonl=False)
    if not isinstance(value, Mapping):
        _refuse(f"{name} must be a JSON object")
    return value


def _enum_value(enum_type: type[Enum], value: Any, *, field: str) -> Any:
    if not isinstance(value, str):
        _refuse(f"{field} must be an explicit string enum")
    try:
        return enum_type(value)
    except ValueError:
        _refuse(f"{field} is not a registered value")


def _label(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _LABEL.fullmatch(value):
        _refuse(f"{field} must be a short public identifier")
    return value


def _relative_parts(value: Any, *, field: str) -> tuple[str, ...]:
    if not isinstance(value, str) or not value or "\\" in value or "\x00" in value:
        _refuse(f"{field} must be a safe relative POSIX path")
    candidate = PurePosixPath(value)
    if candidate.is_absolute() or not candidate.parts:
        _refuse(f"{field} must be a safe relative POSIX path")
    parts = candidate.parts
    if any(
        part in {"", ".", ".."} or not _PATH_SEGMENT.fullmatch(part) for part in parts
    ):
        _refuse(f"{field} must be a safe relative POSIX path")
    normalized_parts = tuple(part.casefold() for part in parts)
    if any(
        part in _SENSITIVE_PATH_MARKERS or part.endswith((".pem", ".key"))
        for part in normalized_parts
    ):
        _refuse(f"{field} names a forbidden sensitive artifact")
    return tuple(parts)


def _parse_release_manifest(value: Mapping[str, Any]) -> _ReleaseManifest:
    if set(value) != _RELEASE_MANIFEST_FIELDS:
        _refuse("release manifest fields do not match the required schema")
    if value["schema_version"] != RELEASE_MANIFEST_SCHEMA:
        _refuse("unsupported release manifest schema")

    kind = _enum_value(
        ReleaseArtifactKind,
        value["artifact_kind"],
        field="artifact_kind",
    )
    source = _enum_value(
        DataSource, value["benchmark_source"], field="benchmark_source"
    )
    classification = _enum_value(
        DataClassification,
        value["data_classification"],
        field="data_classification",
    )
    benchmark_label = _label(value["benchmark_label"], field="benchmark_label")
    expected_classification = _SOURCE_CLASSIFICATION[source]
    if classification is not expected_classification:
        _refuse("classification conflicts with registered benchmark lineage")
    if kind is ReleaseArtifactKind.PUBLIC_TRACE:
        if source is DataSource.FARM_V2:
            _refuse("FARM traces are never public-release artifacts")
        if classification is not DataClassification.PUBLIC:
            _refuse("public traces require explicit public classification")

    raw_files = value["files"]
    if (
        not isinstance(raw_files, Sequence)
        or isinstance(raw_files, (str, bytes))
        or not raw_files
    ):
        _refuse("release manifest requires a nonempty file allowlist")
    files: list[_ManifestFile] = []
    seen_sources: set[tuple[str, ...]] = set()
    seen_destinations: set[tuple[str, ...]] = set()
    allowed_suffixes = (
        _AGGREGATE_SUFFIXES
        if kind is ReleaseArtifactKind.AGGREGATE_ONLY
        else _TRACE_SUFFIXES
    )
    for raw_file in raw_files:
        if not isinstance(raw_file, Mapping) or set(raw_file) != _FILE_FIELDS:
            _refuse("every allowlist entry must match the required file schema")
        source_parts = _relative_parts(raw_file["source"], field="files.source")
        destination_parts = _relative_parts(
            raw_file["destination"],
            field="files.destination",
        )
        if source_parts in seen_sources or destination_parts in seen_destinations:
            _refuse("duplicate allowlist sources or destinations are forbidden")
        if Path(source_parts[-1]).suffix.casefold() not in allowed_suffixes:
            _refuse("artifact extension is not allowed for the selected package kind")
        if Path(destination_parts[-1]).suffix.casefold() not in allowed_suffixes:
            _refuse(
                "destination extension is not allowed for the selected package kind"
            )
        digest = raw_file["sha256"]
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            _refuse("allowlisted artifact requires a lowercase whole-file SHA-256")
        size_bytes = raw_file["size_bytes"]
        if (
            isinstance(size_bytes, bool)
            or not isinstance(size_bytes, int)
            or size_bytes < 0
        ):
            _refuse("allowlisted artifact requires a nonnegative byte size")
        files.append(
            _ManifestFile(
                source_parts=source_parts,
                destination_parts=destination_parts,
                sha256=digest,
                size_bytes=size_bytes,
            )
        )
        seen_sources.add(source_parts)
        seen_destinations.add(destination_parts)

    return _ReleaseManifest(
        artifact_kind=kind,
        benchmark_source=source,
        classification=classification,
        benchmark_label=benchmark_label,
        files=tuple(files),
    )


def _validate_license_manifest(
    value: Mapping[str, Any],
    *,
    release_manifest_sha256: str,
    release_manifest: _ReleaseManifest,
) -> None:
    if set(value) != _LICENSE_MANIFEST_FIELDS:
        _refuse("license-clearance manifest fields do not match the required schema")
    if value["schema_version"] != LICENSE_MANIFEST_SCHEMA:
        _refuse("unsupported license-clearance manifest schema")
    if value["benchmark_source"] != release_manifest.benchmark_source.value:
        _refuse("license clearance is for a different benchmark source")
    if value["data_classification"] != DataClassification.PUBLIC.value:
        _refuse("license clearance must declare public classification")
    if value["benchmark_label"] != release_manifest.benchmark_label:
        _refuse("license clearance is for a different benchmark label")
    if value["release_manifest_sha256"] != release_manifest_sha256:
        _refuse("license clearance is not bound to this release manifest")
    if value["scope"] != "public_benchmark_traces":
        _refuse("license clearance has the wrong redistribution scope")
    if (
        value["decision"] != "approved"
        or value["trace_redistribution_permitted"] is not True
    ):
        _refuse("trace redistribution has not been affirmatively approved")

    license_expression = value["license_expression"]
    if (
        not isinstance(license_expression, str)
        or not _LICENSE_EXPRESSION.fullmatch(license_expression)
        or license_expression.casefold() in {"unknown", "none", "tbd", "todo"}
    ):
        _refuse("license expression is missing or unresolved")
    if not isinstance(value["evidence_sha256"], str) or not _SHA256.fullmatch(
        value["evidence_sha256"]
    ):
        _refuse("license evidence requires a lowercase whole-file SHA-256")
    checked_by = value["checked_by"]
    if (
        not isinstance(checked_by, str)
        or not _REVIEWER.fullmatch(checked_by)
        or checked_by.casefold() in {"unknown", "nobody", "tbd", "todo"}
    ):
        _refuse("license clearance requires an identified reviewer")
    evidence_uri = value["evidence_uri"]
    if not isinstance(evidence_uri, str):
        _refuse("license evidence URI must be an HTTPS URL")
    parsed_uri = urlsplit(evidence_uri)
    if (
        parsed_uri.scheme != "https"
        or not parsed_uri.hostname
        or parsed_uri.username is not None
        or parsed_uri.password is not None
        or parsed_uri.query
        or parsed_uri.fragment
    ):
        _refuse(
            "license evidence URI must be an HTTPS URL without credentials or parameters"
        )
    checked_at = value["checked_at_utc"]
    if not isinstance(checked_at, str) or not checked_at.endswith("Z"):
        _refuse("license clearance requires an RFC3339 UTC timestamp")
    try:
        parsed_time = datetime.fromisoformat(checked_at[:-1] + "+00:00")
    except ValueError:
        _refuse("license clearance requires an RFC3339 UTC timestamp")
    if parsed_time.utcoffset() != timedelta(0):
        _refuse("license clearance timestamp must be UTC")
    if parsed_time > datetime.now(timezone.utc) + timedelta(minutes=5):
        _refuse("license clearance timestamp cannot be in the future")


def _normalized_field(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _lineage_from_value(value: str) -> DataSource | None:
    normalized = _normalized_field(value)
    if "farm" in normalized:
        return DataSource.FARM_V2
    if normalized.startswith("recipegen"):
        return DataSource.RECIPEGEN
    if normalized.startswith("bfcl"):
        return DataSource.BFCL_V4
    if normalized.startswith("targe"):
        return DataSource.TARGE
    if normalized.startswith("interactiveifttt") or normalized.startswith("yao"):
        return DataSource.INTERACTIVE_IFTTT
    return None


def _reject_trace_lineage_violations(
    value: Any, *, expected_source: DataSource
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                _refuse("trace contains a non-string field name")
            normalized_key = _normalized_field(key)
            if "farm" in normalized_key:
                _refuse("trace contains FARM-labelled content")
            if any(marker in normalized_key for marker in _PRIVATE_IDENTIFIER_MARKERS):
                _refuse("trace contains a private identifier field")
            if normalized_key in _CLASSIFICATION_KEYS:
                if not isinstance(child, str) or _normalized_field(child) != "public":
                    _refuse("trace contains a non-public classification marker")
            if normalized_key in _SOURCE_KEYS:
                if not isinstance(child, str):
                    _refuse("trace contains an ambiguous source-lineage marker")
                observed_source = _lineage_from_value(child)
                if observed_source is DataSource.FARM_V2:
                    _refuse("trace contains FARM lineage")
                if (
                    observed_source is not None
                    and observed_source is not expected_source
                ):
                    _refuse("trace contains mixed benchmark lineage")
            _reject_trace_lineage_violations(child, expected_source=expected_source)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for child in value:
            _reject_trace_lineage_violations(child, expected_source=expected_source)


def _validate_aggregate(
    value: Any,
    *,
    classification: DataClassification,
    benchmark_label: str,
) -> None:
    records: list[Any]
    if isinstance(value, Mapping):
        records = [value]
    elif isinstance(value, list) and value:
        records = value
    else:
        _refuse("aggregate artifact must contain one object or a nonempty object list")
    for record in records:
        if not isinstance(record, Mapping):
            _refuse("aggregate list entries must be objects")
        try:
            exported = export_public_aggregate(
                record,
                source_classification=classification,
            )
        except PrivacyViolation as error:
            raise ReleaseStagingError(f"release staging refused: {error}") from error
        if exported != record:
            _refuse(
                "aggregate is not byte-source-equivalent to the strict public contract"
            )
        if (
            classification is DataClassification.PUBLIC
            and exported["benchmark_label"] != benchmark_label
        ):
            _refuse("aggregate benchmark label does not match its release manifest")
        hashes = exported.get("hashes")
        if not isinstance(hashes, Mapping) or not hashes:
            _refuse("aggregate must include at least one whole-artifact SHA-256")


def _prepare_artifact(
    source_root: Path,
    manifest_file: _ManifestFile,
    *,
    release_manifest: _ReleaseManifest,
    forbidden_control_paths: set[Path],
) -> _PreparedArtifact:
    source_path = source_root.joinpath(*manifest_file.source_parts)
    absolute_source = _absolute(source_path)
    if absolute_source in forbidden_control_paths:
        _refuse("control manifests cannot be copied into a release package")
    if release_manifest.artifact_kind is ReleaseArtifactKind.PUBLIC_TRACE:
        normalized_source = tuple(
            part.casefold() for part in manifest_file.source_parts
        )
        if any("farm" in part or "confidential" in part for part in normalized_source):
            _refuse("FARM or confidential trace paths are forbidden")

    payload = _read_regular_file(absolute_source)
    if len(payload) != manifest_file.size_bytes:
        _refuse("allowlisted artifact byte size does not match its manifest")
    if hashlib.sha256(payload).hexdigest() != manifest_file.sha256:
        _refuse("allowlisted artifact SHA-256 does not match its manifest")

    suffix = Path(manifest_file.source_parts[-1]).suffix.casefold()
    parsed = _strict_json_loads(payload, expect_jsonl=suffix == ".jsonl")
    if release_manifest.artifact_kind is ReleaseArtifactKind.AGGREGATE_ONLY:
        _validate_aggregate(
            parsed,
            classification=release_manifest.classification,
            benchmark_label=release_manifest.benchmark_label,
        )
    else:
        _reject_trace_lineage_violations(
            parsed,
            expected_source=release_manifest.benchmark_source,
        )
        try:
            redacted = redact_public_trace(
                parsed,
                source_classification=release_manifest.classification,
                source=release_manifest.benchmark_source,
            )
        except PrivacyViolation as error:
            raise ReleaseStagingError(f"release staging refused: {error}") from error
        if redacted != parsed:
            _refuse("public trace still contains credential-shaped material")
    return _PreparedArtifact(
        destination_parts=manifest_file.destination_parts,
        content=payload,
    )


def _ensure_existing_directory(path: Path) -> Path:
    absolute = _absolute(path)
    _assert_no_symlink_components(absolute)
    try:
        status = os.stat(absolute, follow_symlinks=False)
    except OSError:
        _refuse("a required directory does not exist")
    if not stat.S_ISDIR(status.st_mode):
        _refuse("a required path is not a directory")
    return absolute


def _mkdir_private(path: Path) -> None:
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        _refuse("output destination already exists; overwrite is forbidden")
    os.chmod(path, 0o700, follow_symlinks=False)


def _write_prepared_artifacts(
    output: Path, artifacts: Sequence[_PreparedArtifact]
) -> None:
    _mkdir_private(output)
    for artifact in artifacts:
        parent = output
        for segment in artifact.destination_parts[:-1]:
            parent = parent / segment
            if not parent.exists():
                _mkdir_private(parent)
            else:
                status = os.stat(parent, follow_symlinks=False)
                if not stat.S_ISDIR(status.st_mode) or stat.S_ISLNK(
                    os.lstat(parent).st_mode
                ):
                    _refuse("destination parent is not a safe directory")
                os.chmod(parent, 0o700, follow_symlinks=False)
        destination = parent / artifact.destination_parts[-1]
        flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        try:
            descriptor = os.open(destination, flags, 0o600)
        except OSError:
            _refuse("unable to create an allowlisted destination without overwrite")
        try:
            os.fchmod(descriptor, 0o600)
            view = memoryview(artifact.content)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    _refuse("unable to write a staged artifact completely")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directory_flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        directory_flags |= os.O_DIRECTORY
    descriptor = os.open(output, directory_flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def stage_public_release(
    *,
    manifest_path: Path,
    source_root: Path,
    output_directory: Path,
    expected_kind: ReleaseArtifactKind,
    expected_source: DataSource,
    expected_classification: DataClassification,
    license_clearance_path: Path | None = None,
) -> StagingResult:
    """Validate and stage one explicit, homogeneous release allowlist.

    All source artifacts are read, hash-checked, and content-validated before
    the fresh output directory is created.  The caller must repeat the kind,
    source, and classification declared by the release manifest so a wrong
    manifest cannot silently select a weaker release lane.
    """

    if not isinstance(expected_kind, ReleaseArtifactKind):
        _refuse("expected_kind must be an explicit ReleaseArtifactKind")
    if not isinstance(expected_source, DataSource):
        _refuse("expected_source must be an explicit registered DataSource")
    if not isinstance(expected_classification, DataClassification):
        _refuse("expected_classification must be an explicit DataClassification")

    manifest_absolute = _absolute(manifest_path)
    manifest_payload = _read_regular_file(manifest_absolute)
    manifest_sha256 = hashlib.sha256(manifest_payload).hexdigest()
    manifest = _parse_release_manifest(
        _parse_control_json(manifest_payload, name="release manifest")
    )
    if (
        manifest.artifact_kind is not expected_kind
        or manifest.benchmark_source is not expected_source
        or manifest.classification is not expected_classification
    ):
        _refuse("explicit caller declarations do not match the release manifest")

    license_absolute: Path | None = None
    if manifest.artifact_kind is ReleaseArtifactKind.PUBLIC_TRACE:
        if license_clearance_path is None:
            _refuse("public traces require a checked license-clearance manifest")
        license_absolute = _absolute(license_clearance_path)
        license_payload = _read_regular_file(license_absolute)
        _validate_license_manifest(
            _parse_control_json(license_payload, name="license-clearance manifest"),
            release_manifest_sha256=manifest_sha256,
            release_manifest=manifest,
        )
    elif license_clearance_path is not None:
        _refuse("aggregate-only and trace-license workflows must remain separate")

    source_root_absolute = _ensure_existing_directory(source_root)
    output_absolute = _absolute(output_directory)
    output_parent = _ensure_existing_directory(output_absolute.parent)
    output_absolute = output_parent / output_absolute.name
    if os.path.lexists(output_absolute):
        _refuse("output destination already exists; overwrite is forbidden")
    forbidden_control_paths = {manifest_absolute}
    if license_absolute is not None:
        forbidden_control_paths.add(license_absolute)

    artifacts = [
        _prepare_artifact(
            source_root_absolute,
            manifest_file,
            release_manifest=manifest,
            forbidden_control_paths=forbidden_control_paths,
        )
        for manifest_file in manifest.files
    ]
    _write_prepared_artifacts(output_absolute, artifacts)
    return StagingResult(
        artifact_kind=manifest.artifact_kind,
        benchmark_source=manifest.benchmark_source,
        files_staged=len(artifacts),
        bytes_staged=sum(len(artifact.content) for artifact in artifacts),
        output_directory=output_absolute,
    )
