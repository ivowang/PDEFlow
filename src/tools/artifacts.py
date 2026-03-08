from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

try:
    import h5py
except ImportError:  # pragma: no cover - runtime environments install h5py for real validation
    h5py = None

from common import now_utc, short_hash
from config import RetrievalConfig
from state import (
    ArtifactChecksumRecord,
    ArtifactRecord,
    ArtifactStatus,
    ArtifactValidationResult,
)


class ArtifactValidationMixin:
    def compute_file_checksum(self, path: str, algorithm: str = "md5") -> dict[str, Any]:
        resolved = self._resolve_path(path)
        digest = hashlib.new(algorithm)
        with resolved.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        payload = {
            "path": str(resolved),
            "algorithm": algorithm,
            "checksum": digest.hexdigest(),
        }
        self._record_tool_event("compute_file_checksum", payload)
        return payload

    def _artifact_min_size(self, artifact: ArtifactRecord, retrieval: RetrievalConfig) -> int | None:
        metadata = artifact.metadata or {}
        candidates = [
            metadata.get("min_size_bytes"),
            metadata.get("expected_min_size_bytes"),
            metadata.get("expected_size_bytes"),
        ]
        for candidate in candidates:
            try:
                if candidate is not None:
                    return int(candidate)
            except (TypeError, ValueError):
                continue
        if artifact.artifact_type == "dataset":
            return retrieval.default_min_dataset_size_bytes
        return None

    def _extract_checksum_record(self, artifact: ArtifactRecord, actual: str | None = None) -> ArtifactChecksumRecord | None:
        metadata = artifact.metadata or {}
        expected = (
            metadata.get("official_md5")
            or metadata.get("official_checksum")
            or metadata.get("expected_md5")
            or metadata.get("expected_checksum")
        )
        checksum_source = None
        if metadata.get("official_md5") or metadata.get("official_checksum"):
            checksum_source = "official_registry"
        elif expected:
            checksum_source = "artifact_metadata"
        algorithm = str(metadata.get("checksum_algorithm") or "md5")
        if expected is None and actual is None:
            return None
        matched = None
        if expected is not None and actual is not None:
            matched = str(expected).strip().lower() == str(actual).strip().lower()
        return ArtifactChecksumRecord(
            algorithm=algorithm,
            expected=str(expected) if expected is not None else None,
            actual=actual,
            source=checksum_source,
            matched=matched,
        )

    def _validate_hdf5(
        self,
        path: Path,
        artifact: ArtifactRecord,
        size_bytes: int,
        min_size_bytes: int | None,
        size_ok: bool,
        checksum: ArtifactChecksumRecord | None,
    ) -> ArtifactValidationResult:
        failure_reasons: list[str] = []
        top_level_keys: list[str] = []
        sample_target: str | None = None
        sample_shape: list[int] = []
        details: dict[str, Any] = {}
        required_keys = list((artifact.metadata or {}).get("required_keys", []))
        format_valid = False
        if h5py is None:
            failure_reasons.append("h5py_unavailable")
            return ArtifactValidationResult(
                validator="hdf5",
                status=ArtifactStatus.CORRUPTED,
                exists=True,
                size_bytes=size_bytes,
                min_size_bytes=min_size_bytes,
                size_ok=size_ok,
                format_valid=False,
                ready_for_training=False,
                checksum=checksum,
                failure_reasons=failure_reasons,
                details={"path": str(path)},
            )
        try:
            with h5py.File(path, "r") as handle:
                top_level_keys = sorted(handle.keys())
                details["top_level_key_count"] = len(top_level_keys)
                missing_required = [key for key in required_keys if key not in handle]
                if missing_required:
                    failure_reasons.append(f"missing_required_keys:{','.join(missing_required)}")
                sample_dataset = None
                for key in top_level_keys:
                    item = handle[key]
                    if hasattr(item, "shape"):
                        sample_dataset = item
                        sample_target = key
                        sample_shape = [int(dim) for dim in getattr(item, "shape", ()) if isinstance(dim, (int, float))]
                        break
                if sample_dataset is not None:
                    try:
                        if getattr(sample_dataset, "ndim", 0) == 0:
                            _ = sample_dataset[()]
                        else:
                            index = tuple(0 for _ in range(getattr(sample_dataset, "ndim", 0)))
                            _ = sample_dataset[index]
                    except Exception as exc:  # pragma: no cover - h5py error variants depend on file damage
                        failure_reasons.append(f"sample_read_failed:{exc}")
                format_valid = not failure_reasons
        except Exception as exc:
            failure_reasons.append(f"hdf5_open_failed:{exc}")
            format_valid = False

        ready = size_ok and format_valid and (checksum is None or checksum.matched is not False)
        if not size_ok:
            failure_reasons.insert(0, f"size_below_minimum:{size_bytes}<{min_size_bytes}")
        if checksum is not None and checksum.expected is not None and checksum.matched is False:
            failure_reasons.append("checksum_mismatch")
        if ready:
            status = ArtifactStatus.READY_FOR_TRAINING
        elif checksum is not None and checksum.matched:
            status = ArtifactStatus.CHECKSUM_VERIFIED if not format_valid else ArtifactStatus.FORMAT_VERIFIED
        else:
            status = ArtifactStatus.CORRUPTED
        return ArtifactValidationResult(
            validator="hdf5",
            status=status,
            exists=True,
            size_bytes=size_bytes,
            min_size_bytes=min_size_bytes,
            size_ok=size_ok,
            format_valid=format_valid,
            ready_for_training=ready,
            top_level_keys=top_level_keys,
            sample_read_target=sample_target,
            sample_shape=sample_shape,
            checksum=checksum,
            failure_reasons=failure_reasons,
            details=details,
        )

    def _validate_generic_file(
        self,
        path: Path,
        artifact: ArtifactRecord,
        size_bytes: int,
        min_size_bytes: int | None,
        size_ok: bool,
        checksum: ArtifactChecksumRecord | None,
    ) -> ArtifactValidationResult:
        failure_reasons: list[str] = []
        if not size_ok:
            failure_reasons.append(f"size_below_minimum:{size_bytes}<{min_size_bytes}")
        if checksum is not None and checksum.expected is not None and checksum.matched is False:
            failure_reasons.append("checksum_mismatch")
        ready = size_ok and (checksum is None or checksum.matched is not False)
        if ready:
            status = ArtifactStatus.READY_FOR_TRAINING
        elif checksum is not None and checksum.matched:
            status = ArtifactStatus.CHECKSUM_VERIFIED
        else:
            status = ArtifactStatus.CORRUPTED if failure_reasons else ArtifactStatus.DOWNLOADED
        return ArtifactValidationResult(
            validator="generic_file",
            status=status,
            exists=True,
            size_bytes=size_bytes,
            min_size_bytes=min_size_bytes,
            size_ok=size_ok,
            format_valid=None,
            ready_for_training=ready,
            checksum=checksum,
            failure_reasons=failure_reasons,
            details={"suffix": path.suffix.lower()},
        )

    def _quarantine_artifact(self, artifact: ArtifactRecord, path: Path) -> str:
        quarantine_dir = self.quarantine_root / artifact.artifact_id
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        target = quarantine_dir / f"{path.name}.{short_hash(path.name, now_utc())}"
        shutil.move(str(path), str(target))
        return str(target)

    def validate_artifact_record(
        self,
        artifact: ArtifactRecord,
        quarantine_on_failure: bool = True,
    ) -> ArtifactRecord:
        if not artifact.local_path:
            validation = ArtifactValidationResult(
                validator="missing_local_path",
                status=ArtifactStatus.BLOCKED,
                exists=False,
                ready_for_training=False,
                failure_reasons=["missing_local_path"],
                details={"artifact_type": artifact.artifact_type},
            )
            updated = artifact.model_copy(update={"validation": validation})
            self._record_tool_event(
                "validate_artifact",
                {
                    "artifact_id": updated.artifact_id,
                    "status": updated.status,
                    "ready_for_training": False,
                },
            )
            return updated

        path = self._resolve_path(artifact.local_path, default_root=self.shared_workspace_root)
        if not path.exists() or not path.is_file():
            validation = ArtifactValidationResult(
                validator="missing_file",
                status=ArtifactStatus.BLOCKED,
                exists=False,
                ready_for_training=False,
                failure_reasons=["file_missing"],
                details={"path": str(path)},
            )
            updated = artifact.model_copy(update={"validation": validation, "status": ArtifactStatus.BLOCKED.value})
            self._record_tool_event(
                "validate_artifact",
                {
                    "artifact_id": updated.artifact_id,
                    "status": updated.status,
                    "ready_for_training": False,
                },
            )
            return updated

        size_bytes = path.stat().st_size
        min_size = self._artifact_min_size(artifact, self.config.retrieval)
        size_ok = min_size is None or size_bytes >= min_size

        checksum = self._extract_checksum_record(artifact)
        if checksum is not None and checksum.expected is not None:
            actual = self.compute_file_checksum(str(path), algorithm=checksum.algorithm)["checksum"]
            checksum = self._extract_checksum_record(artifact, actual=actual)

        suffix = path.suffix.lower()
        if suffix in {".h5", ".hdf5"}:
            validation = self._validate_hdf5(path, artifact, size_bytes, min_size, size_ok, checksum)
        else:
            validation = self._validate_generic_file(path, artifact, size_bytes, min_size, size_ok, checksum)

        update_payload: dict[str, Any] = {
            "validation": validation,
            "status": validation.status.value,
            "download_metadata": artifact.download_metadata.model_copy(
                update={
                    "file_size": size_bytes,
                    "validation_status": validation.status.value,
                    "checksum": checksum,
                }
            )
            if artifact.download_metadata
            else None,
            "quarantine_path": artifact.quarantine_path,
        }
        if not validation.ready_for_training and validation.status == ArtifactStatus.CORRUPTED and quarantine_on_failure:
            if self.config.retrieval.quarantine_corrupted_artifacts:
                quarantine_path = self._quarantine_artifact(artifact, path)
                validation = validation.model_copy(update={"status": ArtifactStatus.QUARANTINED})
                update_payload["validation"] = validation
                update_payload["status"] = ArtifactStatus.QUARANTINED.value
                update_payload["quarantine_path"] = quarantine_path
        updated = artifact.model_copy(update=update_payload)
        self._record_tool_event(
            "validate_artifact",
            {
                "artifact_id": updated.artifact_id,
                "status": updated.status,
                "ready_for_training": validation.ready_for_training,
            },
        )
        return updated
