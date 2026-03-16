from __future__ import annotations

import hashlib
import json
import shlex
import shutil
from pathlib import Path
import subprocess
import sys
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
    def _effective_size_ok(self, size_ok: bool, checksum: ArtifactChecksumRecord | None) -> bool:
        return size_ok or bool(checksum is not None and checksum.matched is True)

    def _candidate_hdf5_validation_pythons(self) -> list[str]:
        candidates = sorted(self.managed_env_root.glob("*/bin/python"))
        sibling_candidates = sorted(self.workspace_family_root.glob("*/envs/*/bin/python"))
        candidates.extend(sibling_candidates)
        candidates.append(Path(sys.executable))
        seen: set[str] = set()
        ordered: list[str] = []
        for candidate in candidates:
            absolute = str(candidate.absolute())
            if absolute not in seen and Path(absolute).exists():
                seen.add(absolute)
                ordered.append(absolute)
        return ordered

    def _ensure_hdf5_validator_python(self) -> str | None:
        validator_env = self.managed_env_root / "artifact-validator"
        validator_python = validator_env / "bin" / "python"
        if validator_python.exists():
            return str(validator_python)
        if not self.config.execution.auto_bootstrap_environments or not self.config.execution.allow_package_installation:
            return None
        bootstrap_python = shutil.which("python3") or sys.executable
        create_command = (
            "uv venv --seed --allow-existing "
            f"--python {shlex.quote(str(bootstrap_python))} {shlex.quote(str(validator_env))}"
        )
        create_result = self.run_command(create_command, cwd=self.repo_root, allow_failure=True)
        if create_result["returncode"] != 0 or not validator_python.exists():
            return None
        install_command = (
            f"uv pip install --python {shlex.quote(str(validator_python))} h5py"
        )
        install_result = self.run_command(install_command, cwd=self.repo_root, allow_failure=True)
        if install_result["returncode"] != 0:
            return None
        return str(validator_python)

    def _validate_hdf5_via_subprocess(
        self,
        path: Path,
        required_keys: list[str],
    ) -> dict[str, Any] | None:
        script = (
            "import json\n"
            "import h5py\n"
            f"path = {path.as_posix()!r}\n"
            f"required_keys = {required_keys!r}\n"
            "payload = {'format_valid': False, 'top_level_keys': [], 'sample_read_target': None, 'sample_shape': [], 'failure_reasons': []}\n"
            "try:\n"
            "    with h5py.File(path, 'r') as handle:\n"
            "        keys = sorted(handle.keys())\n"
            "        payload['top_level_keys'] = keys\n"
            "        missing = [key for key in required_keys if key not in handle]\n"
            "        if missing:\n"
            "            payload['failure_reasons'].append('missing_required_keys:' + ','.join(missing))\n"
            "        sample = None\n"
            "        for key in keys:\n"
            "            item = handle[key]\n"
            "            if hasattr(item, 'shape'):\n"
            "                sample = item\n"
            "                payload['sample_read_target'] = key\n"
            "                payload['sample_shape'] = [int(dim) for dim in getattr(item, 'shape', ()) if isinstance(dim, (int, float))]\n"
            "                break\n"
            "        if sample is not None:\n"
            "            try:\n"
            "                if getattr(sample, 'ndim', 0) == 0:\n"
            "                    _ = sample[()]\n"
            "                else:\n"
            "                    index = tuple(0 for _ in range(getattr(sample, 'ndim', 0)))\n"
            "                    _ = sample[index]\n"
            "            except Exception as exc:\n"
            "                payload['failure_reasons'].append('sample_read_failed:' + str(exc))\n"
            "        payload['format_valid'] = not payload['failure_reasons']\n"
            "except Exception as exc:\n"
            "    payload['failure_reasons'].append('hdf5_open_failed:' + str(exc))\n"
            "print(json.dumps(payload))\n"
        )
        candidate_pythons = self._candidate_hdf5_validation_pythons()
        extra_validator = self._ensure_hdf5_validator_python()
        if extra_validator and extra_validator not in candidate_pythons:
            candidate_pythons.insert(0, extra_validator)
        for python_executable in candidate_pythons:
            try:
                probe = subprocess.run(
                    [python_executable, "-c", script],
                    cwd=str(path.parent),
                    capture_output=True,
                    text=True,
                    timeout=120,
                    check=False,
                )
            except (OSError, subprocess.SubprocessError):
                continue
            stdout = probe.stdout or ""
            if probe.returncode != 0 or not stdout.strip():
                continue
            try:
                payload = json.loads(stdout.splitlines()[-1])
            except json.JSONDecodeError:
                continue
            payload["validator_python"] = python_executable
            return payload
        return None

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
        download_checksum = artifact.download_metadata.checksum if artifact.download_metadata else None
        expected = (
            metadata.get("official_md5")
            or metadata.get("official_checksum")
            or metadata.get("expected_md5")
            or metadata.get("expected_checksum")
            or (download_checksum.expected if download_checksum else None)
        )
        checksum_source = None
        if metadata.get("official_md5") or metadata.get("official_checksum"):
            checksum_source = "official_registry"
        elif expected:
            checksum_source = "artifact_metadata"
        if checksum_source is None and download_checksum and download_checksum.expected:
            checksum_source = download_checksum.source or "download_metadata"
        algorithm = str(metadata.get("checksum_algorithm") or (download_checksum.algorithm if download_checksum else None) or "md5")
        if expected is None and actual is None:
            return None
        matched = None
        if expected is not None and actual is not None:
            matched = str(expected).strip().lower() == str(actual).strip().lower()
        elif download_checksum is not None:
            matched = download_checksum.matched
        return ArtifactChecksumRecord(
            algorithm=algorithm,
            expected=str(expected) if expected is not None else None,
            actual=actual or (download_checksum.actual if download_checksum else None),
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
        effective_size_ok = self._effective_size_ok(size_ok, checksum)
        if h5py is None:
            subprocess_payload = self._validate_hdf5_via_subprocess(path, required_keys)
            if subprocess_payload is None:
                failure_reasons.append("format_validator_unavailable")
                if not effective_size_ok:
                    failure_reasons.insert(0, f"size_below_minimum:{size_bytes}<{min_size_bytes}")
                if checksum is not None and checksum.expected is not None and checksum.matched is False:
                    failure_reasons.append("checksum_mismatch")
                if checksum is not None and checksum.matched:
                    status = ArtifactStatus.CHECKSUM_VERIFIED
                elif effective_size_ok:
                    status = ArtifactStatus.DOWNLOADED
                else:
                    status = ArtifactStatus.CORRUPTED
                return ArtifactValidationResult(
                    validator="hdf5",
                    status=status,
                    exists=True,
                    size_bytes=size_bytes,
                    min_size_bytes=min_size_bytes,
                    size_ok=size_ok,
                    format_valid=None,
                    ready_for_training=False,
                    checksum=checksum,
                    failure_reasons=failure_reasons,
                    details={"path": str(path)},
                )
            top_level_keys = list(subprocess_payload.get("top_level_keys", []))
            sample_target = subprocess_payload.get("sample_read_target")
            sample_shape = list(subprocess_payload.get("sample_shape", []))
            failure_reasons.extend(list(subprocess_payload.get("failure_reasons", [])))
            format_valid = bool(subprocess_payload.get("format_valid", False))
            details.update({"validator_python": subprocess_payload.get("validator_python")})
            ready = effective_size_ok and format_valid and (checksum is None or checksum.matched is not False)
            if not effective_size_ok:
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

        ready = effective_size_ok and format_valid and (checksum is None or checksum.matched is not False)
        if not effective_size_ok:
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
        effective_size_ok = self._effective_size_ok(size_ok, checksum)
        if not effective_size_ok:
            failure_reasons.append(f"size_below_minimum:{size_bytes}<{min_size_bytes}")
        if checksum is not None and checksum.expected is not None and checksum.matched is False:
            failure_reasons.append("checksum_mismatch")
        ready = effective_size_ok and (checksum is None or checksum.matched is not False)
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
            self.memory.record_artifact(updated)
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
            self.memory.record_artifact(updated)
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
        metadata = artifact.metadata or {}
        download_metadata = artifact.download_metadata
        allow_quarantine = quarantine_on_failure
        if metadata.get("human_provided") or metadata.get("manual_imported"):
            allow_quarantine = False
        if download_metadata and (
            download_metadata.strategy_id in {"local_discovery", "manual_local_import"}
            or download_metadata.source_type == "human_provided_local"
        ):
            allow_quarantine = False
        if not validation.ready_for_training and validation.status == ArtifactStatus.CORRUPTED and quarantine_on_failure:
            if allow_quarantine and self.config.retrieval.quarantine_corrupted_artifacts:
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
        self.memory.record_artifact(updated)
        return updated
