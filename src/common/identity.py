from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

from .utils import slugify


_HEX_SUFFIX_RE = re.compile(r"[-_][0-9a-f]{8,}$")
_PARAM_RE = re.compile(r"(nu|rho)[-_]?([0-9]+(?:[.p][0-9]+)?)", flags=re.IGNORECASE)
_OWNER_REPO_RE = re.compile(r"github\.com[:/]+([^/\s]+)/([^/\s]+?)(?:\.git)?/?$", flags=re.IGNORECASE)


def normalize_numeric_token(value: str | float | int | None) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    text = text.replace("p", ".")
    try:
        numeric = float(text)
    except ValueError:
        return slugify(text)
    normalized = f"{numeric:.12g}"
    return normalized.replace(".", "p")


def _normalized_tokens(text: str) -> list[str]:
    cleaned = slugify(text).replace("_", "-")
    return [token for token in cleaned.split("-") if token]


def canonicalize_env_id(value: str, project_hint: str | None = None) -> str:
    base = project_hint or value
    normalized = slugify(base).replace("_", "-")
    normalized = _HEX_SUFFIX_RE.sub("", normalized)
    normalized = re.sub(r"[-_]?env$", "", normalized)
    return f"{normalized or 'environment'}-env"


def canonicalize_repo_id(name: str, remote_url: str | None = None) -> str:
    if remote_url:
        match = _OWNER_REPO_RE.search(remote_url)
        if match:
            owner = slugify(match.group(1))
            repo = slugify(match.group(2).replace(".git", ""))
            return f"github-{owner}-{repo}"
    normalized = slugify(name).replace("_", "-")
    normalized = normalized.removesuffix("-official")
    normalized = _HEX_SUFFIX_RE.sub("", normalized)
    return normalized or "repository"


def repo_resolution_keys(name: str, remote_url: str | None = None) -> list[str]:
    keys = {canonicalize_repo_id(name, remote_url)}
    if remote_url:
        keys.add(remote_url.rstrip("/"))
    keys.add(slugify(name))
    return sorted(item for item in keys if item)


def infer_dataset_semantic_spec(*values: str) -> dict[str, str]:
    joined = " ".join(value for value in values if value)
    lowered = joined.lower()
    spec: dict[str, str] = {}
    if "pdebench" in lowered:
        spec["benchmark"] = "PDEBench"
    if any(token in lowered for token in ("reacdiff", "reactiondiffusion", "reaction-diffusion")):
        spec["equation"] = "ReactionDiffusion"
    elif "burgers" in lowered:
        spec["equation"] = "Burgers"
    elif "diffusion" in lowered and "reaction" in lowered:
        spec["equation"] = "ReactionDiffusion"
    if any(token in lowered for token in ("/train/", "-train", "_train", " split train", "\\train\\")):
        spec["split"] = "train"
    elif any(token in lowered for token in ("/test/", "-test", "_test", " split test", "\\test\\")):
        spec["split"] = "test"
    params: dict[str, str] = {}
    for name, raw in _PARAM_RE.findall(joined):
        normalized = normalize_numeric_token(raw)
        if normalized:
            params[name.lower()] = normalized
    spec.update(params)
    return spec


def canonicalize_artifact_id(
    artifact_id: str,
    local_path: str | None = None,
    title: str | None = None,
    metadata: dict[str, object] | None = None,
    artifact_type: str | None = None,
) -> tuple[str, dict[str, str]]:
    metadata = metadata or {}
    sources = [
        artifact_id,
        title or "",
        local_path or "",
        str(metadata.get("expected_filename") or ""),
        str(metadata.get("benchmark") or ""),
        str(metadata.get("equation") or ""),
    ]
    spec = infer_dataset_semantic_spec(*sources)
    if benchmark := metadata.get("benchmark"):
        spec["benchmark"] = str(benchmark)
    if equation := metadata.get("equation"):
        spec["equation"] = str(equation)
    if split := metadata.get("split"):
        spec["split"] = str(split).lower()
    if nu := normalize_numeric_token(metadata.get("nu")):
        spec["nu"] = nu
    if rho := normalize_numeric_token(metadata.get("rho")):
        spec["rho"] = rho
    filename = Path(local_path or metadata.get("expected_filename") or title or artifact_id).name
    if filename:
        spec["filename"] = filename
    family = slugify(artifact_type or metadata.get("asset_family") or "artifact").replace("_", "-")
    parts = [slugify(spec.get("benchmark", "artifact")), family]
    if equation := spec.get("equation"):
        parts.append(slugify(equation))
    if split := spec.get("split"):
        parts.append(slugify(split))
    if nu := spec.get("nu"):
        parts.append(f"nu-{nu}")
    if rho := spec.get("rho"):
        parts.append(f"rho-{rho}")
    if len(parts) <= 2:
        normalized = slugify(artifact_id).replace("_", "-")
        normalized = normalized.removesuffix("-official").replace("--", "-")
        normalized = _HEX_SUFFIX_RE.sub("", normalized)
        return normalized or "artifact", spec
    return "-".join(part for part in parts if part), spec


def choose_preferred_identifier(aliases: Iterable[str]) -> str:
    best = ""
    for alias in aliases:
        if not alias:
            continue
        if not best or len(alias) < len(best):
            best = alias
    return best or "artifact"


def canonicalize_source_url(url: str) -> str:
    parsed = urlparse(url)
    normalized_path = parsed.path.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}{normalized_path}" if parsed.scheme and parsed.netloc else url
