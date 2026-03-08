from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from common import ensure_dir


class ReportingToolsMixin:
    def parse_json_file(self, path: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        payload = json.loads(resolved.read_text(encoding="utf-8"))
        result = {"path": str(resolved), "payload": payload}
        self._record_tool_event("parse_json_file", {"path": str(resolved)})
        return result

    def parse_metrics_file(self, path: str) -> dict[str, Any]:
        resolved = self._resolve_path(path)
        suffix = resolved.suffix.lower()
        if suffix == ".json":
            payload = json.loads(resolved.read_text(encoding="utf-8"))
            result = {"path": str(resolved), "metrics": payload}
            self._record_tool_event("parse_metrics_file", {"path": str(resolved), "format": "json"})
            return result
        text = resolved.read_text(encoding="utf-8")
        metrics: dict[str, Any] = {}
        for match in re.finditer(r"([A-Za-z0-9_./-]+)\s*[:=]\s*(-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)", text):
            key = match.group(1)
            value = match.group(2)
            metrics[key] = float(value)
        result = {"path": str(resolved), "metrics": metrics}
        self._record_tool_event("parse_metrics_file", {"path": str(resolved), "format": "text", "count": len(metrics)})
        return result

    def write_report(self, filename: str, content: str) -> Path:
        path = self.memory.reports_dir / filename
        ensure_dir(path.parent)
        path.write_text(content, encoding="utf-8")
        self._record_tool_event("write_report", {"path": str(path), "chars": len(content)})
        return path
