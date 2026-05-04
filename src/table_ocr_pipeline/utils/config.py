from __future__ import annotations

from pathlib import Path
from typing import Any, Dict


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    text = path.read_text(encoding="utf-8")

    # Preferred: yaml
    try:
        import yaml
        return yaml.safe_load(text) or {}
    except Exception:
        return _parse_simple_yaml(text)


# SIMPLE YAML (fallback only)
def _parse_simple_yaml(text: str) -> Dict[str, Any]:
    config: Dict[str, Any] = {}
    current = None

    for raw in text.splitlines():
        line = raw.split("#", 1)[0].rstrip()

        if not line:
            continue

        # section
        if not raw.startswith(" ") and line.endswith(":"):
            section = line[:-1]
            config[section] = {}
            current = config[section]
            continue

        if current is None or ":" not in line:
            continue

        key, value = map(str.strip, line.split(":", 1))
        current[key] = _parse_scalar(value)

    return config


def _parse_scalar(value: str) -> Any:
    v = value.lower()

    if v in {"true", "false"}:
        return v == "true"

    if value.startswith("[") and value.endswith("]"):
        return [x.strip() for x in value[1:-1].split(",") if x.strip()]

    for caster in (int, float):
        try:
            return caster(value)
        except ValueError:
            pass

    return value