"""Agent Manifest loader — discover, parse, and template agent YAML files.

Typical usage::

    loader = ManifestLoader(Path("agents"))
    manifests = loader.load_all()
    rendered = loader.render_prompt(manifests["summariser"], extra="Be concise.")
"""

from __future__ import annotations

import json
from pathlib import Path  # noqa: TC003
from string import Template
from typing import Any

from uac.core.orchestration.models import AgentManifest

# YAML parsing is deferred — we support both PyYAML and the stdlib JSON
# as a fallback (manifests can be JSON too).
_yaml_load: Any = None


def _get_yaml_loader() -> Any:
    """Return ``yaml.safe_load`` or raise if PyYAML is not installed."""
    global _yaml_load
    if _yaml_load is None:
        try:
            import yaml  # pyright: ignore[reportMissingImports]

            _yaml_load = yaml.safe_load
        except ImportError as exc:
            raise ImportError(
                "PyYAML is required for YAML manifest loading. "
                "Install it with: pip install pyyaml"
            ) from exc
    return _yaml_load


def parse_manifest(raw: str, *, format: str = "yaml") -> AgentManifest:
    """Parse a raw string into a validated :class:`AgentManifest`.

    Args:
        raw: The raw file contents.
        format: ``"yaml"`` (default) or ``"json"``.
    """
    if format == "json":
        data = json.loads(raw)
    else:
        loader = _get_yaml_loader()
        data = loader(raw)
    return AgentManifest.model_validate(data)


def render_prompt(manifest: AgentManifest, **variables: Any) -> str:
    """Render the manifest's ``system_prompt_template`` with the given variables.

    Uses Python's :class:`string.Template` (``$name`` / ``${name}`` syntax)
    for safe, dependency-free templating.  The manifest's own fields are
    available as default variables (``name``, ``description``, ``version``).

    Unknown placeholders are left as-is (``safe_substitute``).
    """
    defaults: dict[str, Any] = {
        "name": manifest.name,
        "description": manifest.description,
        "version": manifest.version,
    }
    merged = {**defaults, **variables}
    return Template(manifest.system_prompt_template).safe_substitute(merged)


class ManifestLoader:
    """Load and cache agent manifests from a directory.

    Scans the directory for ``.yaml``, ``.yml``, and ``.json`` files.
    Each file is expected to contain a single agent manifest.
    """

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self._cache: dict[str, AgentManifest] = {}

    def load_all(self) -> dict[str, AgentManifest]:
        """Load all manifests from the directory, keyed by agent name.

        Re-reads from disk every call (the cache is per-call).
        """
        manifests: dict[str, AgentManifest] = {}
        if not self.directory.is_dir():
            return manifests

        for path in sorted(self.directory.iterdir()):
            if path.suffix in (".yaml", ".yml", ".json"):
                manifest = self._load_file(path)
                manifests[manifest.name] = manifest

        self._cache = manifests
        return manifests

    def load_one(self, name: str) -> AgentManifest:
        """Load a single manifest by agent name (searches directory).

        Raises:
            FileNotFoundError: If no matching file is found.
        """
        if name in self._cache:
            return self._cache[name]

        for path in self.directory.iterdir():
            if path.suffix in (".yaml", ".yml", ".json"):
                manifest = self._load_file(path)
                self._cache[manifest.name] = manifest
                if manifest.name == name:
                    return manifest

        raise FileNotFoundError(f"No manifest found for agent '{name}' in {self.directory}")

    def _load_file(self, path: Path) -> AgentManifest:
        """Read and parse a single manifest file."""
        raw = path.read_text(encoding="utf-8")
        fmt = "json" if path.suffix == ".json" else "yaml"
        return parse_manifest(raw, format=fmt)
