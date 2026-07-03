"""
Domain plugin registry — the only place core resolves a domain.

Resolution order (explicit before heuristic):
1. ``payload["domain"]`` / ``payload["domain_profile"]`` exact id match.
2. ``[domain_profile:<id>]`` tag in the question.
3. Each plugin's own ``matches(question, payload)`` (domain-owned heuristics).

Core code calls :func:`resolve_domain_plugin` / :func:`get_domain_plugin`; it
must not import a specific plugin module or inspect question text for domain
keywords itself.
"""
from __future__ import annotations

import re
from typing import Any

from propab.domain_modules.base import DomainPlugin

_TAG = re.compile(r"\[domain_profile:([a-z0-9_]+)\]", re.IGNORECASE)

_PLUGINS: list[DomainPlugin] = []
_BY_ID: dict[str, DomainPlugin] = {}
_LOADED = False


def register_plugin(plugin: DomainPlugin) -> None:
    """Register a plugin instance. Later registration of the same id overrides."""
    if not plugin.domain_id:
        raise ValueError("DomainPlugin.domain_id must be set")
    _BY_ID[plugin.domain_id] = plugin
    _PLUGINS[:] = [p for p in _PLUGINS if p.domain_id != plugin.domain_id]
    _PLUGINS.append(plugin)


def _ensure_loaded() -> None:
    """Import and register the built-in plugins exactly once."""
    global _LOADED
    if _LOADED:
        return
    _LOADED = True
    # Ordered by detection precedence: materials before mandrake preserves the
    # historical worker ordering (is_materials_campaign checked before mandrake).
    from propab.domain_modules.materials.plugin import MaterialsPlugin
    from propab.domain_modules.mandrake.plugin import MandrakePlugin
    from propab.domain_modules.enzyme_kinetics.plugin import EnzymeKineticsPlugin
    from propab.domain_modules.graph_invariants.plugin import GraphInvariantsPlugin
    from propab.domain_modules.network_diffusion.plugin import NetworkDiffusionPlugin
    from propab.domain_modules.math_combinatorics.plugin import MathCombinatoricsPlugin

    for plugin_cls in (
        MaterialsPlugin,
        MandrakePlugin,
        EnzymeKineticsPlugin,
        GraphInvariantsPlugin,
        NetworkDiffusionPlugin,
        MathCombinatoricsPlugin,
    ):
        try:
            register_plugin(plugin_cls())
        except Exception:  # noqa: BLE001 — a broken domain must not break the registry
            continue


def all_plugins() -> tuple[DomainPlugin, ...]:
    _ensure_loaded()
    return tuple(_PLUGINS)


def get_domain_plugin(domain_id: str | None) -> DomainPlugin | None:
    _ensure_loaded()
    if not domain_id:
        return None
    return _BY_ID.get(str(domain_id).strip().lower())


def resolve_domain_plugin(
    *,
    question: str = "",
    payload: dict[str, Any] | None = None,
) -> DomainPlugin | None:
    """Resolve the owning plugin from explicit signals, then plugin matchers."""
    _ensure_loaded()
    payload = payload or {}

    explicit = str(
        payload.get("domain_profile")
        or payload.get("domain_profile_id")
        or payload.get("domain")
        or ""
    ).strip().lower()
    if explicit and explicit in _BY_ID:
        return _BY_ID[explicit]

    m = _TAG.search(question or "")
    if m:
        tagged = m.group(1).lower()
        if tagged in _BY_ID:
            return _BY_ID[tagged]

    for plugin in _PLUGINS:
        try:
            if plugin.matches(question=question, payload=payload):
                return plugin
        except Exception:  # noqa: BLE001 — a broken matcher must not break routing
            continue
    return None


def hypothesis_is_on_topic(
    text: str,
    *,
    question: str = "",
    domain_id: str | None = None,
) -> bool:
    """Return False when a domain plugin rejects hypothesis text as off-topic."""
    plugin = get_domain_plugin(domain_id) if domain_id else resolve_domain_plugin(question=question)
    if plugin is None:
        return True
    return plugin.hypothesis_on_topic(text)
