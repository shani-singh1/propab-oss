"""
Domain plugin registry — the only place core resolves a domain.

Resolution order (explicit before heuristic):
1. ``payload["domain"]`` / ``payload["domain_profile"]`` exact id match.
2. ``[domain_profile:<id>]`` tag in the question.
3. Heuristic: every plugin that ``matches(question, payload)`` is scored via
   ``match_score(...)`` and the HIGHEST-scoring plugin wins. Registration order
   is only the final, deterministic tie-break — an older plugin can no longer
   silently shadow a better-matching one on a keyword collision. Near-ties (top
   two scores within a small margin) are logged as a routing ambiguity.

Core code calls :func:`resolve_domain_plugin` / :func:`get_domain_plugin`; it
must not import a specific plugin module or inspect question text for domain
keywords itself.
"""
from __future__ import annotations

import logging
import re
from typing import Any

from propab.domain_modules.base import DomainPlugin

logger = logging.getLogger(__name__)

_TAG = re.compile(r"\[domain_profile:([a-z0-9_]+)\]", re.IGNORECASE)

_PLUGINS: list[DomainPlugin] = []
_BY_ID: dict[str, DomainPlugin] = {}
_LOADED = False

# Two matching plugins whose scores differ by no more than this are treated as a
# genuine routing ambiguity: the higher score still wins (deterministically), but
# the near-tie is logged so a real collision is visible instead of silently
# resolved on registration order.
_AMBIGUITY_MARGIN = 1.0


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
    from propab.domain_modules.genomics.plugin import GenomicsPlugin

    for plugin_cls in (
        MaterialsPlugin,
        MandrakePlugin,
        EnzymeKineticsPlugin,
        GraphInvariantsPlugin,
        NetworkDiffusionPlugin,
        MathCombinatoricsPlugin,
        GenomicsPlugin,
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

    # Heuristic routing: ask every plugin how strongly it claims the campaign and
    # pick the MAX score — not the first match. Collisions (two plugins whose
    # keyword sets overlap) now resolve to the better-matching domain instead of
    # whichever was registered first. Registration order survives only as the
    # final, deterministic tie-break (enumerate index), so behaviour is identical
    # when exactly one plugin matches.
    scored: list[tuple[float, int, DomainPlugin]] = []
    for idx, plugin in enumerate(_PLUGINS):
        try:
            if not plugin.matches(question=question, payload=payload):
                continue
            score = float(plugin.match_score(question=question, payload=payload))
        except Exception:  # noqa: BLE001 — a broken matcher must not break routing
            continue
        # A plugin that gates True but scores 0 still owns the question (base
        # default scores 1.0 on match); clamp so it can never lose to a non-match.
        scored.append((max(score, 1.0), idx, plugin))

    if not scored:
        return None

    # Highest score wins; ties break on registration order (lowest index first).
    scored.sort(key=lambda t: (-t[0], t[1]))
    best_score, _best_idx, best = scored[0]

    if len(scored) > 1:
        runner_score, _runner_idx, runner = scored[1]
        if best_score - runner_score <= _AMBIGUITY_MARGIN:
            logger.warning(
                "domain routing ambiguity: %r resolved to '%s' (score=%.1f) over "
                "'%s' (score=%.1f); near-tie within margin %.1f. matched=%s",
                (question or "")[:160],
                best.domain_id,
                best_score,
                runner.domain_id,
                runner_score,
                _AMBIGUITY_MARGIN,
                [(p.domain_id, s) for s, _i, p in scored],
            )

    return best


def hypothesis_is_on_topic(
    text: str,
    *,
    question: str = "",
    domain_id: str | None = None,
    test_methodology: str | None = None,
) -> bool:
    """Return False when a domain plugin rejects hypothesis text as off-topic."""
    plugin = get_domain_plugin(domain_id) if domain_id else resolve_domain_plugin(question=question)
    if plugin is None:
        return True
    return plugin.hypothesis_on_topic(text, methodology=test_methodology)


def all_theme_rules() -> list[tuple[str, tuple[str, ...]]]:
    """Merged theme taxonomy from all registered plugins (first match wins in extract_theme_vector)."""
    _ensure_loaded()
    out: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for plugin in _PLUGINS:
        for theme_id, keywords in getattr(plugin, "theme_rules", ()) or ():
            if theme_id in seen:
                continue
            out.append((theme_id, keywords))
            seen.add(theme_id)
    return out


def all_theme_fallbacks() -> list[tuple[str, tuple[str, ...], float]]:
    _ensure_loaded()
    out: list[tuple[str, tuple[str, ...], float]] = []
    for plugin in _PLUGINS:
        out.extend(list(getattr(plugin, "theme_fallbacks", ()) or ()))
    return out
