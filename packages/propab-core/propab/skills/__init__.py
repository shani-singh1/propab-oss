"""Propab research-skills system.

Specialized, reusable methodological guidance injected into the LLM's research
prompts so an agent reasons like a domain scientist instead of improvising from a
few lines of prompt. This module is the DOMAIN-GENERAL loader + injector; the skill
CONTENT is authored as Markdown files and split into:

  * ``skills/core/``            — domain-independent methodology (hypothesis design,
                                  adversarial test design, evidence honesty, novelty /
                                  anti-rediscovery, iteration). Applies to EVERY domain.
  * ``skills/domains/<id>/``    — domain-dependent methodology for one domain
                                  (e.g. cap-set construction techniques for
                                  math_combinatorics). Injected only for that domain.

A skill file is ``<name>.skill.md`` with a small YAML-ish frontmatter block:

    ---
    name: falsifiable-hypothesis-design
    description: Form a novel, falsifiable, scoped hypothesis that targets an open gap
    phase: hypothesis           # hypothesis | experiment | evidence | iteration | any
    scope: core                 # core | <domain_id> (must match the directory)
    priority: 10                # lower number = injected earlier (default 50)
    ---
    <markdown body: the actual methodology the LLM should follow>

Design rules (KEEP THIS DOMAIN-GENERAL):
  * The loader never hardcodes a domain — it keys on ``domain_id`` + ``phase`` only.
  * Core skills must contain NO domain vocabulary; domain flavour lives only in
    ``skills/domains/<id>/``.
  * A missing / malformed skill file is skipped, never fatal (research must not break
    because a skill file has a typo).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass, field
from typing import Any

_SKILLS_ROOT = pathlib.Path(__file__).resolve().parent
_CORE_DIR = _SKILLS_ROOT / "core"
_DOMAINS_DIR = _SKILLS_ROOT / "domains"

# Recognised research phases. "any" matches every phase.
PHASES = ("hypothesis", "experiment", "evidence", "iteration")

# Audience scoping: a skill may be intended for the orchestrator only, workers only,
# or both. Absent/unknown values default to "both" so a skill is never hidden by a
# typo (consistent with the "malformed skill file is skipped, never fatal" rule).
AUDIENCES = ("orchestrator", "worker", "both")
_DEFAULT_AUDIENCE = "both"

_DEFAULT_PRIORITY = 50


def _normalize_audience(value: str | None) -> str:
    """Coerce a front-matter ``audience`` value to a valid audience (default "both")."""
    if value is None:
        return _DEFAULT_AUDIENCE
    v = str(value).strip().lower()
    return v if v in AUDIENCES else _DEFAULT_AUDIENCE


def _audience_matches(skill_audience: str, requested: str) -> bool:
    """True when a skill scoped to ``skill_audience`` is visible to ``requested``.

    Visible when scoped to the requested audience OR to "both".
    """
    return skill_audience == requested or skill_audience == _DEFAULT_AUDIENCE


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    phase: str
    scope: str  # "core" or a domain_id
    body: str
    priority: int = _DEFAULT_PRIORITY
    audience: str = _DEFAULT_AUDIENCE  # "orchestrator" | "worker" | "both"
    source_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def applies_to_phase(self, phase: str) -> bool:
        return self.phase in ("any", "", phase)


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Split a ``---`` frontmatter block from the markdown body.

    Deliberately dependency-free (no PyYAML): frontmatter is simple ``key: value``
    lines. Returns ({} , full_text) when there is no valid frontmatter.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text
    meta: dict[str, str] = {}
    body_start = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            body_start = i + 1
            break
        raw = lines[i]
        if ":" in raw:
            key, _, val = raw.partition(":")
            meta[key.strip().lower()] = val.strip()
    if body_start is None:  # unterminated frontmatter → treat whole file as body
        return {}, text
    return meta, "\n".join(lines[body_start:]).strip()


def _load_skill_file(path: pathlib.Path, expected_scope: str) -> Skill | None:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:  # noqa: BLE001 — an unreadable skill file is skipped, never fatal
        return None
    meta, body = _parse_frontmatter(text)
    if not body.strip():
        return None
    name = meta.get("name") or path.stem.replace(".skill", "")
    phase = (meta.get("phase") or "any").lower()
    try:
        priority = int(meta.get("priority") or _DEFAULT_PRIORITY)
    except (TypeError, ValueError):
        priority = _DEFAULT_PRIORITY
    return Skill(
        name=name,
        description=meta.get("description", ""),
        phase=phase,
        scope=meta.get("scope", expected_scope) or expected_scope,
        body=body,
        priority=priority,
        audience=_normalize_audience(meta.get("audience")),
        source_path=str(path),
        extra={k: v for k, v in meta.items()
               if k not in ("name", "description", "phase", "scope", "priority", "audience")},
    )


def _load_dir(directory: pathlib.Path, expected_scope: str, phase: str) -> list[Skill]:
    if not directory.is_dir():
        return []
    out: list[Skill] = []
    for path in sorted(directory.glob("*.skill.md")):
        skill = _load_skill_file(path, expected_scope)
        if skill is not None and skill.applies_to_phase(phase):
            out.append(skill)
    return out


def load_skills(domain_id: str | None = None, phase: str = "hypothesis") -> list[Skill]:
    """Return the skills to inject for a given domain + research phase.

    Domain-general: always the ``core`` skills for the phase, plus the
    ``domains/<domain_id>`` skills for the phase when a domain is given. Ordered by
    (priority asc, scope core-first, name) so core methodology frames the domain
    specifics. Never raises — returns [] if nothing is available.
    """
    skills = _load_dir(_CORE_DIR, "core", phase)
    if domain_id:
        skills += _load_dir(_DOMAINS_DIR / domain_id, domain_id, phase)
    skills.sort(key=lambda s: (s.priority, 0 if s.scope == "core" else 1, s.name))
    return skills


def render_skills_block(skills: list[Skill]) -> str:
    """Format skills for prompt injection. Empty string when there are none."""
    if not skills:
        return ""
    parts = [
        "RESEARCH METHODOLOGY — apply these skills (general methodology first, then "
        "domain-specific technique):",
    ]
    for s in skills:
        tag = "core" if s.scope == "core" else f"domain:{s.scope}"
        header = f"\n### {s.name} [{tag}]"
        if s.description:
            header += f" — {s.description}"
        parts.append(header + "\n" + s.body.strip())
    return "\n".join(parts)


def skills_prompt_block(domain_id: str | None = None, phase: str = "hypothesis") -> str:
    """Convenience: load + render in one call for a prompt builder."""
    return render_skills_block(load_skills(domain_id, phase))


# --------------------------------------------------------------------------------
# Agentic catalog + on-demand read.
#
# Rather than auto-inject skill bodies by domain/phase (context bloat + a hardcoded
# rule), the agent is shown a compact CATALOG (names + descriptions) of everything
# available and READS only the skills it decides it needs. It infers its own domain
# and picks the core methodology + domain technique it wants — nothing is pre-selected
# from the question. ``skills_catalog``/``render_catalog`` = awareness; ``read_skills``
# = the on-demand pull.
# --------------------------------------------------------------------------------

def skills_catalog(phase: str | None = None, audience: str | None = None) -> list[Skill]:
    """Every available skill (core + ALL domains), optionally narrowed to one phase.

    Does NOT pre-filter by domain or question — the agent self-selects from the whole
    catalog so it can infer its own domain and pull cross-cutting skills. Ordered
    core-first, then by domain, priority, name.

    When ``audience`` is given, only skills visible to that audience are returned:
    those scoped to it OR to "both" (the default for a skill without an explicit
    ``audience``). ``audience=None`` (the default) applies no audience filter.
    """
    requested = None if audience is None else _normalize_audience(audience)

    def _keep(s: Skill) -> bool:
        if phase is not None and not s.applies_to_phase(phase):
            return False
        if requested is not None and not _audience_matches(s.audience, requested):
            return False
        return True

    out: list[Skill] = []
    if _CORE_DIR.is_dir():
        for p in sorted(_CORE_DIR.glob("*.skill.md")):
            s = _load_skill_file(p, "core")
            if s and _keep(s):
                out.append(s)
    if _DOMAINS_DIR.is_dir():
        for dom in sorted(d for d in _DOMAINS_DIR.iterdir() if d.is_dir()):
            for p in sorted(dom.glob("*.skill.md")):
                s = _load_skill_file(p, dom.name)
                if s and _keep(s):
                    out.append(s)
    out.sort(key=lambda s: (0 if s.scope == "core" else 1, s.scope, s.priority, s.name))
    return out


def render_catalog(
    entries: list[Skill] | None = None,
    phase: str | None = None,
    audience: str | None = None,
) -> str:
    """Compact, prompt-ready catalog — one line per skill (name, tag, phase, description).

    This is the AWARENESS layer the agent sees; full bodies are pulled on demand via
    ``read_skills``. Empty string when there are no skills. ``audience`` narrows the
    catalog the same way as ``skills_catalog`` (ignored when ``entries`` is passed in).
    """
    if entries is None:
        entries = skills_catalog(phase, audience)
    if not entries:
        return ""
    lines = [
        "AVAILABLE RESEARCH SKILLS — you decide which to read. Infer your own domain, "
        "then request (by exact name) the core methodology + domain-technique skills you "
        "need; do not request skills you will not use.",
    ]
    for s in entries:
        tag = "core" if s.scope == "core" else f"domain:{s.scope}"
        lines.append(f"- {s.name} [{tag}, phase:{s.phase}] — {s.description}")
    return "\n".join(lines)


def read_skills(names: list[str]) -> str:
    """On-demand READ: full rendered bodies of the skills the agent named (any scope).

    This is the "demand a read" step after the agent has seen the catalog. Unknown or
    malformed names are silently skipped; returns "" if nothing matches.
    """
    wanted = {str(n).strip().lower() for n in (names or []) if str(n).strip()}
    if not wanted:
        return ""
    found: list[Skill] = []
    seen: set[str] = set()
    for s in skills_catalog():
        key = s.name.lower()
        if key in wanted and key not in seen:
            found.append(s)
            seen.add(key)
    found.sort(key=lambda s: (0 if s.scope == "core" else 1, s.priority, s.name))
    return render_skills_block(found)


def catalog_skill_names(phase: str | None = None, audience: str | None = None) -> set[str]:
    """The set of valid skill names the agent may request (for parsing/validation).

    ``audience`` narrows the set the same way as ``skills_catalog``.
    """
    return {s.name for s in skills_catalog(phase, audience)}


def available_skill_index() -> dict[str, list[str]]:
    """Diagnostic: {scope -> [skill names]} across all phases (for tests / tooling)."""
    index: dict[str, list[str]] = {"core": []}
    for path in sorted(_CORE_DIR.glob("*.skill.md")) if _CORE_DIR.is_dir() else []:
        s = _load_skill_file(path, "core")
        if s:
            index["core"].append(s.name)
    if _DOMAINS_DIR.is_dir():
        for dom_dir in sorted(p for p in _DOMAINS_DIR.iterdir() if p.is_dir()):
            names = []
            for path in sorted(dom_dir.glob("*.skill.md")):
                s = _load_skill_file(path, dom_dir.name)
                if s:
                    names.append(s.name)
            if names:
                index[dom_dir.name] = names
    return index
