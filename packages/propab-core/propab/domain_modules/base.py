"""
DomainPlugin — the interface every scientific domain implements.

Core Propab calls only these methods. It never imports a domain adapter, a
dataset name, a feature name, or a threshold directly. The contract mirrors the
one described in ``fixes.md`` ("The domain plugin interface").

Design notes:
- Heavy or optional dependencies (pymatgen, dataset files, network access) must
  be imported lazily inside methods, never at module import time, so that
  merely *registering* a plugin is cheap and cannot break unrelated domains.
- ``matches`` is where domain-detection heuristics live. Core routing asks the
  plugin "is this yours?" — core never inspects question text for domain keywords
  itself.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class PreflightResult:
    """Result of a domain preflight. A campaign must not launch if ``passed`` is False."""

    passed: bool
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {"passed": self.passed, "reason": self.reason, "details": dict(self.details)}


class DomainPlugin(ABC):
    """
    A scientific domain, self-contained.

    Subclasses set the identity attributes and implement (at minimum) ``matches``
    and ``available_features``. Everything else has a safe default so a domain
    only overrides what it actually needs.
    """

    # --- Identity -----------------------------------------------------------
    domain_id: str = ""
    display_name: str = ""
    version: str = "1.0"

    # --- Scope template -----------------------------------------------------
    # Broad keywords used only to pick a default OOD-scope template when the LLM
    # omits scope fields. Deliberately looser than ``matches`` (campaign
    # detection): a bare mention of the domain is enough to fill a template.
    scope_question_markers: tuple[str, ...] = ()

    # --- Artifact-gate vocabulary ------------------------------------------
    # Keywords that signal a claim belongs to this domain for the purpose of
    # selecting domain-specific artifact models in the (domain-agnostic) artifact
    # gate. Owned here so core holds no per-domain vocabulary; core asks the
    # registry for these markers instead of hardcoding them.
    artifact_question_markers: tuple[str, ...] = ()

    # Theme taxonomy for hypothesis classification (P4.1). Each domain owns its vocabulary.
    theme_rules: tuple[tuple[str, tuple[str, ...]], ...] = ()
    theme_fallbacks: tuple[tuple[str, tuple[str, ...], float], ...] = ()

    def scope_template(self) -> dict[str, str] | None:
        """
        Default scoped-claim fields (population, distribution,
        claimed_generalization, expected_failure_modes, ood_test) for this domain,
        or None if the domain has no template. Lives here so core's scope gate
        holds no domain-specific text.

        D3 honesty note: returning ``None`` DISABLES the domain's OOD-scope
        template — core's scope gate then has nothing to fill and the scope check
        degrades to a no-op that reports "passed". That silent degradation is not
        the same as "the scope check ran and passed". Callers that need to tell
        the two apart should consult :meth:`has_scope_template` rather than
        treating a missing template as an affirmative pass.
        """
        return None

    def has_scope_template(self) -> bool:
        """
        True if this plugin supplies a non-empty OOD-scope template. When False,
        the domain's scope check is effectively DISABLED (no template to enforce);
        the pipeline should treat that as "no scope check available", not as a
        scope check that ran and passed.
        """
        try:
            tmpl = self.scope_template()
        except Exception:  # noqa: BLE001 — a broken template means no usable check
            return False
        return bool(tmpl)

    def matches_scope(self, question: str) -> bool:
        q = (question or "").lower()
        return bool(self.scope_question_markers) and any(m in q for m in self.scope_question_markers)

    # --- Detection ----------------------------------------------------------
    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        """
        Return True if this plugin owns the given campaign.

        Explicit signals (an exact ``domain``/``domain_profile`` on the payload or
        a ``[domain_profile:<id>]`` tag in the question) are handled by the
        registry before this is called; a plugin may additionally recognize its
        own domain from question content here. Content-based heuristics belong
        in the plugin — never in core routing.

        This is the *gate* (owns it or not). When two plugins both gate True on
        the same question, the registry breaks the tie by :meth:`match_score`
        rather than registration order — so a plugin should implement
        ``match_score`` when its markers can collide with another domain's.
        """
        return False

    def match_score(self, *, question: str = "", payload: dict[str, Any] | None = None) -> float:
        """
        How strongly this plugin claims the campaign, as a non-negative float.

        The registry uses this to resolve keyword collisions: when several plugins
        all :meth:`matches` the same question, the one with the highest score wins
        (registration order is only the final tie-break). Score 0 means "not
        mine".

        Default: ``1.0`` when :meth:`matches` is True, else ``0.0`` — so a plugin
        that only overrides ``matches`` keeps working (it just carries no
        collision-breaking signal beyond "I match"). A plugin whose markers can
        overlap another domain's should override this to count *how many* of its
        own specific markers appear, so the better-matching domain is preferred
        over whichever happened to be registered first.
        """
        try:
            return 1.0 if self.matches(question=question, payload=payload) else 0.0
        except Exception:  # noqa: BLE001 — a broken matcher must not break scoring
            return 0.0

    # --- Feature space ------------------------------------------------------
    @abstractmethod
    def available_features(self) -> list[str]:
        """Feature names this domain can compute and verify."""

    # --- Verification -------------------------------------------------------
    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        """
        Run domain-appropriate verification for a hypothesis and return an
        evidence dict the verdict pipeline can interpret (lofo_r2,
        label_shuffle_null_p95, verified_true_steps, verification_method, ...).

        Default: not supported (a domain that only configures the artifact gate
        need not implement this).
        """
        raise NotImplementedError(f"{self.domain_id} does not implement run_verification")

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        """
        Map a verification ``result`` to (verdict, rationale, confidence).

        Default: neutral/inconclusive. Domains with a dedicated verification path
        (e.g. materials, mandrake) override this with their own classifier. Core's
        generic ``run_verdict_pipeline`` handles verdicts for domains that do not.
        """
        return "inconclusive", "no domain verdict classifier configured", 0.5

    def hypothesis_on_topic(self, text: str, methodology: str | None = None) -> bool:
        """Return True if hypothesis text is on-topic for this domain (default: accept all)."""
        _ = methodology
        return True

    # --- Artifact detection -------------------------------------------------
    def has_artifact_models(self) -> bool:
        """
        True if this plugin can supply domain-specific artifact models (i.e. it
        has a :class:`DomainProfile`). When False, :meth:`artifact_models` returns
        an empty list and the artifact gate has no domain vocabulary to work with,
        so that gate degrades to a no-op that reports "passed". Callers should use
        this to distinguish "no artifact models available" from "the artifact gate
        ran and found nothing".
        """
        try:
            return self.domain_profile() is not None
        except Exception:  # noqa: BLE001 — a broken profile means no usable models
            return False

    def artifact_models(
        self,
        evidence: dict[str, Any] | None = None,
        hypothesis: dict[str, Any] | None = None,
    ) -> list[Any]:
        """
        Ranked plausible artifact explanations for this domain (via its profile).

        D3 honesty note: with no :meth:`domain_profile` this returns ``[]`` and the
        domain contributes no artifact vocabulary; the artifact gate then has
        nothing to check. See :meth:`has_artifact_models` to detect that a domain
        has no artifact models rather than mistaking the empty list for a passed
        artifact gate.
        """
        profile = self.domain_profile()
        if profile is None:
            return []
        from propab.artifact_verification import EvidenceContext

        ctx = evidence if isinstance(evidence, EvidenceContext) else EvidenceContext(
            hypothesis_text=str((hypothesis or {}).get("text", "")),
        )
        return profile.generate_artifact_models(ctx)

    # --- Confirmation criteria ---------------------------------------------
    def confirmation_criteria(self) -> dict[str, Any]:
        """
        Domain-appropriate confirmation thresholds. Core uses these to decide
        what "confirmed" means — it never hardcodes them.
        """
        profile = self.domain_profile()
        if profile is not None:
            return {
                "min_metric_steps_for_confirm": profile.min_metric_steps_for_confirm,
                "min_samples_per_group": profile.min_samples_per_group,
                "min_groups": profile.min_groups,
                "requires_holdout": True,
                "holdout_type": profile.evidence_method,
                "null_test": profile.permutation_null,
            }
        return {
            "min_metric_steps_for_confirm": 2,
            "requires_holdout": False,
            "holdout_type": "",
            "null_test": "",
        }

    # --- Domain preflight ---------------------------------------------------
    def has_verification_capability(self) -> bool:
        """
        Return True if this plugin can actually verify a hypothesis for its
        domain — i.e. it has a runnable verification path rather than only the
        base :meth:`run_verification` stub (which raises ``NotImplementedError``).

        A plugin has a verification capability when EITHER:
        - it overrides :meth:`run_verification` (its own worker/adapter path), OR
        - it exposes a :class:`DomainProfile` via :meth:`domain_profile` (the
          profile-backed generic verification path core can drive).

        This is what the default :meth:`preflight` consults to decide whether a
        campaign for an otherwise-unconfigured domain may launch. A "scope only"
        plugin (no ``run_verification`` override, no profile) reports False; such
        a plugin never owns campaign routing (its ``matches`` returns False), so
        core never calls its preflight for launch — but if it were ever resolved
        explicitly, failing closed is the honest answer.
        """
        overrides_run_verification = (
            type(self).run_verification is not DomainPlugin.run_verification
        )
        if overrides_run_verification:
            return True
        try:
            return self.domain_profile() is not None
        except Exception:  # noqa: BLE001 — a broken profile means no usable path
            return False

    def preflight(self) -> PreflightResult:
        """
        Check dataset access / feature computability / power before launch.

        Fail-CLOSED default: a plugin that provides no runnable verification
        capability (see :meth:`has_verification_capability`) must NOT silently
        wave a campaign through. Historically this returned ``passed=True`` for
        any un-overridden plugin, so the preflight power-gate protected only the
        domains that least needed it and launched full campaigns for genuinely
        unsupported domains that then produced nothing. It now refuses those.

        A plugin that DOES have a verification path but configures no explicit
        power check still passes here (real domains — materials, genomics, … —
        override this method with a genuine dataset/power check anyway), but the
        reason is explicit rather than a silent "no preflight configured".
        """
        if not self.has_verification_capability():
            return PreflightResult(
                passed=False,
                reason=(
                    "default preflight: no verification capability "
                    "(run_verification not overridden and no domain profile) "
                    "- refusing to launch an unsupported domain"
                ),
                details={
                    "domain": self.domain_id,
                    "has_verification_capability": False,
                },
            )
        return PreflightResult(
            passed=True,
            reason="default preflight: verification path present, no power check configured",
            details={"domain": self.domain_id, "has_verification_capability": True},
        )

    # --- Literature prior ---------------------------------------------------
    def literature_prior(self, question: str) -> dict[str, Any]:
        """Domain-specific literature context to seed a campaign. Default: none."""
        return {}

    def literature_profile(self) -> dict[str, Any]:
        """
        Domain-specific configuration for the standalone literature intelligence
        service (``services/literature/``). That service is domain-agnostic —
        all knowledge about which sources, sequences, classifications, and
        papers matter for this domain lives here, never in the service itself.

        Returns a dict with keys: ``seed_papers``, ``search_terms``,
        ``source_priorities``, ``classification_codes``, ``open_problem_sources``,
        ``tabulation_sources``, ``canonical_surveys``, ``novelty_criteria``.
        See ``agent3.md`` for the full schema. Default: empty — the literature
        service falls back to keyword search on the research question alone.
        """
        return {
            "seed_papers": [],
            "search_terms": [],
            "source_priorities": ["arxiv", "semantic_scholar"],
            "classification_codes": {},
            "open_problem_sources": [],
            "tabulation_sources": [],
            "canonical_surveys": [],
            "novelty_criteria": "",
        }

    def belief_promotion_threshold(self) -> dict[str, Any]:
        """
        Domain-appropriate rules for when a belief becomes active.
        Statistical domains require strong confidence; math trends may use weak.
        """
        return {
            "requires_supporting_nodes": 1,
            "requires_confidence": "strong",
            "allow_trend_promotion": False,
        }

    def implementable_methodologies(self) -> list[str]:
        """Keywords mapping to implemented verifier features (empty = no filter)."""
        return []

    def extract_numerical_seeds(self, confirmed_nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Extract numerical findings from confirmed nodes for next campaign seeding.

        D3 honesty note: the default returns ``[]``, which DISABLES numerical-seed
        compounding for this domain — successive campaigns get no carried-forward
        numerical findings. This is silent by design (empty is a valid result too),
        so a domain that wants seed compounding must override this.
        """
        _ = confirmed_nodes
        return []

    # --- Data provenance ----------------------------------------------------
    def uses_synthetic_data(self) -> bool:
        """
        True if this domain's dataset is SYNTHETIC (seed-generated), rather than a
        real public dataset.

        DOM2 honesty fix: three demo domains (genomics, graph_invariants,
        enzyme_kinetics) run entirely on locally seed-generated frames that present
        under real-dataset names ("GTEx v8 subset", "SNAP subset", "BRENDA
        subset"). The adapters already record ``synthetic: True`` in their cache
        meta, but that flag was never surfaced at verdict/paper time, so findings
        backed by synthetic data read as real-world results. This accessor is the
        single place the pipeline consults to decide whether a finding must be
        labelled "synthetic dataset (illustrative)".

        Default: ``False`` — real-data domains (materials, mandrake) inherit this
        and report False. A synthetic-data domain overrides this to return True.
        """
        return False

    # --- Link to the artifact-gate profile ---------------------------------
    def domain_profile(self):  # -> DomainProfile | None
        """The :class:`~propab.domain_profiles.base.DomainProfile` for this domain, if any."""
        return None
