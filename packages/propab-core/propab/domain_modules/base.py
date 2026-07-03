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

    def scope_template(self) -> dict[str, str] | None:
        """
        Default scoped-claim fields (population, distribution,
        claimed_generalization, expected_failure_modes, ood_test) for this domain,
        or None if the domain has no template. Lives here so core's scope gate
        holds no domain-specific text.
        """
        return None

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
        """
        return False

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
    def artifact_models(
        self,
        evidence: dict[str, Any] | None = None,
        hypothesis: dict[str, Any] | None = None,
    ) -> list[Any]:
        """Ranked plausible artifact explanations for this domain (via its profile)."""
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
    def preflight(self) -> PreflightResult:
        """Check dataset access / feature computability / power before launch."""
        return PreflightResult(passed=True, reason="no preflight configured")

    # --- Literature prior ---------------------------------------------------
    def literature_prior(self, question: str) -> dict[str, Any]:
        """Domain-specific literature context to seed a campaign. Default: none."""
        return {}

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
        """Extract numerical findings from confirmed nodes for next campaign seeding."""
        _ = confirmed_nodes
        return []

    # --- Link to the artifact-gate profile ---------------------------------
    def domain_profile(self):  # -> DomainProfile | None
        """The :class:`~propab.domain_profiles.base.DomainProfile` for this domain, if any."""
        return None
