"""Mandrake DomainPlugin — delegates to the (already tested) mandrake adapter."""
from __future__ import annotations

from typing import Any

from propab.domain_modules.base import DomainPlugin, PreflightResult


class MandrakePlugin(DomainPlugin):
    domain_id = "mandrake"
    display_name = "Mandrake Retroviral Wall (RT-family LOFO)"
    version = "1.0"
    scope_question_markers = (
        "rt activity",
        "retroviral",
        "biophysical",
        "evolutionary family",
        "mandrake",
    )

    def scope_template(self) -> dict[str, str]:
        return {
            "population": "56 retroviral RT sequences with handcrafted biophysical features",
            "distribution": "7 rt_family groups with ≥4 sequences each",
            "claimed_generalization": "Signal must survive leave-one-family-out across held-out families",
            "expected_failure_modes": (
                "Collapses when geometry/fold features proxy family ID; thermal-only axis"
            ),
            "ood_test": "LOFO on held-out family; label-shuffle permutation p<0.05 required before confirm",
        }

    def matches(self, *, question: str = "", payload: dict[str, Any] | None = None) -> bool:
        # Mirrors the historical is_mandrake_campaign.
        if payload and str(payload.get("domain") or "") == "mandrake":
            return True
        q = (question or "").lower()
        markers = ("rt activity", "reverse transcriptase", "evolutionary family", "biophysical propert")
        return sum(1 for m in markers if m in q) >= 2

    def available_features(self) -> list[str]:
        from propab.domain_adapters.mandrake_adapter import _KNOWN_FEATURES

        return list(_KNOWN_FEATURES)

    def run_verification(
        self,
        hypothesis: dict[str, Any],
        evidence: dict[str, Any] | None = None,
        features: list[str] | None = None,
    ) -> dict[str, Any]:
        from propab.domain_adapters.mandrake_adapter import (
            MandrakeAdapter,
            MandrakeExperimentSpec,
        )

        hyp = dict(hypothesis)
        if features:
            hyp["feature_subset"] = list(features)
        spec = MandrakeExperimentSpec.from_hypothesis(hyp, question=str(hyp.get("question", "")))
        return MandrakeAdapter().run_experiment(spec)

    def classify_verdict(self, hypothesis_text: str, result: dict[str, Any]) -> tuple[str, str, float]:
        from propab.domain_adapters.mandrake_adapter import classify_mandrake_verdict

        return classify_mandrake_verdict(hypothesis_text, result)

    def preflight(self) -> PreflightResult:
        try:
            from propab.domain_adapters.mandrake_adapter import MandrakeAdapter

            df = MandrakeAdapter().load_frame()
            n = int(len(df))
            if n < 20:
                return PreflightResult(False, f"insufficient rows: {n}", {"n_samples": n})
            return PreflightResult(True, "mandrake frame loaded", {"n_samples": n})
        except Exception as exc:  # noqa: BLE001
            return PreflightResult(False, f"mandrake dataset unavailable: {exc}")

    def domain_profile(self):
        # Mandrake uses the generic artifact gate (no dedicated profile registered);
        # the enzyme-kinetics profile is the closest family-LOFO analogue but the
        # adapter historically applies the generic gate override, so keep None.
        return None
