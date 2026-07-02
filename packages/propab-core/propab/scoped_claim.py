"""
Scoped claims — fixes.md Step 2/3 + scope gate integrity audit.

Every hypothesis must state where it applies and where it should transfer (OOD)
before it earns verification. Declared OOD must match executed OOD (P0).
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any

SCOPE_FIELDS = (
    "population",
    "distribution",
    "claimed_generalization",
    "expected_failure_modes",
    "ood_test",
)

SCOPE_INTEGRITY_PASS = "PASS"
SCOPE_INTEGRITY_FAIL = "FAIL"
SCOPE_INTEGRITY_PENDING = "PENDING"

FAILURE_TYPES = (
    "scope_inflation",
    "single_context",
    "distribution_leakage",
    "sample_size",
    "significance_only",
    "overfitting",
    "simulator_artifact",
    "other",
)

AUDIT_VALID_SCOPE = "valid_scope"
AUDIT_BOILERPLATE_SCOPE = "boilerplate_scope"
AUDIT_MISMATCHED_SCOPE = "mismatched_scope"
AUDIT_FAKE_OOD = "fake_ood"


@dataclass
class ScopedClaim:
    population: str
    distribution: str
    claimed_generalization: str
    expected_failure_modes: str
    ood_test: str
    text: str = ""

    def to_dict(self) -> dict[str, str]:
        return {k: getattr(self, k) for k in SCOPE_FIELDS}

    def formatted_text(self) -> str:
        """Human-readable claim with explicit boundaries."""
        body = (self.text or "").strip()
        if body and all(f"{k}:" in body.lower() for k in ("population", "ood_test")):
            return body
        return (
            f"{body}\n"
            f"Population: {self.population.strip()}\n"
            f"Distribution: {self.distribution.strip()}\n"
            f"Claimed generalization: {self.claimed_generalization.strip()}\n"
            f"Expected failure modes: {self.expected_failure_modes.strip()}\n"
            f"OOD test: {self.ood_test.strip()}"
        ).strip()

    def methodology_json(self, *, base: str = "") -> str:
        payload = {"methodology": base or "scoped_verification", **self.to_dict()}
        return json.dumps(payload, ensure_ascii=False)


@dataclass
class ExecutedOOD:
    train_contexts: list[str]
    held_out_contexts: list[str]
    metric_used: str
    evaluation_code_hash: str
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ScopeIntegrityResult:
    scope_gate_result: str
    reason: str
    declared_ood: str
    executed_summary: str
    similarity: float
    audit_class: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nonempty(s: str | None) -> bool:
    return bool((s or "").strip()) and len((s or "").strip()) >= 8


def validate_scoped_claim(scope: ScopedClaim | None) -> tuple[bool, list[str]]:
    """Return (ok, missing_fields)."""
    if scope is None:
        return False, list(SCOPE_FIELDS)
    missing = [f for f in SCOPE_FIELDS if not _nonempty(getattr(scope, f, ""))]
    return len(missing) == 0, missing


def parse_scope_from_entry(entry: dict[str, Any]) -> ScopedClaim | None:
    """Parse scope from LLM hypothesis object or nested claim_scope."""
    nested = entry.get("claim_scope")
    if isinstance(nested, dict):
        return _scope_from_mapping(entry.get("text", ""), nested)
    return _scope_from_mapping(
        str(entry.get("text") or ""),
        {k: entry.get(k, "") for k in SCOPE_FIELDS},
    )


def parse_scope_from_methodology(text: str, methodology: str | None) -> ScopedClaim | None:
    tm = (methodology or "").strip()
    if tm.startswith("{"):
        try:
            obj = json.loads(tm)
            if isinstance(obj, dict) and any(obj.get(k) for k in SCOPE_FIELDS):
                return _scope_from_mapping(text, obj)
        except json.JSONDecodeError:
            pass
    return parse_scope_from_text(text)


def parse_scope_from_text(text: str) -> ScopedClaim | None:
    """Parse inline Population:/OOD test: labels from hypothesis text."""
    if not text:
        return None
    fields: dict[str, str] = {}
    patterns = {
        "population": r"(?im)^population:\s*(.+)$",
        "distribution": r"(?im)^distribution:\s*(.+)$",
        "claimed_generalization": r"(?im)^claimed generalization:\s*(.+)$",
        "expected_failure_modes": r"(?im)^expected failure modes:\s*(.+)$",
        "ood_test": r"(?im)^ood test:\s*(.+)$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, text)
        if m:
            fields[key] = m.group(1).strip()
    if len(fields) < 3:
        return None
    core = re.split(r"(?im)^population:\s*", text, maxsplit=1)[0].strip()
    return _scope_from_mapping(core, fields)


def _scope_from_mapping(text: str, mapping: dict[str, Any]) -> ScopedClaim | None:
    vals = {k: str(mapping.get(k) or "").strip() for k in SCOPE_FIELDS}
    if sum(1 for v in vals.values() if _nonempty(v)) < 3:
        return None
    return ScopedClaim(text=str(text or "").strip(), **vals)


def infer_domain_scope_template(question: str) -> ScopedClaim:
    """Default scope template when LLM omits fields (contagion / mandrake aware)."""
    ql = (question or "").lower()
    if any(k in ql for k in ("contagion", "diffusion", "spreading", "sis", "sir", "network")):
        return ScopedClaim(
            text="",
            population="N=300–5000 node graphs, 30+ instances per topology family",
            distribution="Barabási–Albert and stochastic block model ensembles; avg degree 6–12",
            claimed_generalization="Effect should transfer to Watts–Strogatz graphs with matched average degree",
            expected_failure_modes="Fails on ER graphs or when modularity Q<0.2; breaks if seed set >5% of nodes",
            ood_test="Train/evaluate on BA+SBM; hold out WS family; require LOFO R²>0 on WS or refute",
        )
    if any(k in ql for k in ("rt activity", "retroviral", "biophysical", "evolutionary family", "mandrake")):
        return ScopedClaim(
            text="",
            population="56 retroviral RT sequences with handcrafted biophysical features",
            distribution="7 rt_family groups with ≥4 sequences each",
            claimed_generalization="Signal must survive leave-one-family-out across held-out families",
            expected_failure_modes="Collapses when geometry/fold features proxy family ID; thermal-only axis",
            ood_test="LOFO on held-out family; label-shuffle permutation p<0.05 required before confirm",
        )
    return ScopedClaim(
        text="",
        population="Finite sample explicitly sized in experiment plan",
        distribution="Training distribution named (dataset, simulator, or graph family)",
        claimed_generalization="Named transfer target distinct from training distribution",
        expected_failure_modes="At least one regime where effect should weaken or vanish",
        ood_test="Hold out one family/cluster; metric must hold on OOD split before confirm",
    )


def enrich_entry_with_scope(
    entry: dict[str, Any],
    question: str,
    *,
    allow_template_fill: bool = True,
) -> dict[str, Any]:
    """Ensure entry has valid scope; fill from template only when allowed (fallbacks)."""
    scope = parse_scope_from_entry(entry)
    if scope is None and allow_template_fill:
        tmpl = infer_domain_scope_template(question)
        scope = ScopedClaim(
            text=str(entry.get("text") or ""),
            population=str(entry.get("population") or tmpl.population),
            distribution=str(entry.get("distribution") or tmpl.distribution),
            claimed_generalization=str(entry.get("claimed_generalization") or tmpl.claimed_generalization),
            expected_failure_modes=str(entry.get("expected_failure_modes") or tmpl.expected_failure_modes),
            ood_test=str(entry.get("ood_test") or tmpl.ood_test),
        )
    ok, missing = validate_scoped_claim(scope)
    out = dict(entry)
    if scope is not None:
        out["claim_scope"] = scope.to_dict()
        out["text"] = scope.formatted_text()
        base_m = str(entry.get("test_methodology") or "")
        out["test_methodology"] = scope.methodology_json(
            base=base_m[:200] if base_m and not base_m.startswith("{") else "",
        )
    out["_scope_valid"] = ok
    out["_scope_missing"] = missing
    if scope and is_boilerplate_scope(scope, question):
        out["_scope_boilerplate"] = True
    return out


def _tokenize_ood(text: str) -> set[str]:
    low = (text or "").lower()
    tokens = set(re.findall(r"[a-z0-9]+", low))
    keywords = {
        "lofo", "hold", "out", "held", "train", "evaluate", "transfer", "ws", "ba", "sbm", "er",
        "watts", "strogatz", "barabasi", "family", "topology", "label", "shuffle", "permutation",
        "mandrake", "rt", "cross", "generalization", "ood",
    }
    return {t for t in tokens if t in keywords or len(t) >= 4}


def ood_similarity(declared: str, executed_summary: str) -> float:
    """Overlap between declared OOD test and executed OOD description."""
    a = _tokenize_ood(declared)
    b = _tokenize_ood(executed_summary)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def is_boilerplate_scope(scope: ScopedClaim, question: str) -> bool:
    """True if scope is verbatim domain template (cosmetic gate bypass)."""
    tmpl = infer_domain_scope_template(question)
    pairs = zip(
        [scope.population, scope.distribution, scope.claimed_generalization,
         scope.expected_failure_modes, scope.ood_test],
        [tmpl.population, tmpl.distribution, tmpl.claimed_generalization,
         tmpl.expected_failure_modes, tmpl.ood_test],
        strict=True,
    )
    exact = sum(1 for a, b in pairs if a.strip().lower() == b.strip().lower())
    return exact >= 4


def extract_executed_ood_from_experiment(
    output: dict[str, Any],
    *,
    code: str = "",
) -> ExecutedOOD | None:
    """Build ExecutedOOD from mandrake LOFO or sandbox metadata."""
    if not output:
        return None
    code_hash = hashlib.sha256((code or output.get("executed_code") or "").encode()).hexdigest()[:16]
    families = output.get("family_breakdown") or {}
    if isinstance(families, dict) and families:
        fam_keys = [k for k in families if not str(k).endswith("_mae")]
        if fam_keys:
            held = fam_keys[-1:]
            train = [f for f in fam_keys if f not in held]
            metric = str(output.get("metric") or "lofo_r2")
            summary = f"LOFO train={train} hold_out={held} metric={output.get('mean_r2')}"
            return ExecutedOOD(
                train_contexts=[str(x) for x in train],
                held_out_contexts=[str(x) for x in held],
                metric_used=metric,
                evaluation_code_hash=code_hash,
                summary=summary,
            )
    methodology = str(output.get("methodology") or "")
    if "LOFO" in methodology.upper() or output.get("mean_r2") is not None:
        n_fam = output.get("n_families")
        summary = f"LOFO n_families={n_fam} lofo_r2={output.get('mean_r2')} label_p={output.get('label_shuffle_permutation_p')}"
        return ExecutedOOD(
            train_contexts=["in-sample families"],
            held_out_contexts=["leave-one-family-out"],
            metric_used="lofo_r2",
            evaluation_code_hash=code_hash,
            summary=summary,
        )
    ood_reason = output.get("ood_reason") or output.get("ood_summary")
    if ood_reason:
        return ExecutedOOD(
            train_contexts=[],
            held_out_contexts=[],
            metric_used=str(output.get("metric_name") or "unknown"),
            evaluation_code_hash=code_hash,
            summary=str(ood_reason),
        )
    return None


def check_scope_executed_integrity(
    scope: ScopedClaim | None,
    executed: ExecutedOOD | None,
    *,
    question: str = "",
    min_similarity: float = 0.25,
) -> ScopeIntegrityResult:
    """P0 — declared OOD must approximate executed OOD."""
    if scope is None:
        return ScopeIntegrityResult(
            scope_gate_result=SCOPE_INTEGRITY_FAIL,
            reason="missing declared scope",
            declared_ood="",
            executed_summary="",
            similarity=0.0,
            audit_class=AUDIT_MISMATCHED_SCOPE,
        )
    if is_boilerplate_scope(scope, question):
        return ScopeIntegrityResult(
            scope_gate_result=SCOPE_INTEGRITY_FAIL,
            reason="boilerplate scope matches domain template",
            declared_ood=scope.ood_test,
            executed_summary=executed.summary if executed else "",
            similarity=0.0,
            audit_class=AUDIT_BOILERPLATE_SCOPE,
        )
    if executed is None:
        return ScopeIntegrityResult(
            scope_gate_result=SCOPE_INTEGRITY_FAIL,
            reason="no executed OOD recorded",
            declared_ood=scope.ood_test,
            executed_summary="",
            similarity=0.0,
            audit_class=AUDIT_FAKE_OOD,
        )
    sim = ood_similarity(scope.ood_test, executed.summary)
    if sim < min_similarity:
        return ScopeIntegrityResult(
            scope_gate_result=SCOPE_INTEGRITY_FAIL,
            reason="declared OOD differs from executed OOD",
            declared_ood=scope.ood_test,
            executed_summary=executed.summary,
            similarity=round(sim, 4),
            audit_class=AUDIT_MISMATCHED_SCOPE,
        )
    return ScopeIntegrityResult(
        scope_gate_result=SCOPE_INTEGRITY_PASS,
        reason="declared OOD matches executed experiment",
        declared_ood=scope.ood_test,
        executed_summary=executed.summary,
        similarity=round(sim, 4),
        audit_class=AUDIT_VALID_SCOPE,
    )


def classify_manual_audit(
    scope: ScopedClaim | None,
    executed: ExecutedOOD | None,
    *,
    question: str = "",
    experiment_code: str = "",
) -> str:
    """P1 classification for manual audit samples."""
    if scope is None:
        return AUDIT_MISMATCHED_SCOPE
    if is_boilerplate_scope(scope, question):
        return AUDIT_BOILERPLATE_SCOPE
    integrity = check_scope_executed_integrity(scope, executed, question=question)
    if integrity.audit_class:
        return integrity.audit_class
    if executed is None and scope.ood_test:
        return AUDIT_FAKE_OOD
    return AUDIT_VALID_SCOPE


def passes_scope_gate(entry: dict[str, Any], *, question: str, allow_template_fill: bool = True) -> bool:
    if allow_template_fill:
        enriched = enrich_entry_with_scope(entry, question)
        return bool(enriched.get("_scope_valid"))
    scope = parse_scope_from_entry(entry)
    ok, _ = validate_scoped_claim(scope)
    return ok


def check_ood_evidence(evidence: dict[str, Any], scope: ScopedClaim | None) -> tuple[bool, str]:
    """
    Step 3 — OOD evidence required before confirmation.
    Accepts explicit ood_passed or LOFO/mandrake holdout fields.
    """
    if evidence.get("ood_passed") is True:
        return True, "OOD test passed"
    if evidence.get("ood_passed") is False:
        return False, "OOD test failed on transfer target"

    lofo = evidence.get("lofo_r2", evidence.get("mean_r2", evidence.get("metric_value")))
    label_p = evidence.get("label_shuffle_permutation_p")
    if lofo is not None and label_p is not None:
        if float(lofo) > 0.0 and float(label_p) < 0.05:
            return True, f"LOFO={float(lofo):.3f} with label-shuffle p={float(label_p):.3f}"
        return False, f"LOFO/generalization insufficient (lofo={lofo}, label_p={label_p})"

    if scope and _nonempty(scope.ood_test):
        if evidence.get("n_metric_steps", 0) == 0:
            return False, "OOD test specified but not executed"
        return False, f"OOD required ({scope.ood_test[:80]}) but ood_passed not recorded"

    return False, "No OOD evidence — claim lacks transfer validation"


def apply_ood_gate_to_verdict(
    verdict: str,
    verdict_reason: str,
    evidence: dict[str, Any],
    *,
    hypothesis_text: str = "",
    test_methodology: str | None = None,
) -> tuple[str, str]:
    if verdict != "confirmed":
        return verdict, verdict_reason
    scope = parse_scope_from_methodology(hypothesis_text, test_methodology)
    ok, reason = check_ood_evidence(evidence, scope)
    if ok:
        return verdict, f"{verdict_reason}; OOD: {reason}"
    return "inconclusive", f"OOD gate: {reason} (was: {verdict_reason})"


def _scope_field_changed(parent_val: str, child_val: str) -> bool:
    a = (parent_val or "").strip().lower()
    b = (child_val or "").strip().lower()
    if not a and not b:
        return False
    return a != b


def compute_scope_delta(
    parent: ScopedClaim | None,
    child: ScopedClaim | None,
) -> str | None:
    """P2 — human-readable delta when child scope differs from parent."""
    if parent is None or child is None:
        return None
    deltas: list[str] = []
    if _scope_field_changed(parent.population, child.population):
        deltas.append("population changed")
    if _scope_field_changed(parent.distribution, child.distribution):
        deltas.append("distribution changed")
    if _scope_field_changed(parent.claimed_generalization, child.claimed_generalization):
        deltas.append("claimed generalization changed")
    if _scope_field_changed(parent.expected_failure_modes, child.expected_failure_modes):
        deltas.append("expected failure modes changed")
    if _scope_field_changed(parent.ood_test, child.ood_test):
        deltas.append("OOD target changed")
    return "; ".join(deltas) if deltas else None


EXPANSION_VALID_INHERITANCE = "valid_inheritance"
EXPANSION_VALID_MUTATION = "valid_mutation"
EXPANSION_MISSING_SCOPE = "missing_scope"
EXPANSION_BOILERPLATE = "boilerplate_scope"


def classify_expansion_scope(
    child: ScopedClaim | None,
    parent: ScopedClaim | None,
    *,
    question: str = "",
) -> str:
    """P5 classification for tree-expanded children."""
    if child is None:
        return EXPANSION_MISSING_SCOPE
    ok, _ = validate_scoped_claim(child)
    if not ok:
        return EXPANSION_MISSING_SCOPE
    if is_boilerplate_scope(child, question):
        return EXPANSION_BOILERPLATE
    delta = compute_scope_delta(parent, child)
    if delta:
        return EXPANSION_VALID_MUTATION
    return EXPANSION_VALID_INHERITANCE


def validate_expansion_child(
    entry: dict[str, Any],
    *,
    parent: ScopedClaim | None,
    question: str,
) -> tuple[bool, dict[str, Any], str | None]:
    """
    P3 expansion gate — no template fill; require explicit scope on every child.
    Returns (accepted, enriched_entry, rejection_reason).
    """
    enriched = enrich_entry_with_scope(entry, question, allow_template_fill=False)
    scope = parse_scope_from_entry(enriched)
    ok, missing = validate_scoped_claim(scope)
    if not ok:
        return False, enriched, "missing_scope"
    if scope and is_boilerplate_scope(scope, question):
        return False, enriched, "boilerplate_scope"
    delta = compute_scope_delta(parent, scope)
    if delta:
        enriched["scope_delta"] = delta
    if scope is not None:
        enriched["claim_scope"] = scope.to_dict()
    return True, enriched, None
