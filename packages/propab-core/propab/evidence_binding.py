"""
Evidence Binding — mechanical citation checks (fixes.md Phase 1).

Any field claiming a relationship to specific evidence must pass a deterministic
subject-overlap check before write. Empty honest fields beat populated false ones.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from propab.scoped_claim import parse_scope_from_text

# ── Subject tags (deterministic, not LLM) ─────────────────────────────────────

TAG_CROSS_FAMILY_LOFO = "cross_family_lofo"
TAG_WITHIN_FAMILY = "within_family"
TAG_SPLIT_SENSITIVITY = "split_sensitivity"
TAG_FEATURE_SUFFICIENCY = "feature_sufficiency"
TAG_SEQUENCE_REDUNDANCY = "sequence_redundancy"
TAG_CONFOUND_BATCH = "confound_batch"
TAG_NULL_MODEL = "null_model"
TAG_ASSAY_NOISE = "assay_noise"
TAG_FAMILY_CATEGORICAL = "family_categorical"

_SCOPE_CROSS = "cross_family"
_SCOPE_WITHIN = "within_family"
_SCOPE_MIXED = "mixed"
_SCOPE_UNKNOWN = "unknown"

_INCOMPATIBLE_TAG_PAIRS: frozenset[frozenset[str]] = frozenset({
    frozenset({TAG_FEATURE_SUFFICIENCY, TAG_CROSS_FAMILY_LOFO}),
    frozenset({TAG_SEQUENCE_REDUNDANCY, TAG_CROSS_FAMILY_LOFO}),
    frozenset({TAG_WITHIN_FAMILY, TAG_CROSS_FAMILY_LOFO}),
    frozenset({TAG_SPLIT_SENSITIVITY, TAG_CROSS_FAMILY_LOFO}),
    frozenset({TAG_FEATURE_SUFFICIENCY, TAG_FAMILY_CATEGORICAL}),
})

_LOFO_RE = re.compile(r"lofo|leave[- ]one[- ](?:family|group)|cross[- ]family", re.I)
_WITHIN_RE = re.compile(r"within[- ]family|intra[- ]family|single family|per family", re.I)
_SPLIT_RE = re.compile(r"clustered split|random split|split[- ]sensitivity", re.I)
_FEAT_SUFF_RE = re.compile(
    r"insufficient|inadequate|cannot capture|fundamentally insufficient|feature set is|98.feature",
    re.I,
)
_REDUND_RE = re.compile(r"redundan|nearest[- ]neighbor|sequence identity|<\s*50\s*%", re.I)
_CONFOUND_RE = re.compile(r"batch|plate id|confound|nuisance variable", re.I)
_NULL_RE = re.compile(r"null hypothesis|label[- ]shuffle|permutation null", re.I)
_ASSAY_RE = re.compile(r"assay resolution|measurement noise|stochastic noise", re.I)
_FAMILY_CAT_RE = re.compile(r"categorical property of the evolutionary family|family identity explains", re.I)

_FEATURE_TOKEN_RE = re.compile(r"\b(t\d+_raw|triad_best_rmsd|sp_motif_found|g_factor_overall|D2_D3_dist)\b", re.I)


@dataclass(frozen=True)
class ClaimSubject:
    """Canonical subject extracted from structured fields + deterministic text tags."""

    claim_type: str | None
    population_scope: str
    test_targets: frozenset[str]
    feature_variables: frozenset[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_type": self.claim_type,
            "population_scope": self.population_scope,
            "test_targets": sorted(self.test_targets),
            "feature_variables": sorted(self.feature_variables),
        }


@dataclass
class BindingResult:
    match: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"match": self.match, "reason": self.reason}


@dataclass
class BindingMetrics:
    binding_rejected_count: int = 0
    binding_accepted_count: int = 0
    falsifiability_rejected_count: int = 0
    belief_cap_rejected_count: int = 0
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "binding_rejected_count": self.binding_rejected_count,
            "binding_accepted_count": self.binding_accepted_count,
            "falsifiability_rejected_count": self.falsifiability_rejected_count,
            "belief_cap_rejected_count": self.belief_cap_rejected_count,
            "rejection_reasons": list(self.rejection_reasons[-20:]),
        }


def infer_test_targets(text: str) -> frozenset[str]:
    t = text or ""
    tags: set[str] = set()
    if _LOFO_RE.search(t):
        tags.add(TAG_CROSS_FAMILY_LOFO)
    if _WITHIN_RE.search(t):
        tags.add(TAG_WITHIN_FAMILY)
    if _SPLIT_RE.search(t):
        tags.add(TAG_SPLIT_SENSITIVITY)
    if _FEAT_SUFF_RE.search(t):
        tags.add(TAG_FEATURE_SUFFICIENCY)
    if _REDUND_RE.search(t):
        tags.add(TAG_SEQUENCE_REDUNDANCY)
    if _CONFOUND_RE.search(t):
        tags.add(TAG_CONFOUND_BATCH)
    if _NULL_RE.search(t):
        tags.add(TAG_NULL_MODEL)
    if _ASSAY_RE.search(t):
        tags.add(TAG_ASSAY_NOISE)
    if _FAMILY_CAT_RE.search(t):
        tags.add(TAG_FAMILY_CATEGORICAL)
    return frozenset(tags)


def infer_population_scope(text: str, claim_scope: dict[str, str] | None = None) -> str:
    if claim_scope:
        blob = " ".join(str(claim_scope.get(k) or "") for k in (
            "population", "distribution", "claimed_generalization", "ood_test",
        )).lower()
        if _WITHIN_RE.search(blob) and not _LOFO_RE.search(blob):
            return _SCOPE_WITHIN
        if _LOFO_RE.search(blob) or "cross" in blob:
            return _SCOPE_CROSS
        if blob.strip():
            return _SCOPE_MIXED
    t = (text or "").lower()
    has_within = bool(_WITHIN_RE.search(t))
    has_cross = bool(_LOFO_RE.search(t))
    if has_within and not has_cross:
        return _SCOPE_WITHIN
    if has_cross and not has_within:
        return _SCOPE_CROSS
    if has_within and has_cross:
        return _SCOPE_MIXED
    return _SCOPE_UNKNOWN


def extract_feature_variables(text: str, feature_subset: list[str] | None = None) -> frozenset[str]:
    feats = {f.lower() for f in (feature_subset or []) if f}
    for m in _FEATURE_TOKEN_RE.finditer(text or ""):
        feats.add(m.group(1).lower())
    return frozenset(feats)


def extract_subject_from_node(node: dict[str, Any]) -> ClaimSubject:
    text = str(node.get("text") or "")
    scope = node.get("claim_scope")
    if isinstance(scope, dict):
        scope_dict = {k: str(scope.get(k) or "") for k in scope}
    else:
        parsed = parse_scope_from_text(text)
        scope_dict = parsed.to_dict() if parsed else None  # type: ignore[union-attr]

    return ClaimSubject(
        claim_type=node.get("claim_type"),
        population_scope=infer_population_scope(text, scope_dict),
        test_targets=infer_test_targets(text),
        feature_variables=extract_feature_variables(
            text, list(node.get("feature_subset") or []),
        ),
    )


def extract_subject_from_statement(statement: str) -> ClaimSubject:
    return ClaimSubject(
        claim_type=None,
        population_scope=infer_population_scope(statement),
        test_targets=infer_test_targets(statement),
        feature_variables=extract_feature_variables(statement),
    )


def extract_subject_from_mechanism(mechanism: dict[str, Any]) -> ClaimSubject:
    text = " ".join([
        str(mechanism.get("explanation") or ""),
        " ".join(str(x) for x in (mechanism.get("candidate_features") or [])),
        " ".join(str(x) for x in (mechanism.get("assumptions_challenged") or [])),
    ])
    return ClaimSubject(
        claim_type=None,
        population_scope=infer_population_scope(text),
        test_targets=infer_test_targets(text),
        feature_variables=extract_feature_variables(
            text, list(mechanism.get("candidate_features") or []),
        ),
    )


def extract_subject_from_anomaly(anomaly: dict[str, Any]) -> ClaimSubject:
    meta = anomaly.get("metadata") if isinstance(anomaly.get("metadata"), dict) else {}
    bucket = str(meta.get("bucket") or "")
    text = " ".join([
        bucket,
        str(anomaly.get("anomaly_type") or ""),
        " ".join(str(f) for f in (anomaly.get("feature_subset") or [])),
        f"lofo={anomaly.get('observed_score')}",
        f"within_r2={meta.get('within_family_r2')}",
    ])
    tags = infer_test_targets(text)
    if bucket in {"collapse"}:
        tags = tags | {TAG_WITHIN_FAMILY, TAG_CROSS_FAMILY_LOFO}
    elif bucket in {"survivor", "cross_family"}:
        tags = tags | {TAG_CROSS_FAMILY_LOFO}
    return ClaimSubject(
        claim_type=None,
        population_scope=_SCOPE_CROSS if TAG_CROSS_FAMILY_LOFO in tags else _SCOPE_UNKNOWN,
        test_targets=tags,
        feature_variables=extract_feature_variables(
            text, list(anomaly.get("feature_subset") or []),
        ),
    )


def _tags_incompatible(a: frozenset[str], b: frozenset[str]) -> str | None:
    if not a or not b:
        return None
    for pair in _INCOMPATIBLE_TAG_PAIRS:
        if pair <= a and pair <= b:
            return f"incompatible_tags:{sorted(pair)}"
    # citing within-family / split / redundancy must not cite pure cross-family LOFO only
    citing_needs = a & {TAG_WITHIN_FAMILY, TAG_SPLIT_SENSITIVITY, TAG_SEQUENCE_REDUNDANCY, TAG_FEATURE_SUFFICIENCY}
    if citing_needs and b == {TAG_CROSS_FAMILY_LOFO}:
        return "citing_within_family_claim_vs_cross_family_lofo_only"
    if TAG_FEATURE_SUFFICIENCY in a and TAG_CROSS_FAMILY_LOFO in b and not (a & b):
        return "feature_sufficiency_vs_lofo_only"
    return None


def binding_check(citing: ClaimSubject, cited: ClaimSubject) -> BindingResult:
    """Return match if cited evidence plausibly bears on citing claim's subject."""
    shared_tags = citing.test_targets & cited.test_targets
    if shared_tags:
        return BindingResult(True, f"shared_test_targets:{sorted(shared_tags)}")

    incompat = _tags_incompatible(citing.test_targets, cited.test_targets)
    if incompat:
        return BindingResult(False, incompat)

    # Population scope alignment with some tag overlap on cited side
    if (
        citing.population_scope == cited.population_scope
        and citing.population_scope != _SCOPE_UNKNOWN
        and cited.test_targets
    ):
        return BindingResult(True, f"shared_population_scope:{citing.population_scope}")

    # Feature-variable overlap only when citing names specific features
    if citing.feature_variables and cited.feature_variables:
        shared_feats = citing.feature_variables & cited.feature_variables
        if shared_feats and (citing.test_targets & cited.test_targets or citing.population_scope == cited.population_scope):
            return BindingResult(True, f"shared_features:{sorted(shared_feats)}")

    if not cited.test_targets and not citing.test_targets:
        return BindingResult(False, "both_subjects_untyped")

    if citing.test_targets and not cited.test_targets:
        return BindingResult(False, "cited_node_untyped_for_citing_claim")

    return BindingResult(False, "no_subject_overlap")


def binding_check_statement_to_node(
    citing_statement: str,
    cited_node: dict[str, Any],
) -> BindingResult:
    return binding_check(
        extract_subject_from_statement(citing_statement),
        extract_subject_from_node(cited_node),
    )


def filter_node_citations(
    citing_statement: str,
    node_ids: list[str],
    nodes: dict[str, Any],
    *,
    metrics: BindingMetrics | None = None,
) -> list[str]:
    """Keep only node IDs that mechanically match the citing claim."""
    accepted: list[str] = []
    citing = extract_subject_from_statement(citing_statement)
    for nid in node_ids:
        node = nodes.get(nid)
        if not isinstance(node, dict):
            if metrics:
                metrics.binding_rejected_count += 1
                metrics.rejection_reasons.append(f"missing_node:{nid}")
            continue
        result = binding_check(citing, extract_subject_from_node(node))
        if result.match:
            accepted.append(nid)
            if metrics:
                metrics.binding_accepted_count += 1
        elif metrics:
            metrics.binding_rejected_count += 1
            metrics.rejection_reasons.append(f"{nid}:{result.reason}")
    return accepted


def filter_mechanism_anomalies(
    mechanism: dict[str, Any],
    anomalies_by_key: dict[str, dict[str, Any]],
    *,
    metrics: BindingMetrics | None = None,
) -> list[str]:
    citing = extract_subject_from_mechanism(mechanism)
    accepted: list[str] = []
    cand_feats = {str(f).lower() for f in (mechanism.get("candidate_features") or [])}
    expl = str(mechanism.get("explanation") or "").lower()

    for key in mechanism.get("supporting_anomalies") or []:
        key_s = str(key)
        anomaly = anomalies_by_key.get(key_s)
        if anomaly is None:
            # key is feature join — match if features appear in candidate_features or explanation
            parts = {p.lower() for p in key_s.split("|") if p}
            if parts and (parts <= cand_feats or all(p in expl for p in parts)):
                accepted.append(key_s)
                if metrics:
                    metrics.binding_accepted_count += 1
                continue
            if metrics:
                metrics.binding_rejected_count += 1
                metrics.rejection_reasons.append(f"anomaly_missing:{key_s}")
            continue

        cited = extract_subject_from_anomaly(anomaly)
        result = binding_check(citing, cited)
        feat_parts = {f.lower() for f in (anomaly.get("feature_subset") or [])}
        feat_ok = not feat_parts or feat_parts <= cand_feats or feat_parts.issubset(
            extract_feature_variables(expl).union(cand_feats),
        )
        if result.match and feat_ok:
            accepted.append(key_s)
            if metrics:
                metrics.binding_accepted_count += 1
        else:
            if metrics:
                metrics.binding_rejected_count += 1
                reason = result.reason if not feat_ok else result.reason
                if not feat_ok:
                    reason = f"feature_mismatch:{sorted(feat_parts)}"
                metrics.rejection_reasons.append(f"{key_s}:{reason}")
    return accepted


def belief_falsifiable_in_dataset(
    statement: str,
    *,
    feature_count: int = 0,
    n_samples: int = 0,
) -> tuple[bool, str]:
    """
    Reject beliefs that cannot be killed by any experiment in a fixed dataset.

    Domain-agnostic: a belief that "the feature set is fundamentally insufficient"
    cannot be falsified within a fixed-feature dataset regardless of the domain.
    ``feature_count``/``n_samples`` are optional domain-provided shape hints
    (reserved for future power checks); no dataset-specific numbers live here.
    """
    t = (statement or "").lower()
    if re.search(r"insufficient|inadequate|fundamentally insufficient|cannot capture", t):
        if re.search(r"feature set|feature space|current feature|\d+[\s-]?feature", t):
            return False, "feature_insufficiency_unfalsifiable_in_fixed_feature_dataset"
    if re.search(r"more research is needed|requires larger dataset|cannot be determined", t):
        return False, "vague_escape_hatch"
    if feature_count > 0 and n_samples > 0:
        pass  # reserved for future shape checks
    return True, ""


RIVAL_MAX_ACTIVE_BELIEFS = 2
