"""Session-level domain hint from the research question (fast heuristic, no LLM)."""

from __future__ import annotations

# v1 focus: DL / ALGO / ML first; other domains for supporting questions.
_KEYWORDS: list[tuple[tuple[str, ...], str]] = [
    (
        (
            "transformer",
            "attention",
            "pytorch",
            "neural",
            "cnn",
            "mlp",
            "deep learning",
            "backprop",
            "embedding",
            "bert",
            "gpt",
            "lstm",
            "gru",
            "resnet",
            "vit ",
            "vit-",
            "fine-tun",
            "finetun",
        ),
        "deep_learning",
    ),
    (
        (
            "gradient",
            "optimizer",
            "adam",
            "sgd",
            "convergence",
            "complexity",
            "benchmark",
            "big-o",
            "big o",
            "loss landscape",
            "hessian",
            "rosenbrock",
        ),
        "algorithm_optimization",
    ),
    (
        (
            "p-value",
            "p value",
            "bootstrap",
            "confidence interval",
            "statistical",
            "significance",
            "flops",
            "ablation",
            "experiment grid",
            "variance",
        ),
        "ml_research",
    ),
    (
        ("regression", "anova", "hypothesis test", "chi-square", "t-test", "t test", "bayesian", "mcmc"),
        "statistics",
    ),
    (("matrix", "eigen", "integral", "derivative", "theorem", "proof", "topology"), "mathematics"),
    (("dataset", "csv", "histogram", "eda", "outlier", "missing value", "aggregate"), "data_analysis"),
]


def infer_session_domain(question: str) -> str:
    q = question.strip().lower()
    if not q:
        return "general_computation"
    for keys, domain in _KEYWORDS:
        for k in keys:
            if k in q:
                return domain
    return "general_computation"
