"""
Optional local cross-encoder reranker (ARCHITECTURE.md §5.3, Phase 2).

Install: ``pip install "propab[rerank]"`` (adds sentence-transformers + torch),
then set ``RERANKER_ENABLED=true``.
"""

from __future__ import annotations


def cross_encoder_rerank_chunk_ids(
    question: str,
    pool_rows: list[tuple[str, int, str]],
    model_name: str,
) -> list[str] | None:
    """
    Rerank chunk ids by cross-encoder(query, chunk_text) scores (higher is better).

    Returns ``None`` if the optional dependency is missing or the model fails to load.
    """
    if not pool_rows:
        return []
    try:
        from sentence_transformers import CrossEncoder
    except ImportError:
        return None

    try:
        model = CrossEncoder(model_name)
    except Exception:
        return None

    pairs: list[list[str]] = [[question[:4000], (txt or "")[:4000]] for _pid, _idx, txt in pool_rows]
    scores = model.predict(pairs, batch_size=16, show_progress_bar=False)
    order = sorted(range(len(pool_rows)), key=lambda i: float(scores[i]), reverse=True)
    return [f"{pool_rows[i][0]}|{pool_rows[i][1]}" for i in order]
