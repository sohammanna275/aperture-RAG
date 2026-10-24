from functools import lru_cache

from sentence_transformers import CrossEncoder
from torch.nn import Identity


from config import RERANKER_MODEL


@lru_cache(maxsize=1)
def get_reranker():
    return CrossEncoder(
        RERANKER_MODEL,
        device="cpu",
        max_length=512,
        activation_fn=Identity(),
    )


def rerank_chunks(question, candidates):
    if not question.strip():
        raise ValueError("The reranking question must not be empty.")

    if not candidates:
        return []

    model = get_reranker()

    pairs = [
        (question, item["text"])
        for item in candidates
    ]

    scores = model.predict(
        pairs,
        batch_size=8,
        show_progress_bar=False,
    )

    if len(scores) != len(candidates):
        raise ValueError("Expected one reranker score per candidate.")

    reranked = []

    for item, score in zip(candidates, scores):
        result = item.copy()
        result["reranker_score"] = float(score)
        reranked.append(result)

    return sorted(
        reranked,
        key=lambda item: -item["reranker_score"],
    )