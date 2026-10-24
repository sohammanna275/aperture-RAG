import math


def cosine_similarity(vector_a, vector_b):
    if len(vector_a) != len(vector_b):
        raise ValueError("Vectors must have the same dimensions.")

    dot_product = sum(
        a * b for a, b in zip(vector_a, vector_b)
    )

    magnitude_a = math.sqrt(sum(a * a for a in vector_a))
    magnitude_b = math.sqrt(sum(b * b for b in vector_b))

    if magnitude_a == 0 or magnitude_b == 0:
        raise ValueError("Cosine similarity requires nonzero vectors.")

    return dot_product / (magnitude_a * magnitude_b)


def rank_chunks(question_embedding, chunks, chunk_embeddings):
    if len(chunks) != len(chunk_embeddings):
        raise ValueError("Each chunk must have one embedding.")

    ranked_chunks = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, chunk_embeddings), start=1
    ):
        score = cosine_similarity(question_embedding, embedding)

        ranked_chunks.append({
            "chunk_id": index,
            "text": chunk,
            "score": score,
        })

    return sorted(
        ranked_chunks,
        key=lambda item: item["score"],
        reverse=True,
    )


def reciprocal_rank_fusion(dense_results, keyword_results, rrf_k=60):
    if rrf_k < 1:
        raise ValueError("rrf_k must be at least 1.")

    fused = {}

    ranked_lists = {
        "dense": dense_results,
        "keyword": keyword_results,
    }

    for source, results in ranked_lists.items():
        for rank, item in enumerate(results, start=1):
            database_id = item["database_id"]

            if database_id not in fused:
                fused[database_id] = {
                    "database_id": database_id,
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "keyword_rank": None,
                }

            fused[database_id]["rrf_score"] += 1 / (rrf_k + rank)
            fused[database_id][f"{source}_rank"] = rank

    return sorted(
        fused.values(),
        key=lambda item: (-item["rrf_score"], item["database_id"]),
    )