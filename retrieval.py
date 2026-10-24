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