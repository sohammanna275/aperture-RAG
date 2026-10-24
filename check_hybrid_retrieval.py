from pathlib import Path

from database import search_chunks, search_keyword_chunks
from main import DOCUMENT_PATH, EMBEDDING_MODEL, generate_embeddings
from retrieval import reciprocal_rank_fusion


def main():
    question = input("Question: ").strip()

    if not question:
        print("Please enter a non-empty question.")
        return

    project_dir = Path(__file__).resolve().parent
    document_path = DOCUMENT_PATH.relative_to(project_dir).as_posix()

    candidate_k = 3
    final_k = 2

    question_embedding = generate_embeddings([question])[0]

    dense_results = search_chunks(
        question_embedding=question_embedding,
        document_path=document_path,
        embedding_model=EMBEDDING_MODEL,
        top_k=candidate_k,
    )

    keyword_results = search_keyword_chunks(
        question=question,
        document_path=document_path,
        top_k=candidate_k,
    )

    fused_results = reciprocal_rank_fusion(
        dense_results,
        keyword_results,
        rrf_k=60,
    )

    print("\nDense ranking:")
    for rank, item in enumerate(dense_results, start=1):
        print(
            f"{rank}. Chunk {item['chunk_id']} "
            f"— cosine similarity: {item['score']:.4f}"
        )

    print("\nKeyword ranking:")
    if not keyword_results:
        print("No keyword matches.")

    for rank, item in enumerate(keyword_results, start=1):
        print(
            f"{rank}. Chunk {item['chunk_id']} "
            f"— keyword score: {item['score']:.4f}"
        )

    print("\nFused ranking:")
    for rank, item in enumerate(fused_results, start=1):
        print(
            f"{rank}. Chunk {item['chunk_id']} "
            f"— RRF: {item['rrf_score']:.6f} "
            f"— dense rank: {item['dense_rank']} "
            f"— keyword rank: {item['keyword_rank']}"
        )

    selected_chunks = fused_results[:final_k]

    print("\nSelected context:")
    for item in selected_chunks:
        print(f"\n[Chunk {item['chunk_id']}]")
        print(item["text"])


if __name__ == "__main__":
    main()