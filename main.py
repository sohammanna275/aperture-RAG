import requests
import psycopg

from pipelines import run_direct, run_standard_rag, run_deep_rag


def main():
    print("Aperture — Adaptive RAG Router")

    route = input(
        "Choose a pipeline [direct/standard/deep]: "
    ).strip().lower()

    if route not in ("direct", "standard", "deep"):
        print("Please choose direct, standard, or deep.")
        return

    question = input("Ask a question: ").strip()

    if not question:
        print("Please enter a non-empty question.")
        return

    print(f"\nRunning the {route} pipeline...")

    try:
        if route == "direct":
            result = run_direct(question)
        elif route == "standard":
            result = run_standard_rag(question)
        else:
            result = run_deep_rag(question)
    except (
        requests.exceptions.RequestException,
        psycopg.Error,
        OSError,
        ValueError,
        RuntimeError,
    ) as error:
        print(f"Pipeline failed: {error}")
        return

    print(f"\nPipeline used: {result['pipeline']}")

    if result["ranking"]:
        print("\nCandidate ranking:")

        for rank, item in enumerate(result["ranking"], start=1):
            if result["pipeline"] == "deep":
                score_text = (
                    f"reranker: {item['reranker_score']:.4f} "
                    f"| RRF: {item['rrf_score']:.6f}"
                )
            else:
                score_text = f"cosine similarity: {item['score']:.4f}"

            print(
                f"{rank}. Chunk {item['chunk_id']} "
                f"— {score_text}"
            )

    selected_ids = [
        item["chunk_id"] for item in result["retrieved_chunks"]
    ]
    print(f"\nSelected chunk IDs: {selected_ids}")

    for item in result["retrieved_chunks"]:
        print(f"\n[Selected chunk {item['chunk_id']}]")
        print(item["text"])

    print(f"\nAperture: {result['answer']}")
    timings = result["timings"]

    print("\nTiming breakdown:")
    print(f"Question embedding: {timings['embedding_seconds']:.3f} seconds")
    print(f"Retrieval + fusion: {timings['retrieval_seconds']:.3f} seconds")
    print(
        f"Reranker preparation: "
        f"{timings['reranker_preparation_seconds']:.3f} seconds"
    )
    print(f"Reranking: {timings['reranking_seconds']:.3f} seconds")
    print(f"Generation request: {timings['generation_seconds']:.3f} seconds")
    print(f"Total pipeline: {timings['total_seconds']:.3f} seconds")

    print(f"\nGeneration input tokens: {result['input_tokens']}")
    print(f"Generation output tokens: {result['output_tokens']}")


if __name__ == "__main__":
    main()
