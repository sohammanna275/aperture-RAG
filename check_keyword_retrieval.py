from pathlib import Path

from database import search_keyword_chunks
from main import DOCUMENT_PATH


def main():
    question = input("Keyword query: ").strip()

    if not question:
        print("Please enter a non-empty query.")
        return

    project_dir = Path(__file__).resolve().parent
    document_path = DOCUMENT_PATH.relative_to(project_dir).as_posix()

    results = search_keyword_chunks(
        question=question,
        document_path=document_path,
        top_k=3,
    )

    print(f"\nKeyword matches: {len(results)}")

    for rank, item in enumerate(results, start=1):
        print(
            f"\n{rank}. Chunk {item['chunk_id']} "
            f"— keyword score: {item['score']:.4f}"
        )
        print(item["text"])


if __name__ == "__main__":
    main()