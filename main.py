import time
import math
from pathlib import Path
import requests
import psycopg
from database import search_chunks

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:1.5b"
DOCUMENT_PATH = Path(__file__).resolve().parent / "data" / "company_policy.txt"
EMBEDDING_URL = "http://localhost:11434/api/embed"
EMBEDDING_MODEL = "all-minilm:22m"
def load_document():
    text = DOCUMENT_PATH.read_text(encoding="utf-8")
    return text.strip()

def chunk_document(text):
    paragraphs = text.split("\n\n")
    chunks = []

    for paragraph in paragraphs:
        chunk = paragraph.strip()

        if chunk:
            chunks.append(chunk)

    return chunks

def generate_embeddings(texts):
    payload = {
        "model": EMBEDDING_MODEL,
        "input": texts,
        "truncate": False,
    }

    response = requests.post(
        EMBEDDING_URL,
        json=payload,
        timeout=(5, 120),
    )
    response.raise_for_status()

    data = response.json()
    embeddings = data["embeddings"]

    if len(embeddings) != len(texts):
        raise ValueError("Expected one embedding for each input text.")

    return embeddings

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


def build_prompt(question, context):
    return f"""Use only the document to answer the question.

Instructions:
- Find evidence about the exact plan or entity named in the question.
- Do not transfer facts from one plan or entity to another.
- First quote the relevant sentence from the document.
- Then answer the question using that evidence.
- If the document does not provide the answer, write:
  Evidence: Not found.
  Answer: I don't know based on the provided document.

Document:
{context}

Question:
{question}

Respond in this format:
Evidence: <exact quote from the document, or Not found>
Answer: <answer supported by the evidence>
"""


def generate_answer(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    start_time = time.perf_counter()

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=(5, 120),
    )
    response.raise_for_status()

    data = response.json()
    elapsed_seconds = time.perf_counter() - start_time

    return {
        "answer": data["response"],
        "latency_seconds": elapsed_seconds,
        "input_tokens": data.get("prompt_eval_count"),
        "output_tokens": data.get("eval_count"),
    }

def run_direct(question):
    result = generate_answer(question)

    result["pipeline"] = "direct"
    result["ranking"] = []
    result["retrieved_chunks"] = []

    return result


def run_standard_rag(question):
    question_embedding = generate_embeddings([question])[0]

    project_dir = Path(__file__).resolve().parent
    document_path = DOCUMENT_PATH.relative_to(project_dir).as_posix()

    selected_chunks = search_chunks(
        question_embedding=question_embedding,
        document_path=document_path,
        embedding_model=EMBEDDING_MODEL,
        top_k=2,
    )

    if not selected_chunks:
        raise ValueError(
            "No saved chunks match this document and embedding model. "
            "Run python ingest.py first."
        )

    selected_context = "\n\n".join(
        item["text"] for item in selected_chunks
    )

    prompt = build_prompt(question, selected_context)
    result = generate_answer(prompt)

    result["pipeline"] = "standard"
    result["ranking"] = selected_chunks
    result["retrieved_chunks"] = selected_chunks

    return result

def main():
    print("Aperture — Adaptive RAG Router")

    route = input("Choose a pipeline [direct/standard]: ").strip().lower()

    if route not in ("direct", "standard"):
        print("Please choose direct or standard.")
        return

    question = input("Ask a question: ").strip()

    if not question:
        print("Please enter a non-empty question.")
        return

    print(f"\nRunning the {route} pipeline...")

    try:
        if route == "direct":
            result = run_direct(question)
        else:
            result = run_standard_rag(question)
    except (
        requests.exceptions.RequestException,
        psycopg.Error,
        OSError,
        ValueError,
    ) as error:
        print(f"Pipeline failed: {error}")
        return

    print(f"\nPipeline used: {result['pipeline']}")

    if result["ranking"]:
        print("\nRetrieved results (top matches):")

        for rank, item in enumerate(result["ranking"], start=1):
            print(
                f"{rank}. Chunk {item['chunk_id']} "
                f"— similarity: {item['score']:.4f}"
            )

    selected_ids = [
        item["chunk_id"] for item in result["retrieved_chunks"]
    ]
    print(f"\nSelected chunk IDs: {selected_ids}")

    for item in result["retrieved_chunks"]:
        print(f"\n[Selected chunk {item['chunk_id']}]")
        print(item["text"])

    print(f"\nAperture: {result['answer']}")
    print(f"\nGeneration request time: {result['latency_seconds']:.2f} seconds")
    print(f"Generation input tokens: {result['input_tokens']}")
    print(f"Generation output tokens: {result['output_tokens']}")

if __name__ == "__main__":
    main()