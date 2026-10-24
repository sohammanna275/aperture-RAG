import time

import requests

from config import EMBEDDING_MODEL, EMBEDDING_URL, MODEL_NAME, OLLAMA_URL


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
