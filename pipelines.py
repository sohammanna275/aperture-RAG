import time

from config import (
    DEEP_CANDIDATE_K,
    DEEP_FINAL_K,
    DOCUMENT_PATH,
    EMBEDDING_MODEL,
    PROJECT_DIR,
    RAG_PROMPT_VERSION,
    RRF_K,
    STANDARD_TOP_K,
)
from database import search_chunks, search_keyword_chunks
from model_client import generate_answer, generate_embeddings
from prompts import build_prompt
from retrieval import reciprocal_rank_fusion


def run_direct(question):
    pipeline_start = time.perf_counter()

    result = generate_answer(question)

    result["pipeline"] = "direct"
    result["ranking"] = []
    result["retrieved_chunks"] = []

    result["timings"] = {
        "embedding_seconds": 0.0,
        "retrieval_seconds": 0.0,
        "reranker_preparation_seconds": 0.0,
        "reranking_seconds": 0.0,
        "generation_seconds": result["latency_seconds"],
        "total_seconds": time.perf_counter() - pipeline_start,
    }

    return result


def run_standard_rag(question):
    pipeline_start = time.perf_counter()

    embedding_start = time.perf_counter()
    question_embedding = generate_embeddings([question])[0]
    embedding_seconds = time.perf_counter() - embedding_start

    document_path = DOCUMENT_PATH.relative_to(PROJECT_DIR).as_posix()

    retrieval_start = time.perf_counter()
    selected_chunks = search_chunks(
        question_embedding=question_embedding,
        document_path=document_path,
        embedding_model=EMBEDDING_MODEL,
        top_k=STANDARD_TOP_K,
    )
    retrieval_seconds = time.perf_counter() - retrieval_start

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
    result["prompt_version"] = RAG_PROMPT_VERSION
    result["ranking"] = selected_chunks
    result["retrieved_chunks"] = selected_chunks

    result["timings"] = {
        "embedding_seconds": embedding_seconds,
        "retrieval_seconds": retrieval_seconds,
        "reranker_preparation_seconds": 0.0,
        "reranking_seconds": 0.0,
        "generation_seconds": result["latency_seconds"],
        "total_seconds": time.perf_counter() - pipeline_start,
    }

    return result


def run_deep_rag(question):
    pipeline_start = time.perf_counter()

    candidate_k = DEEP_CANDIDATE_K
    final_k = DEEP_FINAL_K

    embedding_start = time.perf_counter()
    question_embedding = generate_embeddings([question])[0]
    embedding_seconds = time.perf_counter() - embedding_start

    document_path = DOCUMENT_PATH.relative_to(PROJECT_DIR).as_posix()

    retrieval_start = time.perf_counter()

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
        rrf_k=RRF_K,
    )

    retrieval_seconds = time.perf_counter() - retrieval_start

    if not fused_results:
        raise ValueError(
            "No candidates were retrieved. "
            "Check the saved document and ingestion configuration."
        )

    preparation_start = time.perf_counter()
    from reranker import get_reranker, rerank_chunks

    get_reranker()
    preparation_seconds = time.perf_counter() - preparation_start

    rerank_start = time.perf_counter()
    reranked_results = rerank_chunks(question, fused_results)
    reranking_seconds = time.perf_counter() - rerank_start

    selected_chunks = reranked_results[:final_k]
    selected_context = "\n\n".join(
        item["text"] for item in selected_chunks
    )

    prompt = build_prompt(question, selected_context)
    result = generate_answer(prompt)

    result["pipeline"] = "deep"
    result["prompt_version"] = RAG_PROMPT_VERSION
    result["ranking"] = reranked_results
    result["retrieved_chunks"] = selected_chunks

    result["timings"] = {
        "embedding_seconds": embedding_seconds,
        "retrieval_seconds": retrieval_seconds,
        "reranker_preparation_seconds": preparation_seconds,
        "reranking_seconds": reranking_seconds,
        "generation_seconds": result["latency_seconds"],
        "total_seconds": time.perf_counter() - pipeline_start,
    }

    return result
