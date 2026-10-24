"""Check pipeline wiring offline; never call Ollama, PostgreSQL, or a real reranker."""

import importlib
import sys
import types
import unittest
from unittest.mock import Mock, patch

import config
import model_client
import pipelines
from documents import chunk_document
from prompts import build_prompt


class PipelineBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.question = "How many users?"
        self.chunks = [
            {"database_id": 2, "chunk_id": 2, "text": "Starter: 3 users", "score": 0.8},
            {"database_id": 3, "chunk_id": 3, "text": "Pro: 20 users", "score": 0.7},
        ]
        self.embedding = [0.1] * 384
        self.answer = {
            "answer": "Starter: 3; Pro: 20", "latency_seconds": 0.25,
            "input_tokens": 12, "output_tokens": 8,
        }
        for name, value in (
            ("generate_embeddings", [self.embedding]),
            ("search_chunks", self.chunks),
            ("search_keyword_chunks", self.chunks),
            ("generate_answer", self.answer),
        ):
            patcher = patch.object(pipelines, name, return_value=value)
            setattr(self, name, patcher.start())
            self.addCleanup(patcher.stop)

    def assert_timings(self, result):
        self.assertEqual(set(result["timings"]), {
            "embedding_seconds", "retrieval_seconds", "reranker_preparation_seconds",
            "reranking_seconds", "generation_seconds", "total_seconds",
        })
        self.assertEqual(result["timings"]["generation_seconds"], 0.25)
        self.assertTrue(all(value >= 0 for value in result["timings"].values()))

    def test_direct_passes_question_without_retrieval(self):
        result = pipelines.run_direct(self.question)
        self.generate_answer.assert_called_once_with(self.question)
        self.generate_embeddings.assert_not_called()
        self.search_chunks.assert_not_called()
        self.search_keyword_chunks.assert_not_called()
        self.assertEqual(result["pipeline"], "direct")
        self.assertEqual(result["retrieved_chunks"], [])
        self.assert_timings(result)

    def test_standard_passes_saved_context_to_generator(self):
        result = pipelines.run_standard_rag(self.question)
        self.generate_embeddings.assert_called_once_with([self.question])
        self.search_chunks.assert_called_once_with(
            question_embedding=self.embedding, document_path="data/company_policy.txt",
            embedding_model=config.EMBEDDING_MODEL, top_k=2,
        )
        self.search_keyword_chunks.assert_not_called()
        self.generate_answer.assert_called_once_with(
            build_prompt(self.question, "Starter: 3 users\n\nPro: 20 users")
        )
        self.assertEqual(result["retrieved_chunks"], self.chunks)
        self.assertEqual(result["prompt_version"], config.RAG_PROMPT_VERSION)
        self.assert_timings(result)

    def test_deep_generates_from_reranked_order_and_keeps_timing_fields(self):
        fake = types.ModuleType("reranker")
        fake.get_reranker = Mock()
        fake.rerank_chunks = Mock(side_effect=lambda question, chunks: [
            dict(item, reranker_score=1.0) for item in reversed(chunks)
        ])
        with patch.dict(sys.modules, {"reranker": fake}):
            result = pipelines.run_deep_rag(self.question)
        fake.get_reranker.assert_called_once_with()
        self.assertEqual(self.search_chunks.call_args.kwargs["top_k"], 3)
        self.search_keyword_chunks.assert_called_once_with(
            question=self.question, document_path="data/company_policy.txt", top_k=3,
        )
        self.assertEqual([item["chunk_id"] for item in result["retrieved_chunks"]], [3, 2])
        self.generate_answer.assert_called_once_with(
            build_prompt(self.question, "Pro: 20 users\n\nStarter: 3 users")
        )
        self.assertAlmostEqual(result["ranking"][0]["rrf_score"], 2 / 62)
        self.assert_timings(result)

    def test_empty_retrieval_stops_before_generation(self):
        self.search_chunks.return_value = []
        self.search_keyword_chunks.return_value = []
        for run in (pipelines.run_standard_rag, pipelines.run_deep_rag):
            with self.subTest(pipeline=run.__name__), self.assertRaises(ValueError):
                run(self.question)
        self.generate_answer.assert_not_called()

    def test_importing_entry_point_does_not_run_cli_or_load_reranker(self):
        with patch("builtins.input", side_effect=AssertionError("Import prompted for input")):
            with patch.dict(sys.modules, {"reranker": None}):
                importlib.reload(pipelines)
                importlib.import_module("main")


class ModelClientTests(unittest.TestCase):
    def test_generation_payload_and_result_contract(self):
        response = Mock()
        response.json.return_value = {
            "response": "answer", "prompt_eval_count": 5, "eval_count": 2,
        }
        with patch.object(model_client.requests, "post", return_value=response) as post:
            result = model_client.generate_answer("question")
        post.assert_called_once_with(
            config.OLLAMA_URL,
            json={"model": config.MODEL_NAME, "prompt": "question", "stream": False},
            timeout=(5, 120),
        )
        response.raise_for_status.assert_called_once_with()
        self.assertEqual((result["answer"], result["input_tokens"], result["output_tokens"]),
                         ("answer", 5, 2))

    def test_embedding_count_mismatch_is_rejected(self):
        response = Mock()
        response.json.return_value = {"embeddings": [[1.0]]}
        with patch.object(model_client.requests, "post", return_value=response) as post:
            with self.assertRaisesRegex(ValueError, "one embedding"):
                model_client.generate_embeddings(["one", "two"])
        post.assert_called_once_with(
            config.EMBEDDING_URL,
            json={"model": config.EMBEDDING_MODEL, "input": ["one", "two"], "truncate": False},
            timeout=(5, 120),
        )

    def test_http_failure_propagates(self):
        response = Mock()
        response.raise_for_status.side_effect = model_client.requests.HTTPError("failed")
        with patch.object(model_client.requests, "post", return_value=response):
            with self.assertRaises(model_client.requests.HTTPError):
                model_client.generate_answer("question")

    def test_chunk_boundaries_and_unknown_prompt_version(self):
        self.assertEqual(chunk_document("  first\n\n\n\n second  "), ["first", "second"])
        with self.assertRaisesRegex(ValueError, "Unknown prompt version"):
            build_prompt("question", "context", version="missing")


if __name__ == "__main__":
    unittest.main()
