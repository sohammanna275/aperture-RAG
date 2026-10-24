"""Offline regression checks; no Ollama or PostgreSQL calls are made."""

import importlib.util
import io
import json
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

from evaluation_io import execute_and_save


ROOT = Path(__file__).resolve().parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_rows(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


class EvaluationPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.case = {
            "id": "policy_05", "category": "document",
            "question": "How many users?", "reference": "3 and 20",
        }
        self.result = {
            "answer": "Starter: 3; Pro: 20", "retrieved_chunks": [],
            "input_tokens": 10, "output_tokens": 8,
        }
        fake_config = types.ModuleType("config")
        fake_config.MODEL_NAME = "fake-generator"
        fake_config.EMBEDDING_MODEL = "fake-embedding"
        fake_pipelines = types.ModuleType("pipelines")
        for route in ("direct", "standard_rag", "deep_rag"):
            setattr(fake_pipelines, f"run_{route}", Mock(return_value=self.result))
        fake_prompts = types.ModuleType("prompts")
        fake_prompts.build_prompt = lambda question, context, version: f"{version}:{question}:{context}"
        fake_client = types.ModuleType("model_client")
        fake_client.generate_answer = Mock(return_value=self.result)
        with patch.dict(sys.modules, {
            "config": fake_config, "pipelines": fake_pipelines,
            "prompts": fake_prompts, "model_client": fake_client,
        }):
            self.evaluator = load_module("tested_evaluator", "evaluate.py")
            with patch.dict(sys.modules, {"evaluate": self.evaluator}):
                self.experiment = load_module("tested_experiment", "check_prompt_revision.py")

        self.evaluator.CASES_PATH = self.root / "cases.json"
        self.evaluator.CASES_PATH.write_text(json.dumps([self.case]), encoding="utf-8")
        self.evaluator.OUTPUT_DIR = self.root / "evaluation_results"
        self.experiment.PROJECT_DIR = self.root
        self.experiment.BASELINE_PATH = self.root / "baseline.jsonl"
        targets = [
            {"case": self.case, "route": route, "status": "ok", "result": {
                "retrieved_chunks": [{"text": "policy", "chunk_id": 2}],
            }}
            for route in ("standard", "deep")
        ]
        self.experiment.BASELINE_PATH.write_text(
            "".join(json.dumps(row) + "\n" for row in targets), encoding="utf-8",
        )

    def run_quietly(self, program, args):
        with redirect_stdout(io.StringIO()):
            program.main(args)

    def paths(self, folder):
        directory = self.root / folder
        return next(directory.glob("*.jsonl")), next((directory / "raw").glob("*.jsonl"))

    def test_raw_answer_exists_before_grading_and_stays_ungraded(self):
        def grade():
            _, raw = self.paths("evaluation_results")
            self.assertEqual(read_rows(raw)[-1]["result"], self.result)
            self.assertIsNone(read_rows(raw)[-1]["answer_correct"])
            return 1

        with patch.object(self.evaluator, "ask_grade", side_effect=grade):
            self.run_quietly(self.evaluator, [])
        final, raw = self.paths("evaluation_results")
        self.assertEqual([row["answer_correct"] for row in read_rows(final)], [1, 1, 1])
        self.assertEqual([row["answer_correct"] for row in read_rows(raw)], [None] * 3)

    def test_grading_interruption_keeps_current_generated_answer(self):
        with patch.object(self.evaluator, "ask_grade", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_quietly(self.evaluator, [])
        final, raw = self.paths("evaluation_results")
        self.assertEqual(read_rows(final), [])
        self.assertEqual(len(read_rows(raw)), 1)
        self.assertEqual(read_rows(raw)[0]["status"], "ok")

    def test_no_grade_records_failures_and_continues_without_input(self):
        self.evaluator.PIPELINES["direct"] = Mock(side_effect=RuntimeError("offline failure"))
        with patch("builtins.input", side_effect=AssertionError("Unexpected input")):
            self.run_quietly(self.evaluator, ["--no-grade"])
        final, raw = self.paths("evaluation_results")
        rows = read_rows(final)
        self.assertEqual(rows, read_rows(raw))
        self.assertEqual([row["status"] for row in rows], ["error", "ok", "ok"])
        self.assertIn("offline failure", rows[0]["error"])
        self.assertTrue(all(row["elapsed_seconds"] >= 0 for row in rows))

    def test_prompt_experiment_no_grade_saves_twelve_calls_including_failure(self):
        with patch.object(
            self.experiment, "generate_answer",
            side_effect=[RuntimeError("generation failed")] + [self.result] * 11,
        ), patch("builtins.input", side_effect=AssertionError("Unexpected input")):
            self.run_quietly(self.experiment, ["--no-grade"])
        final, raw = self.paths("prompt_experiments")
        rows = read_rows(final)
        self.assertEqual(rows, read_rows(raw))
        self.assertEqual(len(rows), 12)
        self.assertEqual(sum(row["status"] == "error" for row in rows), 1)
        self.assertEqual({row["prompt_version"] for row in rows}, {"v1", "v2"})

    def test_prompt_experiment_interrupted_grade_keeps_raw_answer(self):
        with patch.object(self.experiment, "ask_grade", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                self.run_quietly(self.experiment, [])
        final, raw = self.paths("prompt_experiments")
        self.assertEqual(read_rows(final), [])
        self.assertEqual(read_rows(raw)[0]["result"], self.result)

    def test_duplicate_baseline_route_rejected_before_generation(self):
        rows = read_rows(self.experiment.BASELINE_PATH)
        rows[1]["route"] = "standard"
        self.experiment.BASELINE_PATH.write_text(
            "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8",
        )
        with self.assertRaisesRegex(ValueError, "one Standard and one Deep"):
            self.run_quietly(self.experiment, ["--no-grade"])
        self.experiment.generate_answer.assert_not_called()

    def test_save_failure_is_not_misreported_as_a_model_failure(self):
        with patch("evaluation_io.save_record", side_effect=OSError("disk full")):
            with self.assertRaisesRegex(OSError, "disk full"):
                execute_and_save({}, lambda: self.result, None)


if __name__ == "__main__":
    unittest.main()
