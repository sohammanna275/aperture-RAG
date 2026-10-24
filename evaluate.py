import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from evaluation_io import execute_and_save, save_record

from config import MODEL_NAME, EMBEDDING_MODEL
from pipelines import (
    run_direct,
    run_standard_rag,
    run_deep_rag,
)


PROJECT_DIR = Path(__file__).resolve().parent
CASES_PATH = PROJECT_DIR / "evaluation_cases.json"
OUTPUT_DIR = PROJECT_DIR / "evaluation_results"

PIPELINES = {
    "direct": run_direct,
    "standard": run_standard_rag,
    "deep": run_deep_rag,
}

def ask_grade():
    while True:
        choice = input(
            "Answer correctness [1=correct, 0=incorrect, u=unsure]: "
        ).strip().lower()

        if choice == "1":
            return 1

        if choice == "0":
            return 0

        if choice == "u":
            return None

        print("Please enter 1, 0, or u.")


def print_summary(records):
    print("\nEvaluation summary — startup costs included")

    for route in PIPELINES:
        route_records = [
            row for row in records
            if row["route"] == route
        ]

        successful = [
            row for row in route_records
            if row["status"] == "ok"
        ]

        grades = [
            row["answer_correct"] for row in successful
            if row["answer_correct"] is not None
        ]

        failures = len(route_records) - len(successful)
        ungraded = len(successful) - len(grades)

        print(f"\n{route.upper()}")
        print(f"Successful calls: {len(successful)}/{len(route_records)}")
        print(f"Failed calls: {failures}")

        if successful:
            average_time = mean(
                row["elapsed_seconds"] for row in successful
            )
            print(f"Mean successful-call time: {average_time:.3f} seconds")

        if grades:
            print(f"Correct answers among graded calls: {sum(grades)}/{len(grades)}")

        print(f"Ungraded successful calls: {ungraded}")

def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate all three pipelines.")
    parser.add_argument(
        "--no-grade", action="store_true",
        help="Save answers and metrics without asking for manual grades.",
    )
    args = parser.parse_args(argv)

    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_path = OUTPUT_DIR / f"{run_id}.jsonl"
    raw_dir = OUTPUT_DIR / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{run_id}.jsonl"

    route_names = list(PIPELINES)
    records = []

    print(f"Cases: {len(cases)}")
    print(f"Pipeline calls: {len(cases) * len(PIPELINES)}")
    print(f"Saving results to: {output_path}")
    print(f"Raw results (saved before grading): {raw_path}")

    with (
        output_path.open("x", encoding="utf-8") as output,
        raw_path.open("x", encoding="utf-8") as raw_output,
    ):
        for case_index, case in enumerate(cases):
            offset = case_index % len(route_names)
            ordered_routes = route_names[offset:] + route_names[:offset]

            for route in ordered_routes:
                print(f"\n{'=' * 60}")
                print(f"Case: {case['id']} | Route: {route}")
                print(f"Question: {case['question']}")

                record = {
                    "case": case,
                    "route": route,
                    "generation_model": MODEL_NAME,
                    "embedding_model": EMBEDDING_MODEL,
                    "first_call_for_route": case_index == 0,
                    "answer_correct": None,
                }

                succeeded = execute_and_save(
                    record,
                    lambda: PIPELINES[route](case["question"]),
                    raw_output,
                )
                if not succeeded:
                    print(f"Failed: {record['error']}")
                else:
                    result = record["result"]

                    print(f"\nAnswer:\n{result['answer']}")
                    print(f"\nReference:\n{case['reference']}")

                    print("\nRetrieved evidence:")
                    if not result["retrieved_chunks"]:
                        print("None.")

                    for item in result["retrieved_chunks"]:
                        print(f"\n[Chunk {item['chunk_id']}]")
                        print(item["text"])

                    if not args.no_grade:
                        record["answer_correct"] = ask_grade()

                save_record(output, record)
                records.append(record)

    print_summary(records)
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Completed calls remain in the raw results file printed above.")
        raise SystemExit(130)
