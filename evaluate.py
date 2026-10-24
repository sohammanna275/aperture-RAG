import json
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

from main import (
    MODEL_NAME,
    EMBEDDING_MODEL,
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

def main():
    cases = json.loads(CASES_PATH.read_text(encoding="utf-8"))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_path = OUTPUT_DIR / f"{run_id}.jsonl"

    route_names = list(PIPELINES)
    records = []

    print(f"Cases: {len(cases)}")
    print(f"Pipeline calls: {len(cases) * len(PIPELINES)}")
    print(f"Saving results to: {output_path}")

    with output_path.open("x", encoding="utf-8") as output:
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

                started = time.perf_counter()

                try:
                    result = PIPELINES[route](case["question"])
                except Exception as error:
                    record["elapsed_seconds"] = time.perf_counter() - started
                    record["status"] = "error"
                    record["error"] = f"{type(error).__name__}: {error}"
                    print(f"Failed: {record['error']}")
                else:
                    record["elapsed_seconds"] = time.perf_counter() - started
                    record["status"] = "ok"
                    record["result"] = result

                    print(f"\nAnswer:\n{result['answer']}")
                    print(f"\nReference:\n{case['reference']}")

                    print("\nRetrieved evidence:")
                    if not result["retrieved_chunks"]:
                        print("None.")

                    for item in result["retrieved_chunks"]:
                        print(f"\n[Chunk {item['chunk_id']}]")
                        print(item["text"])

                    record["answer_correct"] = ask_grade()

                output.write(json.dumps(record, ensure_ascii=False) + "\n")
                output.flush()
                records.append(record)

    print_summary(records)
    print(f"\nSaved results: {output_path}")


if __name__ == "__main__":
    main()