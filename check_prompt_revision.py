import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate import ask_grade
from evaluation_io import execute_and_save, save_record
from main import MODEL_NAME, build_prompt, generate_answer


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_PATH = (
    PROJECT_DIR
    / "baselines"
    / "prompt_v1"
    / "20261024T073208_851651Z.jsonl"
)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compare prompt versions on saved context.")
    parser.add_argument(
        "--no-grade", action="store_true",
        help="Save answers and metrics without asking for manual grades.",
    )
    args = parser.parse_args(argv)

    records = [
        json.loads(line)
        for line in BASELINE_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    targets = [
        record for record in records
        if record["case"]["id"] == "policy_05"
        and record["route"] in ("standard", "deep")
        and record["status"] == "ok"
    ]

    if len(targets) != 2 or {target["route"] for target in targets} != {"standard", "deep"}:
        raise ValueError("Expected one Standard and one Deep baseline result.")

    output_dir = PROJECT_DIR / "prompt_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_path = output_dir / f"{run_id}.jsonl"
    raw_dir = output_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / f"{run_id}.jsonl"
    print(f"Saving experiment to: {output_path}")
    print(f"Raw results (saved before grading): {raw_path}")

    with (
        output_path.open("x", encoding="utf-8") as output,
        raw_path.open("x", encoding="utf-8") as raw_output,
    ):
        for target in targets:
            question = target["case"]["question"]
            context = "\n\n".join(
                item["text"]
                for item in target["result"]["retrieved_chunks"]
            )

            for repeat in range(3):
                versions = (
                    ("v1", "v2") if repeat % 2 == 0
                    else ("v2", "v1")
                )

                for version in versions:
                    print(
                        f"\nContext from: {target['route']} "
                        f"| Prompt: {version} | Repeat: {repeat + 1}"
                    )

                    prompt = build_prompt(question, context, version=version)
                    record = {
                        "baseline_file": BASELINE_PATH.name,
                        "case_id": target["case"]["id"],
                        "context_route": target["route"],
                        "question": question,
                        "context": context,
                        "prompt_version": version,
                        "prompt": prompt,
                        "generation_model": MODEL_NAME,
                        "repeat": repeat + 1,
                        "answer_correct": None,
                    }

                    succeeded = execute_and_save(
                        record,
                        lambda: generate_answer(prompt),
                        raw_output,
                    )
                    if succeeded:
                        print(f"\n{record['result']['answer']}")
                        print(f"\nReference: {target['case']['reference']}")
                        if not args.no_grade:
                            record["answer_correct"] = ask_grade()
                    else:
                        print(f"Failed: {record['error']}")

                    save_record(output, record)

    print(f"\nSaved experiment: {output_path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nStopped. Completed calls remain in the raw results file printed above.")
        raise SystemExit(130)
