import json
from datetime import datetime, timezone
from pathlib import Path

from evaluate import ask_grade
from main import MODEL_NAME, build_prompt, generate_answer


PROJECT_DIR = Path(__file__).resolve().parent
BASELINE_PATH = (
    PROJECT_DIR
    / "baselines"
    / "prompt_v1"
    / "20261024T073208_851651Z.jsonl"
)


def main():
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

    if len(targets) != 2:
        raise ValueError("Expected one Standard and one Deep baseline result.")

    output_dir = PROJECT_DIR / "prompt_experiments"
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output_path = output_dir / f"{run_id}.jsonl"

    with output_path.open("x", encoding="utf-8") as output:
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
                    result = generate_answer(prompt)

                    print(f"\n{result['answer']}")
                    print(f"\nReference: {target['case']['reference']}")

                    grade = ask_grade()

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
                        "answer_correct": grade,
                        "result": result,
                    }

                    output.write(
                        json.dumps(record, ensure_ascii=False) + "\n"
                    )
                    output.flush()

    print(f"\nSaved experiment: {output_path}")


if __name__ == "__main__":
    main()