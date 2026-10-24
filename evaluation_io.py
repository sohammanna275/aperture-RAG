"""Persist model results before any interactive grading takes place."""

import json
import os
import time


def save_record(output, record):
    output.write(json.dumps(record, ensure_ascii=False) + "\n")
    output.flush()
    os.fsync(output.fileno())


def execute_and_save(record, operation, raw_output):
    """Run one call, record success/failure, and save before returning."""
    started = time.perf_counter()
    try:
        result = operation()
    except Exception as error:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["status"] = "error"
        record["error"] = f"{type(error).__name__}: {error}"
    else:
        record["elapsed_seconds"] = time.perf_counter() - started
        record["status"] = "ok"
        record["result"] = result

    # Saving is outside the try block: a disk error must stop the runner.
    save_record(raw_output, record)
    return record["status"] == "ok"
