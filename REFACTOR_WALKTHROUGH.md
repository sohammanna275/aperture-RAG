# Aperture: separating responsibilities before routing

This refactor changes where the code lives. It retains the current model names,
v2 prompt, retrieval limits, request payloads, result dictionaries, and timing
boundaries. It does not implement automatic routing yet.

## Step 1 — configuration (`config.py`)

Read these statements in order:

| Statement | Meaning and reason |
| --- | --- |
| `from pathlib import Path` | Import Python's filesystem-path class. |
| `PROJECT_DIR = Path(__file__).resolve().parent` | `__file__` identifies this module; `resolve()` makes its path absolute; `parent` selects the project directory. Paths therefore do not depend on which folder the terminal started in. |
| `DOCUMENT_PATH = PROJECT_DIR / "data" / "company_policy.txt"` | Join path components to locate the policy document. `/` here joins paths; it is not division. |
| `OLLAMA_URL = "http://localhost:11434/api/generate"` | Address of the local answer-generation endpoint. |
| `EMBEDDING_URL = "http://localhost:11434/api/embed"` | Address of the local embedding endpoint. |
| `MODEL_NAME = "qwen2.5:1.5b"` | Name sent with generation requests. |
| `EMBEDDING_MODEL = "all-minilm:22m"` | Name sent with embedding requests and used to filter stored dense vectors. |
| `RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"` | Model loaded by the existing local reranker. |
| `RAG_PROMPT_VERSION = "v2"` | Default prompt selected by both RAG pipelines. This preserves the current experiment setting; it does not assert that v2 is fully grounded. |
| `STANDARD_TOP_K = 2` | Standard RAG retrieves two chunks. |
| `DEEP_CANDIDATE_K = 3` | Each Deep retrieval method requests up to three candidates. Fusion can produce more than three candidates if their results differ. |
| `DEEP_FINAL_K = 2` | After reranking, Deep sends at most two chunks to generation. |
| `RRF_K = 60` | Constant in `1 / (RRF_K + rank)`. It controls rank contributions, not the number of results. |

Uppercase names are a Python convention for constants; Python does not enforce
immutability. Edit settings here before starting a new process. The existing
`from config import ...` statements bind those values when modules import.

Database connection configuration and its password loading remain in
`database.py`. The password remains in `.env`; it is not copied into this module.
Changing an embedding-model name alone is not a migration: model dimensions,
stored vectors, and the database schema must remain compatible.

## Step 2 — model calls (`model_client.py`)

This module handles the two Ollama HTTP requests. The existing local cross-encoder
stays in `reranker.py`, which now imports its model name from `config.py`.

Imports, line by line:

- `import time`: provides the monotonic performance timer.
- `import requests`: provides HTTP requests and HTTP exceptions.
- `from config import EMBEDDING_MODEL, EMBEDDING_URL, MODEL_NAME, OLLAMA_URL`:
  imports the four settings used to construct requests.

`generate_embeddings(texts)`, statement by statement:

1. `def generate_embeddings(texts):` defines a function taking a collection of input strings.
2. `payload = {` starts the request body dictionary.
3. `"model": EMBEDDING_MODEL` chooses the embedding model.
4. `"input": texts` sends the supplied strings in their original order.
5. `"truncate": False` asks Ollama to reject overlong inputs rather than silently truncate them.
6. `response = requests.post(` starts a POST request and assigns the response object.
7. `EMBEDDING_URL` is the destination.
8. `json=payload` serializes the dictionary as JSON.
9. `timeout=(5, 120)` supplies connect and read timeouts in seconds; it is not a strict total-duration deadline.
10. `response.raise_for_status()` raises an exception for an unsuccessful HTTP status.
11. `data = response.json()` decodes the response body into Python values.
12. `embeddings = data["embeddings"]` extracts the vectors. A missing key raises an error.
13. `if len(embeddings) != len(texts):` checks that every input has an output vector.
14. `raise ValueError(...)` prevents downstream code from pairing mismatched inputs and vectors.
15. `return embeddings` returns the vectors to the caller.

Closing braces and parentheses finish the dictionary and request call; they do
not perform additional operations.

`generate_answer(prompt)`, statement by statement:

1. `def generate_answer(prompt):` accepts the complete text to send to the model.
2. `payload = {` creates a request dictionary.
3. `"model": MODEL_NAME` chooses the generation model.
4. `"prompt": prompt` supplies the question or document-based prompt.
5. `"stream": False` requests one complete response instead of streamed fragments.
6. `start_time = time.perf_counter()` records the start of the generation request.
7. `response = requests.post(` starts the HTTP call.
8. `OLLAMA_URL` selects the generation endpoint.
9. `json=payload` sends the request dictionary as JSON.
10. `timeout=(5, 120)` sets the same connect and read timeouts as before.
11. `response.raise_for_status()` rejects HTTP failures.
12. `data = response.json()` decodes the body.
13. `elapsed_seconds = time.perf_counter() - start_time` measures the request and response parsing.
14. `return {` constructs the common generation result.
15. `"answer": data["response"]` stores generated text.
16. `"latency_seconds": elapsed_seconds` stores the measured duration.
17. `"input_tokens": data.get("prompt_eval_count")` stores Ollama's input count, or `None` if absent.
18. `"output_tokens": data.get("eval_count")` stores Ollama's output count, or `None` if absent.

The client does not select a route, search PostgreSQL, or build prompts. Its
callers decide what text to send. HTTP failures propagate to the CLI or evaluation
runner, which already handle them.

## Step 3 — prompts (`prompts.py`)

`from config import RAG_PROMPT_VERSION` imports the default prompt choice.

`def build_prompt_v1(question, context):` and
`def build_prompt_v2(question, context):` each accept a question and retrieved
context. Their `return f"""..."""` statements return multiline strings. The `f`
prefix substitutes `{question}` and `{context}` with their actual values.
Everything else inside those triple quotes is text sent to the model, not Python
instructions. Both templates were moved without changing their contents.

The v1 instruction lines ask the model to identify the exact entity, keep facts
attached to that entity, quote evidence, answer from that evidence, and abstain
when evidence is absent. `Document:`, `Question:`, `Evidence:`, and `Answer:` are
labels used to organize the text.

The v2 instruction lines additionally ask it to identify every requested fact,
collect separate evidence for each entity, combine facts across sections,
preserve numbers and negation, answer supported parts, and identify missing
parts. The requested evidence format is a list of entity-and-quote pairs. These
instructions are guidance to a probabilistic model, not enforced guarantees.

The dispatcher is short enough to read line by line:

```python
def build_prompt(question, context, version=None):
    if version is None:
        version = RAG_PROMPT_VERSION

    if version == "v1":
        return build_prompt_v1(question, context)

    if version == "v2":
        return build_prompt_v2(question, context)

    raise ValueError(f"Unknown prompt version: {version}")
```

- The function signature permits an explicit version for experiments.
- `if version is None` detects an omitted version.
- `version = RAG_PROMPT_VERSION` selects the configured default.
- The first version check selects v1, and its `return` ends the function.
- The second version check selects v2, and its `return` ends the function.
- The final `raise` rejects unsupported names rather than silently using another prompt.

## Step 4 — document preparation (`documents.py`)

- `from config import DOCUMENT_PATH` imports the document location.
- `def load_document():` defines the document reader.
- `text = DOCUMENT_PATH.read_text(encoding="utf-8")` reads and decodes the file.
- `return text.strip()` removes whitespace at the document's outer edges.
- `def chunk_document(text):` accepts the loaded text.
- `paragraphs = text.split("\n\n")` splits on blank-line separators.
- `chunks = []` starts the output list.
- `for paragraph in paragraphs:` visits each candidate paragraph.
- `chunk = paragraph.strip()` removes its outer whitespace.
- `if chunk:` excludes empty strings.
- `chunks.append(chunk)` keeps each nonempty paragraph.
- `return chunks` returns them in document order.

Ingestion imports these functions directly. Running an answer pipeline does not
re-ingest the document; it still searches the saved PostgreSQL rows.

## Step 5 — pipeline orchestration (`pipelines.py`)

The imports explain its responsibilities:

- `import time`: measure each stage.
- `from config import (...)`: import the model filter, path, prompt label, and retrieval limits.
- `from database import search_chunks, search_keyword_chunks`: obtain the two database searches.
- `from model_client import generate_answer, generate_embeddings`: obtain model-request functions.
- `from prompts import build_prompt`: construct document-based prompts.
- `from retrieval import reciprocal_rank_fusion`: combine dense and lexical rankings.

### Direct, in execution order

1. `def run_direct(question):` defines the simplest pipeline.
2. `pipeline_start = time.perf_counter()` starts end-to-end timing.
3. `result = generate_answer(question)` sends the question without retrieved context.
4. `result["pipeline"] = "direct"` identifies the route taken.
5. `result["ranking"] = []` indicates no ranked candidates.
6. `result["retrieved_chunks"] = []` indicates no selected context.
7. `result["timings"] = {` adds the common timing dictionary.
8. `embedding_seconds`, `retrieval_seconds`, `reranker_preparation_seconds`, and
   `reranking_seconds` each receive `0.0`, because Direct does not perform those stages.
9. `generation_seconds` receives `result["latency_seconds"]` from the model client.
10. `total_seconds` receives the elapsed time since `pipeline_start`.
11. `return result` hands the complete dictionary to the caller.

### Standard RAG, in execution order

1. `def run_standard_rag(question):` accepts a question.
2. `pipeline_start = time.perf_counter()` starts total timing.
3. `embedding_start = time.perf_counter()` starts embedding timing.
4. `question_embedding = generate_embeddings([question])[0]` passes a one-element
   list and selects its first returned vector.
5. `embedding_seconds = time.perf_counter() - embedding_start` records that stage.
6. `document_path = DOCUMENT_PATH.relative_to(PROJECT_DIR).as_posix()` produces
   `data/company_policy.txt`, exactly the document key used during ingestion.
7. `retrieval_start = time.perf_counter()` starts database retrieval timing.
8. `selected_chunks = search_chunks(` calls dense search.
9. `question_embedding=question_embedding` supplies the query vector.
10. `document_path=document_path` restricts the document.
11. `embedding_model=EMBEDDING_MODEL` restricts the stored model's vectors.
12. `top_k=STANDARD_TOP_K` requests two matches with the present configuration.
13. `retrieval_seconds = time.perf_counter() - retrieval_start` measures retrieval.
14. `if not selected_chunks:` detects an empty result.
15. Its `raise ValueError(...)` stops with ingestion guidance before generation.
16. `selected_context = "\n\n".join(` begins building the context string.
17. `item["text"] for item in selected_chunks` supplies each selected chunk's text,
    in ranked order; the join separates them with blank lines.
18. `prompt = build_prompt(question, selected_context)` applies the default template.
19. `result = generate_answer(prompt)` requests the answer.
20. `result["pipeline"] = "standard"` records the route.
21. `result["prompt_version"] = RAG_PROMPT_VERSION` records the configured prompt label.
22. `result["ranking"] = selected_chunks` records the returned dense ranking.
23. `result["retrieved_chunks"] = selected_chunks` records the context used.
24. `result["timings"] = {` collects measured embedding and retrieval durations;
    reranker preparation and reranking remain zero; generation uses the client's
    measured duration; total is the elapsed time since pipeline start.
25. `return result` returns the same result structure as before the refactor.

### Deep RAG, in execution order

1. `def run_deep_rag(question):` accepts the question.
2. `pipeline_start = time.perf_counter()` starts total timing.
3. `candidate_k = DEEP_CANDIDATE_K` reads the per-search candidate limit (three).
4. `final_k = DEEP_FINAL_K` reads the generation-context limit (two).
5. `embedding_start = time.perf_counter()` starts the embedding timer.
6. `question_embedding = generate_embeddings([question])[0]` embeds the question once.
7. `embedding_seconds = time.perf_counter() - embedding_start` records its duration.
8. `document_path = DOCUMENT_PATH.relative_to(PROJECT_DIR).as_posix()` constructs
   the same saved document key as Standard and ingestion.
9. `retrieval_start = time.perf_counter()` starts the retrieval-and-fusion timer.
10. `dense_results = search_chunks(` calls dense search with the question vector,
    document key, embedding model, and `top_k=candidate_k`.
11. `keyword_results = search_keyword_chunks(` calls lexical search with
    `question=question`, `document_path=document_path`, and `top_k=candidate_k`.
12. `fused_results = reciprocal_rank_fusion(` combines `dense_results` and
    `keyword_results`, with `rrf_k=RRF_K` (60).
13. `retrieval_seconds = time.perf_counter() - retrieval_start` includes both
    sequential searches and their fusion.
14. `if not fused_results:` detects no candidates.
15. Its `raise ValueError(...)` stops before model preparation or generation.
16. `preparation_start = time.perf_counter()` starts preparation timing.
17. `from reranker import get_reranker, rerank_chunks` imports the heavier module
    only when this execution reaches Deep's reranking stage.
18. `get_reranker()` loads or reuses the cached model.
19. `preparation_seconds = time.perf_counter() - preparation_start` includes the
    import and model preparation; initial calls may therefore cost more.
20. `rerank_start = time.perf_counter()` starts inference timing.
21. `reranked_results = rerank_chunks(question, fused_results)` scores each
    question/chunk pair and sorts candidates by the cross-encoder score.
22. `reranking_seconds = time.perf_counter() - rerank_start` records reranking time.
23. `selected_chunks = reranked_results[:final_k]` keeps the first two candidates.
24. `selected_context = "\n\n".join(` joins their `item["text"]` values in that order.
25. `prompt = build_prompt(question, selected_context)` uses the same prompt builder as Standard.
26. `result = generate_answer(prompt)` generates the answer.
27. `result["pipeline"] = "deep"` identifies the route.
28. `result["prompt_version"] = RAG_PROMPT_VERSION` identifies the prompt configuration.
29. `result["ranking"] = reranked_results` keeps the full reranked candidate list.
30. `result["retrieved_chunks"] = selected_chunks` keeps only the context used for generation.
31. `result["timings"] = {` stores the measured embedding, retrieval/fusion,
    preparation, reranking, generation, and total durations under the existing keys.
32. `return result` returns the completed pipeline result.

No separate timer was introduced by this refactor. Existing timing boundaries
were retained, although actual wall-clock durations naturally vary between runs.

## Step 6 — the command-line entry point (`main.py`)

The new imports are:

```python
import requests
import psycopg

from pipelines import run_direct, run_standard_rag, run_deep_rag
```

`requests` and `psycopg` remain so the existing error handler can recognize their
exception classes. The third statement imports the three pipeline functions.

The existing `main()` function was moved into this smaller file unchanged:

1. Print the application name.
2. Read the route; `.strip()` removes surrounding spaces and `.lower()` normalizes case.
3. Reject a route outside `direct`, `standard`, and `deep`; print guidance and `return`.
4. Read and strip the question; reject an empty question and `return`.
5. Print the selected route.
6. Enter `try`; `if`, `elif`, and `else` call the corresponding imported pipeline.
7. The `except (...) as error` block catches the same request, database, filesystem,
   validation, and runtime errors, displays the error, and returns.
8. Print `result['pipeline']`.
9. If `result['ranking']` is nonempty, loop with `enumerate(..., start=1)` to show
   one-based ranks. Deep displays `reranker_score` and `rrf_score`; Standard displays
   `score` as cosine similarity. `:.4f` and `:.6f` control displayed decimal places.
10. The list comprehension for `selected_ids` collects each selected `chunk_id`.
11. The next loop prints each selected chunk's ID and text.
12. Print `result['answer']` and bind `timings = result['timings']`.
13. Print the six timing fields, using `:.3f` for three decimal places.
14. Print generation input and output token counts.

Finally:

```python
if __name__ == "__main__":
    main()
```

When Python runs this file as a script, `__name__` equals `"__main__"`, so it
calls `main()`. Importing the module does not call the CLI.

## Step 7 — update the callers

| File | Import changes |
| --- | --- |
| `ingest.py` | Paths/model from `config`; loading/chunking from `documents`; embeddings from `model_client`. |
| `evaluate.py` | Model labels from `config`; route functions from `pipelines`. |
| `check_prompt_revision.py` | Model label from `config`; template builder from `prompts`; generation from `model_client`. |
| `check_keyword_retrieval.py` | Document and project paths from `config`. |
| `check_hybrid_retrieval.py` and `check_reranker.py` | Paths/model from `config`; embeddings from `model_client`. |
| `reranker.py` | Reranker model name from `config`. |

`from module import name` makes the named object available in the importing
file. These scripts now import each object from the module responsible for it.
No active script imports shared functionality from `main.py`.

`cosine_similarity()` and `rank_chunks()` moved intact into `retrieval.py` with
`import math`. They preserve the earlier learning implementation. Production
Standard RAG continues to use PostgreSQL cosine search. RRF also remains in
`retrieval.py` unchanged.

The persistence tests now substitute fake `config`, `pipelines`, `prompts`, and
`model_client` modules where they previously substituted `main`. Their grading
and persistence assertions remain intact. Additional offline checks cover the
pipeline boundaries and request contracts.

## Verification and next use

Run from the project directory with the virtual environment active:

```powershell
python -B -m unittest discover -s tests -v
python main.py
```

The first command runs offline tests; it does not require Ollama or PostgreSQL.
The second launches the real CLI and needs the services required by your chosen route.
Try Direct with `What is a Python list?`, then Standard and Deep with
`How many users do Nimbus Starter and Nimbus Pro support?`.

Check that each route executes, selected chunks are shown for RAG, and all timing
fields still appear. Unchanged generation behavior can still include the model
limitations observed earlier; this refactor does not tune the model.

Original source copies are saved outside the project at
`C:\Users\Soham Manna\Documents\Codex\2026-10-22\rem\aperture_module_backup`.
Historical baselines and evaluation results were not rewritten.

The future router will select one of the three functions in `pipelines.py`.
It will not need to know the CLI input format, SQL implementation, or Ollama JSON
format. This separation also allows a future LangGraph workflow to call the
existing pipeline functions. No framework dependency was added in this step.
