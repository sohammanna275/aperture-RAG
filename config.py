from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
DOCUMENT_PATH = PROJECT_DIR / "data" / "company_policy.txt"

OLLAMA_URL = "http://localhost:11434/api/generate"
EMBEDDING_URL = "http://localhost:11434/api/embed"
MODEL_NAME = "qwen2.5:1.5b"
EMBEDDING_MODEL = "all-minilm:22m"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
RAG_PROMPT_VERSION = "v2"

STANDARD_TOP_K = 2
DEEP_CANDIDATE_K = 3
DEEP_FINAL_K = 2
RRF_K = 60
