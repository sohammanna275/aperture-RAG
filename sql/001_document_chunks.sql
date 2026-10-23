BEGIN;

CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS document_chunks (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_path TEXT NOT NULL,
    chunk_index INTEGER NOT NULL CHECK (chunk_index >= 1),
    content TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    embedding VECTOR(384) NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (document_path, chunk_index)
);

COMMIT;