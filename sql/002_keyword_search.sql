BEGIN;

ALTER TABLE document_chunks
ADD COLUMN IF NOT EXISTS search_vector TSVECTOR
GENERATED ALWAYS AS (
    to_tsvector('english', content)
) STORED;

CREATE INDEX IF NOT EXISTS document_chunks_search_vector_idx
ON document_chunks
USING GIN (search_vector);

COMMIT;