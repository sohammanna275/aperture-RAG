import os
from pathlib import Path
import psycopg
from dotenv import load_dotenv
from pgvector import Vector
from pgvector.psycopg import register_vector

ENV_PATH = Path(__file__).resolve().parent / ".env"
load_dotenv(ENV_PATH)


def get_connection():
    password = os.getenv("POSTGRES_PASSWORD")

    if not password:
        raise ValueError("POSTGRES_PASSWORD is missing or empty.")

    return psycopg.connect(
        host="127.0.0.1",
        port=5433,
        dbname="aperture",
        user="aperture",
        password=password,
        connect_timeout=5,
    )

def search_chunks(
    question_embedding,
    document_path,
    embedding_model,
    top_k=2,
):
    if len(question_embedding) != 384:
        raise ValueError("The database expects a 384-dimensional query vector.")

    if not any(question_embedding):
        raise ValueError("Cosine search requires a nonzero query vector.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    with get_connection() as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT
                    id,
                    chunk_index,
                    content,
                    embedding <=> %s AS distance
                FROM document_chunks
                WHERE document_path = %s
                  AND embedding_model = %s
                ORDER BY distance ASC, chunk_index ASC
                LIMIT %s
                """,
                (
                    Vector(question_embedding),
                    document_path,
                    embedding_model,
                    top_k,
                ),
            )
            rows = cursor.fetchall()

    results = []

    for database_id, chunk_index, content, distance in rows:
        results.append(
            {
                "database_id": database_id,
                "chunk_id": chunk_index,
                "text": content,
                "score": 1 - distance,
            }
        )

    return results

def search_keyword_chunks(question, document_path, top_k=2):
    if not question.strip():
        raise ValueError("The keyword query must not be empty.")

    if top_k < 1:
        raise ValueError("top_k must be at least 1.")

    with get_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                WITH query AS (
                    SELECT plainto_tsquery('english', %s) AS terms
                )
                SELECT
                    chunks.id,
                    chunks.chunk_index,
                    chunks.content,
                    ts_rank_cd(chunks.search_vector, query.terms) AS score
                FROM document_chunks AS chunks
                CROSS JOIN query
                WHERE chunks.document_path = %s
                  AND chunks.search_vector @@ query.terms
                ORDER BY score DESC, chunks.chunk_index ASC
                LIMIT %s
                """,
                (question, document_path, top_k),
            )
            rows = cursor.fetchall()

    results = []

    for database_id, chunk_index, content, score in rows:
        results.append(
            {
                "database_id": database_id,
                "chunk_id": chunk_index,
                "text": content,
                "score": score,
            }
        )

    return results


if __name__ == "__main__":
    with get_connection() as connection:
        row = connection.execute("SELECT current_database();").fetchone()
        print(f"Connected to database: {row[0]}")