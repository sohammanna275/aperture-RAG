from pathlib import Path

from pgvector import Vector
from pgvector.psycopg import register_vector

from database import get_connection
from main import (
    DOCUMENT_PATH,
    EMBEDDING_MODEL,
    load_document,
    chunk_document,
    generate_embeddings,
)


def ingest_document():
    document = load_document()
    chunks = chunk_document(document)

    if not chunks:
        raise ValueError("The document contains no usable chunks.")

    embeddings = generate_embeddings(chunks)

    project_dir = Path(__file__).resolve().parent
    document_path = DOCUMENT_PATH.relative_to(project_dir).as_posix()

    rows = []

    for index, (chunk, embedding) in enumerate(
        zip(chunks, embeddings), start=1
    ):
        if len(embedding) != 384:
            raise ValueError("The database expects 384-dimensional embeddings.")

        rows.append((
            document_path,
            index,
            chunk,
            EMBEDDING_MODEL,
            Vector(embedding),
        ))

    with get_connection() as connection:
        register_vector(connection)

        with connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM document_chunks WHERE document_path = %s",
                (document_path,),
            )

            cursor.executemany(
                """
                INSERT INTO document_chunks (
                    document_path,
                    chunk_index,
                    content,
                    embedding_model,
                    embedding
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                rows,
            )

    print(f"Saved {len(rows)} chunks for {document_path}.")


if __name__ == "__main__":
    ingest_document()