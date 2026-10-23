import os
from pathlib import Path

import psycopg
from dotenv import load_dotenv


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


if __name__ == "__main__":
    with get_connection() as connection:
        row = connection.execute("SELECT current_database();").fetchone()
        print(f"Connected to database: {row[0]}")