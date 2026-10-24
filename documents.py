from config import DOCUMENT_PATH


def load_document():
    text = DOCUMENT_PATH.read_text(encoding="utf-8")
    return text.strip()


def chunk_document(text):
    paragraphs = text.split("\n\n")
    chunks = []

    for paragraph in paragraphs:
        chunk = paragraph.strip()

        if chunk:
            chunks.append(chunk)

    return chunks
