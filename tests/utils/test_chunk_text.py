import pytest
from flib.utils.chunk_text import get_text_chunks

def test_get_text_chunks():
    text = "This is a test text for chunking."
    chunk_size = 10
    chunk_overlap = 2
    chunks = get_text_chunks(text, chunk_size, chunk_overlap)
    assert len(chunks) == 4  # Expected number of chunks
    assert all(isinstance(chunk, str) for chunk in chunks)
