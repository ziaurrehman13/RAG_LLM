"""Beginner-friendly PDF RAG pipeline: extract, chunk, retrieve, generate."""

from __future__ import annotations

import re
from typing import Any

import fitz  # PyMuPDF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline

MODEL_NAME = "google/flan-t5-small"


def extract_pdf_pages(pdf_bytes: bytes) -> list[dict[str, Any]]:
    """Extract selectable text from each PDF page. Scanned PDFs need OCR first."""
    pages: list[dict[str, Any]] = []
    with fitz.open(stream=pdf_bytes, filetype="pdf") as pdf:
        for page_number, page in enumerate(pdf, start=1):
            page_text = page.get_text("text").strip()
            if page_text:
                pages.append({"page": page_number, "text": page_text})
    return pages


def split_into_chunks(
    pages: list[dict[str, Any]], chunk_size: int = 900, overlap: int = 150
) -> list[dict[str, Any]]:
    """Split page text into overlapping character chunks and retain page numbers."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("Use chunk_size > overlap >= 0.")

    chunks: list[dict[str, Any]] = []
    for page in pages:
        text = re.sub(r"\s+", " ", page["text"]).strip()
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            passage = text[start:end].strip()
            if passage:
                chunks.append({"page": page["page"], "text": passage})
            if end >= len(text):
                break
            start = end - overlap
    return chunks


def build_index(chunks: list[dict[str, Any]]) -> tuple[TfidfVectorizer, Any]:
    """Create a simple, free lexical retrieval index using TF-IDF."""
    if not chunks:
        raise ValueError("No text chunks were found in the PDF.")
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    matrix = vectorizer.fit_transform([chunk["text"] for chunk in chunks])
    return vectorizer, matrix


def retrieve(
    question: str,
    chunks: list[dict[str, Any]],
    vectorizer: TfidfVectorizer,
    matrix: Any,
    top_k: int = 3,
) -> list[dict[str, Any]]:
    """Return the most relevant passages, ordered by cosine similarity."""
    question_vector = vectorizer.transform([question])
    scores = cosine_similarity(question_vector, matrix).ravel()
    ranked_indices = scores.argsort()[::-1][:top_k]
    results = []
    for index in ranked_indices:
        if scores[index] > 0:
            result = dict(chunks[int(index)])
            result["score"] = float(scores[index])
            results.append(result)
    return results


def load_generator():
    """Load a small open model locally; no paid API key is required."""
    return pipeline(
        task="text2text-generation",
        model=MODEL_NAME,
        device=-1,  # CPU, compatible with standard free hosting
    )


def generate_answer(question: str, passages: list[dict[str, Any]], generator) -> str:
    """Generate an answer grounded in retrieved passages."""
    if not passages:
        return "I could not find relevant information in the uploaded PDF. Try another question."

    # Keep context compact because FLAN-T5-small has a limited input length.
    context_parts = []
    remaining_chars = 2200
    for passage in passages:
        snippet = passage["text"][:remaining_chars]
        if not snippet:
            break
        context_parts.append(f"[PDF page {passage['page']}] {snippet}")
        remaining_chars -= len(snippet)
        if remaining_chars <= 0:
            break

    # Build the context outside the f-string. This syntax is compatible with Python 3.11.
    context_text = "\n\n".join(context_parts)
    prompt = (
        "Answer the question using only the context. If the context does not contain the answer, "
        "say that you could not find it in the PDF. Do not invent facts. Keep the answer concise.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}\nAnswer:"
    )
    result = generator(
        prompt,
        max_new_tokens=160,
        do_sample=False,
        truncation=True,
    )
    answer = result[0]["generated_text"].strip()
    return answer or "The model did not produce an answer. Try rephrasing the question."
