from __future__ import annotations

from io import BytesIO
from pathlib import Path
import os
import shutil
import tempfile
from typing import List

import faiss
import numpy as np
import pdfplumber
from groq import Groq
from sklearn.feature_extraction.text import TfidfVectorizer


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"}


def is_audio_transcription_available() -> tuple[bool, str]:
    try:
        import whisper  # noqa: F401
    except Exception:
        return False, "Whisper package not available."

    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg binary is not available in this environment."

    return True, "Audio transcription is available."


class AdvanceRAG:
    def __init__(
        self,
        groq_api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        chunk_size: int = 400,
        top_k: int = 3,
    ) -> None:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Missing GROQ API key. Set GROQ_API_KEY or pass groq_api_key.")

        self.client = Groq(api_key=key)
        self.model = model
        self.chunk_size = chunk_size
        self.top_k = top_k

        self.chunks: List[str] = []
        self.chunk_sources: List[str] = []
        self.vectorizer: TfidfVectorizer | None = None
        self.index: faiss.IndexFlatL2 | None = None

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def reset(self) -> None:
        self.chunks = []
        self.chunk_sources = []
        self.vectorizer = None
        self.index = None

    def ingest_text(self, text: str, source: str = "text") -> int:
        cleaned = _normalize_text(text)
        if not cleaned:
            return 0

        new_chunks = _chunk_by_words(cleaned, self.chunk_size)
        self.chunks.extend(new_chunks)
        self.chunk_sources.extend([source] * len(new_chunks))
        return len(new_chunks)

    def ingest_txt_bytes(self, data: bytes, source: str = "text.txt") -> int:
        text = data.decode("utf-8", errors="ignore")
        return self.ingest_text(text, source=source)

    def ingest_pdf_bytes(self, data: bytes, source: str = "document.pdf") -> int:
        text_parts: list[str] = []
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)

        return self.ingest_text("\n".join(text_parts), source=source)

    def transcribe_audio_bytes(self, data: bytes, filename: str, whisper_model: str = "base") -> str:
        available, reason = is_audio_transcription_available()
        if not available:
            raise RuntimeError(f"Audio transcription unavailable: {reason}")

        import whisper

        suffix = Path(filename).suffix.lower() or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(data)
            temp_path = tmp.name

        try:
            model = whisper.load_model(whisper_model)
            result = model.transcribe(temp_path)
            return _normalize_text(result.get("text", ""))
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def build_index(self) -> None:
        if not self.chunks:
            raise ValueError("No chunks available. Ingest data before building the index.")

        self.vectorizer = TfidfVectorizer()
        x_matrix = self.vectorizer.fit_transform(self.chunks)
        embeddings = x_matrix.toarray().astype("float32")

        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def retrieve(self, query: str, top_k: int | None = None) -> list[str]:
        if not self.is_ready or self.vectorizer is None or self.index is None:
            raise ValueError("Index is not ready. Build the index before querying.")

        k = top_k or self.top_k
        q_vec = self.vectorizer.transform([query]).toarray().astype("float32")
        _, indices = self.index.search(q_vec, k)

        results: list[str] = []
        for idx in indices[0]:
            if idx == -1:
                continue
            results.append(self.chunks[int(idx)])
        return results

    def answer(self, query: str, top_k: int | None = None, temperature: float = 0.0) -> dict:
        docs = self.retrieve(query, top_k=top_k)
        context = "\n\n".join(docs)

        prompt = (
            "Answer ONLY using the context below. "
            "If the answer is not present, reply with: 'I do not have enough context to answer that.'\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )

        return {
            "answer": response.choices[0].message.content,
            "context_docs": docs,
        }


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _chunk_by_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]
