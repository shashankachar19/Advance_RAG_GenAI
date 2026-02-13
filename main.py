from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
import base64
from datetime import datetime, timezone
import os
import re
import shutil
import tempfile
import time
from typing import Any

import faiss
import numpy as np
import pdfplumber
import requests
from groq import Groq
from openai import OpenAI


SUPPORTED_AUDIO_SUFFIXES = {".wav", ".mp3", ".mp4", ".m4a", ".flac", ".ogg"}
SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


def is_audio_transcription_available() -> tuple[bool, str]:
    try:
        import whisper  # noqa: F401
    except Exception:
        return False, "Whisper package not available."

    if shutil.which("ffmpeg") is None:
        return False, "ffmpeg binary is not available in this environment."

    return True, "Audio transcription is available."


@dataclass
class ChunkRecord:
    text: str
    source: str
    modality: str
    meta: dict[str, Any]


class JinaEmbeddingsClient:
    def __init__(self, api_key: str, model: str = "jina-embeddings-v4", dimensions: int | None = None) -> None:
        if not api_key:
            raise ValueError("Missing Jina API key. Set JINA_API_KEY or pass jina_api_key.")
        self.api_key = api_key
        self.model = model
        self.dimensions = dimensions
        self.endpoint = "https://api.jina.ai/v1/embeddings"

    def embed_texts(self, texts: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self.model,
            "task": task,
            "embedding_type": "float",
            "input": texts,
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return self._post_embeddings(payload)

    def embed_images(self, images_b64: list[str], task: str) -> list[list[float]]:
        payload = {
            "model": self.model,
            "task": task,
            "embedding_type": "float",
            "input": [{"image": image_b64} for image_b64 in images_b64],
        }
        if self.dimensions:
            payload["dimensions"] = self.dimensions
        return self._post_embeddings(payload)

    def _post_embeddings(self, payload: dict[str, Any]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        response = requests.post(self.endpoint, json=payload, headers=headers, timeout=60)
        response.raise_for_status()
        data = response.json().get("data", [])
        embeddings: list[list[float]] = []
        for item in data:
            if isinstance(item.get("embedding"), dict):
                embeddings.append(item["embedding"].get("float", []))
            else:
                embeddings.append(item.get("embedding", []))
        return embeddings


class AdvanceRAG:
    def __init__(
        self,
        groq_api_key: str | None = None,
        jina_api_key: str | None = None,
        model: str = "llama-3.1-8b-instant",
        vision_model: str = "meta-llama/llama-4-scout-17b-16e-instruct",
        chunk_size: int = 400,
        top_k: int = 3,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ) -> None:
        key = groq_api_key or os.getenv("GROQ_API_KEY")
        if not key:
            raise ValueError("Missing GROQ API key. Set GROQ_API_KEY or pass groq_api_key.")

        jina_key = jina_api_key or os.getenv("JINA_API_KEY")
        if not jina_key:
            raise ValueError("Missing Jina API key. Set JINA_API_KEY or pass jina_api_key.")

        self.client = Groq(api_key=key)
        self.vision_client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
        self.embeddings = JinaEmbeddingsClient(jina_key)

        self.model = model
        self.vision_model = vision_model
        self.chunk_size = chunk_size
        self.top_k = top_k
        self.rerank_model = rerank_model

        self.chunks: list[ChunkRecord] = []
        self.index: faiss.IndexFlatL2 | None = None
        self.modality_indices: dict[str, faiss.IndexFlatL2] = {}
        self.modality_chunk_maps: dict[str, list[int]] = {}
        self.emb_matrix: np.ndarray | None = None
        self._reranker = None

    @property
    def is_ready(self) -> bool:
        return self.index is not None and len(self.chunks) > 0

    def reset(self) -> None:
        self.chunks = []
        self.index = None
        self.modality_indices = {}
        self.modality_chunk_maps = {}
        self.emb_matrix = None

    def ingest_text(self, text: str, source: str = "text", meta: dict[str, Any] | None = None) -> int:
        cleaned = _normalize_text(text)
        if not cleaned:
            return 0

        merged_meta = self._build_base_meta(source, meta)
        policy_flags = scan_policy_flags(cleaned)
        merged_meta.update(policy_flags)
        processed_text = redact_pii(cleaned)

        new_chunks = _chunk_by_words(processed_text, self.chunk_size)
        for chunk in new_chunks:
            self.chunks.append(
                ChunkRecord(
                    text=chunk,
                    source=source,
                    modality="text",
                    meta=dict(merged_meta),
                )
            )
        return len(new_chunks)

    def ingest_txt_bytes(self, data: bytes, source: str = "text.txt") -> int:
        text = data.decode("utf-8", errors="ignore")
        return self.ingest_text(text, source=source, meta={"file_type": "txt"})

    def ingest_pdf_bytes(self, data: bytes, source: str = "document.pdf") -> int:
        added = 0
        with pdfplumber.open(BytesIO(data)) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                page_text = page.extract_text() or ""
                if page_text.strip():
                    added += self.ingest_text(
                        page_text,
                        source=source,
                        meta={"file_type": "pdf", "page": page_num, "source_name": source},
                    )

        added += self._ingest_pdf_images(data, source=source)
        return added

    def ingest_image_bytes(self, data: bytes, source: str = "image.png") -> int:
        caption = self.caption_image_bytes(data, source=source)
        if not caption:
            return 0
        image_b64 = base64.b64encode(data).decode("ascii")
        formatted = f"[Image: {source}] {caption}"
        merged_meta = self._build_base_meta(source, {"file_type": _suffix_to_file_type(source), "source_name": source})
        merged_meta.update(scan_policy_flags(caption))
        merged_meta["caption"] = caption
        merged_meta["image_b64"] = image_b64
        self.chunks.append(
            ChunkRecord(
                text=formatted,
                source=source,
                modality="image",
                meta=merged_meta,
            )
        )
        return 1

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

    def caption_image_bytes(self, data: bytes, source: str) -> str:
        mime = _guess_mime(source)
        image_b64 = base64.b64encode(data).decode("ascii")
        image_url = f"data:{mime};base64,{image_b64}"

        prompt = (
            "Generate a concise, factual caption describing the image. "
            "Focus on visible details only."
        )
        response = self.vision_client.responses.create(
            model=self.vision_model,
            input=[
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": prompt},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        )
        caption = (response.output_text or "").strip()
        return caption

    def build_index(self) -> None:
        if not self.chunks:
            raise ValueError("No chunks available. Ingest data before building the index.")

        text_indices = [idx for idx, chunk in enumerate(self.chunks) if chunk.modality == "text"]
        image_indices = [idx for idx, chunk in enumerate(self.chunks) if chunk.modality == "image"]

        embeddings: list[list[float]] = [None] * len(self.chunks)

        if text_indices:
            texts = [self.chunks[idx].text for idx in text_indices]
            text_embs = self.embeddings.embed_texts(texts, task="retrieval.passage")
            for idx, emb in zip(text_indices, text_embs):
                embeddings[idx] = emb

        if image_indices:
            images_b64 = [self.chunks[idx].meta.get("image_b64", "") for idx in image_indices]
            image_embs = self.embeddings.embed_images(images_b64, task="retrieval.passage")
            for idx, emb in zip(image_indices, image_embs):
                embeddings[idx] = emb

        if any(emb is None for emb in embeddings):
            raise ValueError("Embedding generation failed for one or more chunks.")

        emb_array = np.asarray(embeddings, dtype="float32")
        self.emb_matrix = emb_array
        self.index = faiss.IndexFlatL2(emb_array.shape[1])
        self.index.add(emb_array)
        self._build_modality_indices()

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        modality_filter: str = "both",
        metadata_filter: dict[str, Any] | None = None,
    ) -> list[ChunkRecord]:
        if not self.is_ready or self.index is None or self.emb_matrix is None:
            raise ValueError("Index is not ready. Build the index before querying.")

        k = top_k or self.top_k
        query_emb = self.embeddings.embed_texts([query], task="retrieval.query")[0]
        q_vec = np.asarray([query_emb], dtype="float32")
        index, map_back = self._select_index_for_filter(modality_filter)
        if index is None or index.ntotal == 0:
            return []

        filter_cfg = metadata_filter or {}
        search_k = min(index.ntotal, max(k, k * 4))
        results: list[ChunkRecord] = []

        while search_k <= index.ntotal and len(results) < k:
            _, indices = index.search(q_vec, search_k)
            seen: set[int] = set()
            filtered: list[ChunkRecord] = []
            for local_idx in indices[0]:
                if local_idx == -1:
                    continue
                chunk_idx = int(local_idx) if map_back is None else map_back[int(local_idx)]
                if chunk_idx in seen:
                    continue
                seen.add(chunk_idx)
                record = self.chunks[chunk_idx]
                if self._record_matches_filters(record, filter_cfg):
                    filtered.append(record)
                if len(filtered) >= k:
                    break
            results = filtered
            if len(results) >= k or search_k == index.ntotal:
                break
            search_k = min(index.ntotal, search_k * 2)

        return results[:k]

    def rerank(
        self,
        query: str,
        docs: list[ChunkRecord],
        top_k: int,
        return_scores: bool = False,
    ) -> list[ChunkRecord] | tuple[list[ChunkRecord], list[float]]:
        if not docs:
            return ([], []) if return_scores else []

        if self._reranker is None:
            try:
                from sentence_transformers import CrossEncoder

                self._reranker = CrossEncoder(self.rerank_model)
            except Exception as exc:
                raise RuntimeError(f"Reranker unavailable: {exc}") from exc

        pairs = [(query, doc.text) for doc in docs]
        scores = self._reranker.predict(pairs)
        ranked_pairs = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
        ranked_docs = [doc for _, doc in ranked_pairs][:top_k]
        ranked_scores = [float(score) for score, _ in ranked_pairs][:top_k]
        if return_scores:
            return ranked_docs, ranked_scores
        return ranked_docs

    def answer(
        self,
        query: str,
        top_k: int | None = None,
        candidate_k: int | None = None,
        modality_filter: str = "both",
        use_rerank: bool = True,
        use_vision: bool = False,
        enforce_guardrails: bool = True,
        answer_model: str | None = None,
        metadata_filter: dict[str, Any] | None = None,
        memory: list[dict[str, str]] | None = None,
        temperature: float = 0.0,
    ) -> dict:
        timing: dict[str, float] = {}
        start_total = time.perf_counter()

        start_retrieval = time.perf_counter()
        candidates = self.retrieve(
            query,
            top_k=candidate_k or (top_k or self.top_k),
            modality_filter=modality_filter,
            metadata_filter=metadata_filter,
        )
        timing["retrieval_s"] = time.perf_counter() - start_retrieval

        docs = candidates
        retrieval_scores: list[float] = []
        if use_rerank:
            start_rerank = time.perf_counter()
            docs, retrieval_scores = self.rerank(
                query,
                candidates,
                top_k or self.top_k,
                return_scores=True,
            )
            timing["rerank_s"] = time.perf_counter() - start_rerank

        context = "\n\n".join([doc.text for doc in docs])
        memory_section = _format_memory(memory or [])
        fallback = "I do not have enough context to answer that."

        system_msg = (
            "You are a grounded assistant. Answer ONLY using the provided context. "
            f"If the answer is not present, reply: '{fallback}'"
        )
        user_msg = (
            f"{memory_section}\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\n"
            "Answer:"
        )

        chosen_model = answer_model or self.model
        start_gen = time.perf_counter()
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        if use_vision and any(doc.modality == "image" for doc in docs):
            answer_text, usage = self._answer_with_vision(system_msg, user_msg, docs, temperature)
        else:
            response = self.client.chat.completions.create(
                model=chosen_model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
                temperature=temperature,
            )
            answer_text = response.choices[0].message.content
            if getattr(response, "usage", None):
                usage["prompt_tokens"] = int(getattr(response.usage, "prompt_tokens", 0) or 0)
                usage["completion_tokens"] = int(getattr(response.usage, "completion_tokens", 0) or 0)
                usage["total_tokens"] = int(getattr(response.usage, "total_tokens", 0) or 0)
        timing["generation_s"] = time.perf_counter() - start_gen

        grounded = True
        if enforce_guardrails:
            start_guard = time.perf_counter()
            is_grounded = self._is_grounded_answer(query, context, answer_text, fallback, chosen_model)
            timing["guardrail_s"] = time.perf_counter() - start_guard
            grounded = bool(is_grounded)
            if not is_grounded:
                answer_text = fallback

        timing["total_s"] = time.perf_counter() - start_total

        return {
            "answer": answer_text,
            "context_docs": [doc.__dict__ for doc in docs],
            "latency": timing,
            "usage": usage,
            "model_used": chosen_model,
            "grounded": grounded,
            "retrieval_scores": retrieval_scores,
        }

    def _ingest_pdf_images(self, data: bytes, source: str) -> int:
        try:
            import fitz  # pymupdf
        except Exception:
            return 0

        added = 0
        doc = fitz.open(stream=data, filetype="pdf")
        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            for image_index, img in enumerate(page.get_images(full=True), start=1):
                xref = img[0]
                image_dict = doc.extract_image(xref)
                if not image_dict:
                    continue
                image_bytes = image_dict.get("image")
                ext = image_dict.get("ext", "png")
                if not image_bytes:
                    continue
                name = f"{Path(source).stem}-p{page_index + 1}-img{image_index}.{ext}"
                caption = self.caption_image_bytes(image_bytes, source=name)
                if not caption:
                    continue
                formatted = f"[Image: {name}] {caption}"
                image_b64 = base64.b64encode(image_bytes).decode("ascii")
                self.chunks.append(
                    ChunkRecord(
                        text=formatted,
                        source=source,
                        modality="image",
                        meta=self._build_pdf_image_meta(source, page_index + 1, name, caption, image_b64),
                    )
                )
                added += 1
        doc.close()
        return added

    def has_pdf_image_support(self) -> bool:
        try:
            import fitz  # noqa: F401
            return True
        except Exception:
            return False

    def _answer_with_vision(
        self,
        system_msg: str,
        user_msg: str,
        docs: list[ChunkRecord],
        temperature: float,
    ) -> tuple[str, dict[str, int]]:
        content: list[dict[str, str]] = [{"type": "input_text", "text": f"{system_msg}\n\n{user_msg}"}]
        for doc in docs:
            if doc.modality != "image":
                continue
            image_b64 = doc.meta.get("image_b64")
            if not image_b64:
                continue
            content.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/png;base64,{image_b64}",
                }
            )

        response = self.vision_client.responses.create(
            model=self.vision_model,
            input=[{"role": "user", "content": content}],
            temperature=temperature,
        )
        usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        resp_usage = getattr(response, "usage", None)
        if resp_usage:
            usage["prompt_tokens"] = int(getattr(resp_usage, "input_tokens", 0) or 0)
            usage["completion_tokens"] = int(getattr(resp_usage, "output_tokens", 0) or 0)
            usage["total_tokens"] = usage["prompt_tokens"] + usage["completion_tokens"]
        return (response.output_text or "").strip(), usage

    def _build_modality_indices(self) -> None:
        if self.emb_matrix is None or len(self.chunks) == 0:
            self.modality_indices = {}
            self.modality_chunk_maps = {}
            return

        dim = self.emb_matrix.shape[1]
        self.modality_indices = {}
        self.modality_chunk_maps = {}

        for modality in ("text", "image"):
            chunk_ids = [idx for idx, chunk in enumerate(self.chunks) if chunk.modality == modality]
            if not chunk_ids:
                continue
            sub_embs = self.emb_matrix[chunk_ids]
            sub_index = faiss.IndexFlatL2(dim)
            sub_index.add(sub_embs)
            self.modality_indices[modality] = sub_index
            self.modality_chunk_maps[modality] = chunk_ids

    def _select_index_for_filter(
        self,
        modality_filter: str,
    ) -> tuple[faiss.IndexFlatL2 | None, list[int] | None]:
        if modality_filter == "both":
            return self.index, None
        if modality_filter in self.modality_indices:
            return self.modality_indices[modality_filter], self.modality_chunk_maps[modality_filter]
        return None, None

    def _is_grounded_answer(self, query: str, context: str, answer: str, fallback: str, judge_model: str) -> bool:
        normalized = (answer or "").strip()
        if not normalized:
            return False
        if normalized == fallback:
            return True
        if not context.strip():
            return False

        judge_prompt = (
            "Decide if the answer is fully supported by the context.\n"
            "Reply with exactly SUPPORTED or UNSUPPORTED.\n\n"
            f"Question:\n{query}\n\n"
            f"Context:\n{context}\n\n"
            f"Answer:\n{answer}\n"
        )
        try:
            verdict = self.client.chat.completions.create(
                model=judge_model,
                messages=[{"role": "user", "content": judge_prompt}],
                temperature=0.0,
            ).choices[0].message.content
        except Exception:
            return False
        return "SUPPORTED" in (verdict or "").upper()

    def _record_matches_filters(self, record: ChunkRecord, filters: dict[str, Any]) -> bool:
        source = (filters.get("source") or "").strip().lower()
        if source and source not in record.source.lower():
            return False

        file_type = (filters.get("file_type") or "").strip().lower()
        if file_type and str(record.meta.get("file_type", "")).lower() != file_type:
            return False

        page = filters.get("page")
        if page is not None and page != "":
            try:
                page_num = int(page)
            except Exception:
                page_num = None
            if page_num is not None and int(record.meta.get("page", -1)) != page_num:
                return False

        safe_only = bool(filters.get("safe_only", False))
        if safe_only and bool(record.meta.get("has_prompt_injection", False)):
            return False

        return True

    def _build_base_meta(self, source: str, meta: dict[str, Any] | None) -> dict[str, Any]:
        base = dict(meta or {})
        base.setdefault("source_name", source)
        base.setdefault("file_type", _suffix_to_file_type(source))
        base.setdefault("upload_ts", utc_now_iso())
        return base

    def _build_pdf_image_meta(
        self,
        source: str,
        page: int,
        image_name: str,
        caption: str,
        image_b64: str,
    ) -> dict[str, Any]:
        meta = self._build_base_meta(source, {"file_type": "pdf", "page": page, "image": image_name})
        meta.update(scan_policy_flags(caption))
        meta["caption"] = caption
        meta["image_b64"] = image_b64
        return meta


def _normalize_text(text: str) -> str:
    return " ".join(text.split())


def _chunk_by_words(text: str, chunk_size: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    return [" ".join(words[i : i + chunk_size]) for i in range(0, len(words), chunk_size)]


def _guess_mime(filename: str) -> str:
    suffix = Path(filename).suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".webp":
        return "image/webp"
    return "image/png"


def _format_memory(memory: list[dict[str, str]]) -> str:
    if not memory:
        return ""
    lines = ["Recent conversation (for reference only):"]
    for item in memory:
        role = item.get("role", "user")
        content = item.get("content", "")
        lines.append(f"{role.title()}: {content}")
    return "\n".join(lines) + "\n\n"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _suffix_to_file_type(source: str) -> str:
    suffix = Path(source).suffix.lower().lstrip(".")
    return suffix or "text"


PII_PATTERNS = [
    re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b"),
    re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{3}\)?[-.\s]?)\d{3}[-.\s]?\d{4}\b"),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
]

PROMPT_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"system\s+prompt", re.IGNORECASE),
    re.compile(r"developer\s+message", re.IGNORECASE),
    re.compile(r"do\s+not\s+follow\s+the\s+above", re.IGNORECASE),
]


def scan_policy_flags(text: str) -> dict[str, Any]:
    has_pii = any(pattern.search(text) for pattern in PII_PATTERNS)
    has_prompt_injection = any(pattern.search(text) for pattern in PROMPT_INJECTION_PATTERNS)
    return {
        "has_pii": bool(has_pii),
        "has_prompt_injection": bool(has_prompt_injection),
    }


def redact_pii(text: str) -> str:
    redacted = text
    redacted = re.sub(PII_PATTERNS[0], "[REDACTED_EMAIL]", redacted)
    redacted = re.sub(PII_PATTERNS[1], "[REDACTED_PHONE]", redacted)
    redacted = re.sub(PII_PATTERNS[2], "[REDACTED_SSN]", redacted)
    return redacted
