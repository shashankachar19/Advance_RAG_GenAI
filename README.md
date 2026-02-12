# Advance RAG Studio

A modern Retrieval-Augmented Generation (RAG) app built from your notebook workflow.

## What You Get

- `main.py`: reusable backend engine for ingestion, chunking, indexing, retrieval, and answer generation
- `app.py`: modern Streamlit UI with polished styling, chat layout, and live system status cards
- `requirements.txt`: project dependencies

## Features

- Ingest files: `.txt`, `.md`, `.pdf`, `.wav`, `.mp3`, `.mp4`, `.m4a`, `.flac`, `.ogg`, `.png`, `.jpg`, `.jpeg`, `.webp`
- Audio transcription via Whisper
- Text chunking by words
- Multimodal RAG with image captions (Groq vision model)
- Embeddings via Jina Embeddings v4 API
- Vector search with FAISS
- Cross-encoder reranking for improved relevance
- Guardrails for context-only answers
- Session memory and latency tracking in the UI
- Optional vision-at-answer for image Q&A plus image previews in context
- Metadata filtering (text-only / image-only / both)

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set your API keys:

```bash
set GROQ_API_KEY=your_groq_api_key_here
set JINA_API_KEY=your_jina_api_key_here
```

Or provide them in the Streamlit sidebar at runtime.

## Run UI

```bash
streamlit run app.py
```

## Project Structure

```text
Advance_RAG/
  app.py
  main.py
  requirements.txt
  README.md
  Advance_RAG(11_02_2026).ipynb
```

## Backend Usage (Optional)

```python
from main import AdvanceRAG

rag = AdvanceRAG(groq_api_key="your_key")
rag.ingest_txt_bytes(open("notes.txt", "rb").read(), source="notes.txt")
rag.build_index()
print(rag.answer("Summarize the key topics"))
```

## Notes

- Do not hardcode API keys in source files.
- First run of Whisper may download model weights, so audio transcription can take longer.
- For large files, prefer smaller `top_k` and tune `chunk_size` from the UI sidebar.
- Cross-encoder reranking uses `sentence-transformers` and can add startup latency on first load.

## Streamlit Cloud Deployment

1. Push this repo to GitHub.
2. In Streamlit Cloud, set `app.py` as the entrypoint.
3. Add secret in app settings:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
JINA_API_KEY = "your_jina_api_key_here"
```

4. Deploy.

Notes:
- `runtime.txt` pins Python for better binary wheel compatibility on Streamlit Cloud.
- Audio transcription needs both `openai-whisper` and `ffmpeg`; if unavailable, the app still runs and skips audio files.

