from __future__ import annotations

from pathlib import Path
import streamlit as st

from main import (
    AdvanceRAG,
    SUPPORTED_AUDIO_SUFFIXES,
    SUPPORTED_IMAGE_SUFFIXES,
    is_audio_transcription_available,
)


st.set_page_config(
    page_title="Advance RAG Studio",
    page_icon="AI",
    layout="wide",
)


def apply_custom_css(theme_mode: str) -> None:
    is_dark = theme_mode.lower() == "dark"

    if is_dark:
        bg_a = "#0b1220"
        bg_b = "#0f1a17"
        ink = "#e7ecff"
        ink_soft = "#9aa7c7"
        line = "rgba(156, 172, 214, 0.2)"
        panel = "rgba(18, 27, 48, 0.72)"
        side_bg = "rgba(12, 18, 33, 0.7)"
        shadow = "0 10px 35px rgba(0, 0, 0, 0.45)"
    else:
        bg_a = "#eef4ff"
        bg_b = "#f8fff1"
        ink = "#15182b"
        ink_soft = "#58607a"
        line = "rgba(30, 37, 72, 0.12)"
        panel = "rgba(255, 255, 255, 0.72)"
        side_bg = "rgba(255, 255, 255, 0.65)"
        shadow = "0 8px 30px rgba(17, 24, 39, 0.06)"

    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=IBM+Plex+Mono:wght@400;500&display=swap');

        :root {{
            --bg-a: {bg_a};
            --bg-b: {bg_b};
            --ink: {ink};
            --ink-soft: {ink_soft};
            --line: {line};
            --accent: #0f9d8f;
            --accent-2: #2979ff;
            --panel: {panel};
            --side-bg: {side_bg};
            --shadow: {shadow};
        }}

        .stApp {{
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
            background:
                radial-gradient(circle at 10% 10%, rgba(41, 121, 255, 0.14), transparent 35%),
                radial-gradient(circle at 90% 10%, rgba(15, 157, 143, 0.14), transparent 35%),
                linear-gradient(135deg, var(--bg-a), var(--bg-b));
            transition: background 0.25s ease;
        }}

        section[data-testid="stSidebar"] {{
            border-right: 1px solid var(--line);
            background: var(--side-bg);
            backdrop-filter: blur(10px);
        }}

        .hero {{
            padding: 18px 22px;
            border: 1px solid var(--line);
            border-radius: 18px;
            background: var(--panel);
            backdrop-filter: blur(8px);
            box-shadow: var(--shadow);
            margin-bottom: 14px;
            animation: fadeIn 0.4s ease;
        }}

        .hero h1 {{
            margin: 0;
            font-size: 1.8rem;
            letter-spacing: 0.2px;
        }}

        .hero p {{
            margin: 6px 0 0;
            color: var(--ink-soft);
        }}

        .status-grid {{
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 10px;
            margin: 10px 0 18px;
        }}

        .status-card {{
            border: 1px solid var(--line);
            border-radius: 14px;
            background: var(--panel);
            padding: 12px;
        }}

        .status-label {{
            font-size: 0.72rem;
            text-transform: uppercase;
            color: var(--ink-soft);
            letter-spacing: 0.08em;
            margin-bottom: 6px;
        }}

        .status-value {{
            font-family: 'IBM Plex Mono', monospace;
            font-size: 1.02rem;
            font-weight: 500;
        }}

        @keyframes fadeIn {{
            from {{ opacity: 0; transform: translateY(8px); }}
            to {{ opacity: 1; transform: translateY(0); }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def get_engine(
    groq_api_key: str,
    jina_api_key: str,
    model: str,
    vision_model: str,
    chunk_size: int,
    top_k: int,
    rerank_model: str,
) -> AdvanceRAG:
    signature = (groq_api_key, jina_api_key, model, vision_model, chunk_size, top_k, rerank_model)
    if st.session_state.get("engine_signature") != signature:
        st.session_state.engine = AdvanceRAG(
            groq_api_key=groq_api_key,
            jina_api_key=jina_api_key,
            model=model,
            vision_model=vision_model,
            chunk_size=chunk_size,
            top_k=top_k,
            rerank_model=rerank_model,
        )
        st.session_state.engine_signature = signature
        st.session_state.chat_history = []
    return st.session_state.engine


def render_header(engine: AdvanceRAG | None) -> None:
    chunk_count = len(engine.chunks) if engine else 0
    text_count = sum(1 for chunk in engine.chunks if chunk.modality == "text") if engine else 0
    image_count = sum(1 for chunk in engine.chunks if chunk.modality == "image") if engine else 0
    source_count = len({chunk.source for chunk in engine.chunks}) if engine else 0
    index_state = "Ready" if engine and engine.is_ready else "Not Ready"

    st.markdown(
        """
        <div class="hero">
            <h1>Advance RAG Studio</h1>
            <p>Upload PDF/TXT/audio content, index it instantly, and chat with grounded answers.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        f"""
        <div class="status-grid">
            <div class="status-card">
                <div class="status-label">Chunks</div>
                <div class="status-value">{chunk_count}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Sources</div>
                <div class="status-value">{source_count}</div>
            </div>
            <div class="status-card">
                <div class="status-label">Index</div>
                <div class="status-value">{index_state}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption(f"Text chunks: {text_count} - Image chunks: {image_count}")


def main() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"

    audio_ok, audio_status = is_audio_transcription_available()

    with st.sidebar:
        st.subheader("Engine")
        st.session_state.theme_mode = st.toggle(
            "Dark Theme",
            value=st.session_state.theme_mode == "Dark",
        )
        theme_mode = "Dark" if st.session_state.theme_mode else "Light"

        st.subheader("API Keys")
        default_groq_key = st.secrets.get("GROQ_API_KEY", "") if hasattr(st, "secrets") else ""
        groq_api_key = st.text_input(
            "GROQ API Key",
            type="password",
            value=default_groq_key,
            help="Used in this session. Prefer Streamlit secrets in deployment.",
        )
        default_jina_key = st.secrets.get("JINA_API_KEY", "") if hasattr(st, "secrets") else ""
        jina_api_key = st.text_input(
            "Jina API Key",
            type="password",
            value=default_jina_key,
            help="Used for embeddings. Prefer Streamlit secrets in deployment.",
        )

        st.divider()
        st.subheader("Models")
        model = st.selectbox("Model", ["llama-3.1-8b-instant", "llama-3.3-70b-versatile", "mixtral-8x7b-32768"], index=0)
        vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        st.caption(f"Vision model: {vision_model}")
        rerank_model = st.selectbox(
            "Reranker",
            ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"],
            index=0,
        )
        chunk_size = st.slider("Chunk Size (words)", min_value=120, max_value=900, value=400, step=20)
        top_k = st.slider("Top K Retrieval", min_value=1, max_value=8, value=3, step=1)
        candidate_k = st.slider("Candidate K (before rerank)", min_value=3, max_value=20, value=8, step=1)
        use_rerank = st.toggle("Use Cross-Encoder Rerank", value=True)
        modality_filter = st.selectbox("Retrieve", ["both", "text", "image"], index=0)
        use_vision = st.toggle("Use Vision for Image Q&A", value=True)
        memory_turns = st.slider("Memory Turns", min_value=0, max_value=8, value=4, step=1)

        st.divider()
        st.subheader("Knowledge Files")
        uploads = st.file_uploader(
            "Upload one or more files",
            type=[
                "txt",
                "md",
                "pdf",
                "wav",
                "mp3",
                "mp4",
                "m4a",
                "flac",
                "ogg",
                "png",
                "jpg",
                "jpeg",
                "webp",
            ],
            accept_multiple_files=True,
        )

        process_files = st.button("Process + Build Index", use_container_width=True, type="primary")
        clear_kb = st.button("Clear Knowledge Base", use_container_width=True)

    apply_custom_css(theme_mode)

    engine = None
    if groq_api_key and jina_api_key:
        try:
            engine = get_engine(
                groq_api_key,
                jina_api_key,
                model,
                vision_model,
                chunk_size,
                top_k,
                rerank_model,
            )
        except Exception as exc:
            st.error(f"Unable to initialize engine: {exc}")
            return

    if clear_kb and engine:
        engine.reset()
        st.session_state.chat_history = []
        st.success("Knowledge base cleared.")

    render_header(engine)

    if not audio_ok:
        st.caption(f"Audio note: {audio_status} Audio files will be skipped.")

    if process_files:
        if not groq_api_key or not jina_api_key:
            st.warning("Add both GROQ and Jina API keys before processing files.")
        elif not uploads:
            st.warning("Upload at least one file first.")
        else:
            added = 0
            with st.spinner("Processing files and building index..."):
                for file in uploads:
                    file_name = file.name
                    suffix = Path(file_name).suffix.lower()
                    data = file.getvalue()

                    try:
                        if suffix in {".txt", ".md"}:
                            added += engine.ingest_txt_bytes(data, source=file_name)
                        elif suffix == ".pdf":
                            added += engine.ingest_pdf_bytes(data, source=file_name)
                        elif suffix in SUPPORTED_IMAGE_SUFFIXES:
                            added += engine.ingest_image_bytes(data, source=file_name)
                        elif suffix in SUPPORTED_AUDIO_SUFFIXES:
                            if not audio_ok:
                                st.warning(f"Skipped {file_name}: {audio_status}")
                                continue
                            transcript = engine.transcribe_audio_bytes(data, file_name, whisper_model="base")
                            if transcript:
                                added += engine.ingest_text(transcript, source=f"{file_name} (transcript)")
                        else:
                            st.info(f"Skipped unsupported file: {file_name}")
                    except Exception as exc:
                        st.error(f"Failed to process {file_name}: {exc}")

                if engine.chunks:
                    engine.build_index()
                    st.success(f"Indexed successfully. Added {added} chunks from {len(uploads)} file(s).")
                else:
                    st.warning("No usable text found in the uploaded files.")

    st.subheader("Chat")

    if not groq_api_key or not jina_api_key:
        st.info("Enter your GROQ + Jina API keys in the sidebar to start.")

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("context"):
                with st.expander("Retrieved Context"):
                    for idx, ctx in enumerate(message["context"], start=1):
                        st.caption(f"Chunk {idx} - {ctx.get('modality', 'text')} - {ctx.get('source', 'source')}")
                        st.write(ctx.get("text", ""))
                        if ctx.get("modality") == "image" and ctx.get("meta", {}).get("image_b64"):
                            st.image(ctx["meta"]["image_b64"], caption=ctx.get("meta", {}).get("image", ""))

    user_query = st.chat_input("Ask a question about your uploaded knowledge base...")
    if user_query:
        if not engine or not engine.is_ready:
            st.warning("Process files first so the FAISS index is ready.")
            return

        st.session_state.chat_history.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Generating grounded answer..."):
                try:
                    candidate_k = max(candidate_k, top_k)
                    memory = []
                    if memory_turns > 0:
                        memory = st.session_state.chat_history[:-1]
                        memory = memory[-(memory_turns * 2) :]
                    result = engine.answer(
                        user_query,
                        top_k=top_k,
                        candidate_k=candidate_k,
                        modality_filter=modality_filter,
                        use_rerank=use_rerank,
                        use_vision=use_vision,
                        memory=memory,
                    )
                    answer = result["answer"]
                    context_docs = result["context_docs"]
                    latency = result.get("latency", {})
                except Exception as exc:
                    answer = f"Error while generating answer: {exc}"
                    context_docs = []
                    latency = {}

            st.markdown(answer)
            if context_docs:
                with st.expander("Retrieved Context"):
                    for idx, ctx in enumerate(context_docs, start=1):
                        st.caption(f"Chunk {idx} - {ctx.get('modality', 'text')} - {ctx.get('source', 'source')}")
                        st.write(ctx.get("text", ""))
                        if ctx.get("modality") == "image" and ctx.get("meta", {}).get("image_b64"):
                            st.image(ctx["meta"]["image_b64"], caption=ctx.get("meta", {}).get("image", ""))

            if latency:
                with st.expander("Latency Breakdown"):
                    for key, value in latency.items():
                        st.write(f"{key}: {value:.2f}s")

        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "context": context_docs}
        )


if __name__ == "__main__":
    main()
