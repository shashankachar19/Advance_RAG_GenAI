from __future__ import annotations

from pathlib import Path
import time
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

MODEL_OPTIONS = [
    "llama-3.1-8b-instant",
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
]

# Approximate rates in USD per 1M tokens. Adjust in UI if needed.
MODEL_PRICING_DEFAULT = {
    "llama-3.1-8b-instant": {"input_per_m": 0.05, "output_per_m": 0.08},
    "llama-3.3-70b-versatile": {"input_per_m": 0.59, "output_per_m": 0.79},
    "mixtral-8x7b-32768": {"input_per_m": 0.24, "output_per_m": 0.24},
}


def get_secret_value(key: str, default: str = "") -> str:
    try:
        return str(st.secrets.get(key, default))
    except Exception:
        return default


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


def choose_model_for_query(
    query: str,
    routing_mode: str,
    manual_model: str,
    fast_model: str,
    accurate_model: str,
) -> str:
    if routing_mode == "manual":
        return manual_model
    complex_cues = ("compare", "analyze", "deep", "why", "tradeoff", "reason")
    is_complex = len(query) > 180 or any(cue in query.lower() for cue in complex_cues)
    return accurate_model if is_complex else fast_model


def parse_eval_cases(raw: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in raw.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = [p.strip() for p in stripped.split("|")]
        if len(parts) < 2:
            continue
        row = {
            "question": parts[0],
            "expected_answer": parts[1],
            "expected_source": parts[2] if len(parts) > 2 else "",
        }
        rows.append(row)
    return rows


def estimate_cost_usd(usage: dict, model_name: str, price_table: dict[str, dict[str, float]]) -> float:
    rates = price_table.get(model_name, {"input_per_m": 0.0, "output_per_m": 0.0})
    prompt_tokens = float(usage.get("prompt_tokens", 0) or 0)
    completion_tokens = float(usage.get("completion_tokens", 0) or 0)
    input_cost = (prompt_tokens / 1_000_000.0) * float(rates.get("input_per_m", 0.0))
    output_cost = (completion_tokens / 1_000_000.0) * float(rates.get("output_per_m", 0.0))
    return input_cost + output_cost


def update_model_telemetry(
    telemetry: dict,
    model_name: str,
    latency_total: float,
    usage: dict,
    grounded: bool,
    cost_usd: float,
) -> None:
    bucket = telemetry.setdefault(
        model_name,
        {
            "calls": 0,
            "total_latency_s": 0.0,
            "total_tokens": 0,
            "grounded_calls": 0,
            "estimated_cost_usd": 0.0,
        },
    )
    bucket["calls"] += 1
    bucket["total_latency_s"] += float(latency_total or 0.0)
    bucket["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
    if grounded:
        bucket["grounded_calls"] += 1
    bucket["estimated_cost_usd"] += float(cost_usd or 0.0)


def render_model_telemetry(telemetry: dict) -> None:
    if not telemetry:
        return
    rows = []
    for model_name, stats in telemetry.items():
        calls = max(1, int(stats.get("calls", 0)))
        rows.append(
            {
                "model": model_name,
                "calls": int(stats.get("calls", 0)),
                "avg_latency_s": float(stats.get("total_latency_s", 0.0)) / calls,
                "avg_tokens": int(stats.get("total_tokens", 0)) / calls,
                "grounding_rate": float(stats.get("grounded_calls", 0)) / calls,
                "est_cost_usd": float(stats.get("estimated_cost_usd", 0.0)),
            }
        )
    with st.expander("Model Telemetry"):
        st.dataframe(rows, use_container_width=True)


def telemetry_rows(telemetry: dict) -> list[dict]:
    rows = []
    for model_name, stats in telemetry.items():
        calls = max(1, int(stats.get("calls", 0)))
        rows.append(
            {
                "model": model_name,
                "calls": int(stats.get("calls", 0)),
                "avg_latency_s": float(stats.get("total_latency_s", 0.0)) / calls,
                "avg_tokens": int(stats.get("total_tokens", 0)) / calls,
                "grounding_rate": float(stats.get("grounded_calls", 0)) / calls,
                "est_cost_usd": float(stats.get("estimated_cost_usd", 0.0)),
            }
        )
    return rows


def record_query_log(
    logs: list[dict],
    query: str,
    model_used: str,
    grounded: bool,
    latency: dict,
    usage: dict,
    est_cost: float,
    context_docs: list[dict],
    retrieval_scores: list[float],
) -> None:
    logs.append(
        {
            "query": query,
            "model": model_used,
            "grounded": grounded,
            "retrieval_s": float(latency.get("retrieval_s", 0.0)),
            "rerank_s": float(latency.get("rerank_s", 0.0)),
            "generation_s": float(latency.get("generation_s", 0.0)),
            "total_s": float(latency.get("total_s", 0.0)),
            "tokens": int(usage.get("total_tokens", 0) or 0),
            "cost_usd": float(est_cost),
            "retrieved_count": len(context_docs),
            "sources": [str(doc.get("source", "")) for doc in context_docs],
            "retrieval_scores": retrieval_scores or [],
        }
    )
    if len(logs) > 200:
        del logs[:-200]


def render_analytics(
    engine: AdvanceRAG | None,
    query_logs: list[dict],
    telemetry: dict,
    eval_rows: list[dict],
    eval_metrics: dict,
) -> None:
    st.subheader("Analytics")

    st.markdown("Latency Timeline")
    if query_logs:
        timeline_rows = []
        for idx, row in enumerate(query_logs, start=1):
            timeline_rows.append(
                {
                    "query_idx": idx,
                    "retrieval_s": row.get("retrieval_s", 0.0),
                    "rerank_s": row.get("rerank_s", 0.0),
                    "generation_s": row.get("generation_s", 0.0),
                    "total_s": row.get("total_s", 0.0),
                }
            )
        st.line_chart(timeline_rows, x="query_idx", y=["retrieval_s", "rerank_s", "generation_s", "total_s"])
    else:
        st.caption("No query latency data yet.")

    st.markdown("Model Comparison")
    model_rows = telemetry_rows(telemetry)
    if model_rows:
        metric = st.selectbox("Model Metric", ["avg_latency_s", "grounding_rate", "est_cost_usd", "avg_tokens"], index=0)
        bar_rows = [{"model": row["model"], metric: row[metric]} for row in model_rows]
        st.bar_chart(bar_rows, x="model", y=metric)
        st.dataframe(model_rows, use_container_width=True)
    else:
        st.caption("No model telemetry data yet.")

    st.markdown("Source Contribution")
    if query_logs:
        source_hits: dict[str, int] = {}
        for row in query_logs:
            for source in row.get("sources", []):
                source_hits[source] = source_hits.get(source, 0) + 1
        source_rows = [{"source": key, "hits": val} for key, val in source_hits.items()]
        source_rows = sorted(source_rows, key=lambda x: x["hits"], reverse=True)[:20]
        if source_rows:
            st.bar_chart(source_rows, x="source", y="hits")
            st.dataframe(source_rows, use_container_width=True)
        else:
            st.caption("No source hits recorded.")
    else:
        st.caption("No query logs yet.")

    st.markdown("Retrieval Quality")
    if query_logs:
        all_scores: list[float] = []
        for row in query_logs:
            all_scores.extend([float(s) for s in row.get("retrieval_scores", [])])
        if all_scores:
            score_rows = [{"rank": idx + 1, "score": score} for idx, score in enumerate(all_scores[:200])]
            st.line_chart(score_rows, x="rank", y="score")
        else:
            st.caption("No rerank scores available yet. Turn on reranking to populate this.")
    else:
        st.caption("No retrieval data yet.")
    if eval_rows:
        st.dataframe(eval_rows, use_container_width=True)
    if eval_metrics:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Answer Match", f"{float(eval_metrics.get('answer_rate', 0.0)):.1%}")
        c2.metric("Retrieval Recall", f"{float(eval_metrics.get('recall_rate', 0.0)):.1%}")
        c3.metric("Grounding Rate", f"{float(eval_metrics.get('grounded_rate', 0.0)):.1%}")
        c4.metric("Avg Latency", f"{float(eval_metrics.get('avg_latency', 0.0)):.2f}s")

    st.markdown("Safety Dashboard")
    if engine and engine.chunks:
        safety_by_source: dict[str, dict[str, int]] = {}
        for chunk in engine.chunks:
            source = chunk.source
            if source not in safety_by_source:
                safety_by_source[source] = {"chunks": 0, "pii": 0, "prompt_injection": 0}
            safety_by_source[source]["chunks"] += 1
            if bool(chunk.meta.get("has_pii", False)):
                safety_by_source[source]["pii"] += 1
            if bool(chunk.meta.get("has_prompt_injection", False)):
                safety_by_source[source]["prompt_injection"] += 1
        safety_rows = []
        for source, vals in safety_by_source.items():
            safety_rows.append(
                {
                    "source": source,
                    "chunks": vals["chunks"],
                    "pii_flagged": vals["pii"],
                    "prompt_injection_flagged": vals["prompt_injection"],
                }
            )
        st.dataframe(safety_rows, use_container_width=True)
    else:
        st.caption("No indexed chunks for safety analysis.")


def main() -> None:
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "theme_mode" not in st.session_state:
        st.session_state.theme_mode = "Light"
    if "last_pipeline_latency" not in st.session_state:
        st.session_state.last_pipeline_latency = {}
    if "model_telemetry" not in st.session_state:
        st.session_state.model_telemetry = {}
    if "query_logs" not in st.session_state:
        st.session_state.query_logs = []
    if "eval_rows" not in st.session_state:
        st.session_state.eval_rows = []
    if "eval_metrics" not in st.session_state:
        st.session_state.eval_metrics = {}

    audio_ok, audio_status = is_audio_transcription_available()

    with st.sidebar:
        st.subheader("Engine")
        st.session_state.theme_mode = st.toggle(
            "Dark Theme",
            value=st.session_state.theme_mode == "Dark",
        )
        theme_mode = "Dark" if st.session_state.theme_mode else "Light"

        st.subheader("API Keys")
        default_groq_key = get_secret_value("GROQ_API_KEY", "")
        groq_api_key = st.text_input(
            "GROQ API Key",
            type="password",
            value=default_groq_key,
            help="Used in this session. Prefer Streamlit secrets in deployment.",
        )
        default_jina_key = get_secret_value("JINA_API_KEY", "")
        jina_api_key = st.text_input(
            "Jina API Key",
            type="password",
            value=default_jina_key,
            help="Used for embeddings. Prefer Streamlit secrets in deployment.",
        )

        st.divider()
        st.subheader("Models")
        routing_mode = st.selectbox("Routing Mode", ["manual", "auto"], index=0)
        model = st.selectbox("Manual Model", MODEL_OPTIONS, index=0)
        fast_model = st.selectbox("Fast Model", MODEL_OPTIONS, index=0)
        accurate_model = st.selectbox("Accurate Model", MODEL_OPTIONS, index=1)
        vision_model = "meta-llama/llama-4-scout-17b-16e-instruct"
        st.caption(f"Vision model: {vision_model}")
        rerank_model = st.selectbox(
            "Reranker",
            ["cross-encoder/ms-marco-MiniLM-L-6-v2", "cross-encoder/ms-marco-MiniLM-L-12-v2"],
            index=0,
        )
        st.caption("Estimated token pricing (USD per 1M tokens)")
        price_in = st.number_input("Input Token Rate", min_value=0.0, value=MODEL_PRICING_DEFAULT.get(model, {}).get("input_per_m", 0.0), step=0.01, format="%.4f")
        price_out = st.number_input("Output Token Rate", min_value=0.0, value=MODEL_PRICING_DEFAULT.get(model, {}).get("output_per_m", 0.0), step=0.01, format="%.4f")
        chunk_size = st.slider("Chunk Size (words)", min_value=120, max_value=900, value=400, step=20)
        top_k = st.slider("Top K Retrieval", min_value=1, max_value=8, value=3, step=1)
        candidate_k = st.slider("Candidate K (before rerank)", min_value=3, max_value=20, value=8, step=1)
        use_rerank = st.toggle("Use Cross-Encoder Rerank", value=True)
        modality_filter = st.selectbox("Retrieve", ["both", "text", "image"], index=0)
        use_vision = st.toggle("Use Vision for Image Q&A", value=True)
        enforce_guardrails = st.toggle("Enforce Guardrails", value=True)
        memory_turns = st.slider("Memory Turns", min_value=0, max_value=8, value=4, step=1)
        st.subheader("Metadata Filters")
        source_filter = st.text_input("Source Contains", value="")
        file_type_filter = st.selectbox(
            "File Type",
            ["any", "txt", "md", "pdf", "wav", "mp3", "mp4", "m4a", "flac", "ogg", "png", "jpg", "jpeg", "webp"],
            index=0,
        )
        page_filter = st.text_input("Page Number (PDF only)", value="")
        safe_only = st.toggle("Safe Content Only", value=False)

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
    if engine and not engine.has_pdf_image_support():
        st.caption("PDF image extraction note: PyMuPDF is unavailable, so PDF images will be skipped.")

    if process_files:
        if not groq_api_key or not jina_api_key:
            st.warning("Add both GROQ and Jina API keys before processing files.")
        elif not uploads:
            st.warning("Upload at least one file first.")
        else:
            added = 0
            ingest_total_s = 0.0
            index_s = 0.0
            with st.spinner("Processing files and building index..."):
                for file in uploads:
                    file_name = file.name
                    suffix = Path(file_name).suffix.lower()
                    data = file.getvalue()
                    file_start = time.perf_counter()

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
                    finally:
                        ingest_total_s += time.perf_counter() - file_start

                if engine.chunks:
                    index_start = time.perf_counter()
                    engine.build_index()
                    index_s = time.perf_counter() - index_start
                    st.session_state.last_pipeline_latency = {
                        "ingest_s": ingest_total_s,
                        "index_s": index_s,
                        "pipeline_s": ingest_total_s + index_s,
                    }
                    st.success(f"Indexed successfully. Added {added} chunks from {len(uploads)} file(s).")
                else:
                    st.warning("No usable text found in the uploaded files.")
                    st.session_state.last_pipeline_latency = {}

    metadata_filter = {
        "source": source_filter.strip(),
        "file_type": "" if file_type_filter == "any" else file_type_filter,
        "page": page_filter.strip(),
        "safe_only": safe_only,
    }
    price_table = dict(MODEL_PRICING_DEFAULT)
    for key in price_table:
        if key == model:
            price_table[key] = {"input_per_m": float(price_in), "output_per_m": float(price_out)}

    tab_chat, tab_eval, tab_analytics = st.tabs(["Chat", "Evaluation", "Analytics"])

    with tab_chat:
        st.subheader("Chat")
        if st.session_state.last_pipeline_latency:
            with st.expander("Ingestion and Index Latency"):
                for key, value in st.session_state.last_pipeline_latency.items():
                    st.write(f"{key}: {value:.2f}s")

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
                        selected_model = choose_model_for_query(user_query, routing_mode, model, fast_model, accurate_model)
                        result = engine.answer(
                            user_query,
                            top_k=top_k,
                            candidate_k=candidate_k,
                            modality_filter=modality_filter,
                            metadata_filter=metadata_filter,
                            use_rerank=use_rerank,
                            use_vision=use_vision,
                            enforce_guardrails=enforce_guardrails,
                            answer_model=selected_model,
                            memory=memory,
                        )
                        answer = result["answer"]
                        context_docs = result["context_docs"]
                        latency = result.get("latency", {})
                        usage = result.get("usage", {})
                        model_used = result.get("model_used", selected_model)
                        grounded = bool(result.get("grounded", True))
                        retrieval_scores = result.get("retrieval_scores", [])
                        est_cost = estimate_cost_usd(usage, model_used, price_table)
                        update_model_telemetry(
                            st.session_state.model_telemetry,
                            model_used,
                            float(latency.get("total_s", 0.0)),
                            usage,
                            grounded,
                            est_cost,
                        )
                        record_query_log(
                            st.session_state.query_logs,
                            user_query,
                            model_used,
                            grounded,
                            latency,
                            usage,
                            est_cost,
                            context_docs,
                            retrieval_scores,
                        )
                    except Exception as exc:
                        answer = f"Error while generating answer: {exc}"
                        context_docs = []
                        latency = {}
                        usage = {}
                        model_used = "n/a"
                        grounded = False
                        est_cost = 0.0

                st.markdown(answer)
                st.caption(
                    f"Model: {model_used} | Grounded: {grounded} | "
                    f"Tokens: {usage.get('total_tokens', 0)} | Est. Cost: ${est_cost:.6f}"
                )
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

        render_model_telemetry(st.session_state.model_telemetry)

    with tab_eval:
        st.subheader("Evaluation")
        eval_raw = st.text_area(
            "Eval Set (`question | expected_answer | expected_source` per line)",
            value="What is this document about? | summary | \nWho is mentioned in page 2? | name | report.pdf",
            height=140,
        )
        run_eval = st.button("Run Evaluation", use_container_width=True)
        if run_eval:
            if not engine or not engine.is_ready:
                st.warning("Process files first so evaluation can run.")
            else:
                cases = parse_eval_cases(eval_raw)
                if not cases:
                    st.warning("No valid eval rows found.")
                else:
                    rows = []
                    answer_hits = 0
                    recall_hits = 0
                    recall_total = 0
                    grounded_hits = 0
                    total_latency = 0.0
                    with st.spinner("Running evaluation..."):
                        for case in cases:
                            selected_model = accurate_model if routing_mode == "auto" else model
                            result = engine.answer(
                                case["question"],
                                top_k=top_k,
                                candidate_k=max(candidate_k, top_k),
                                modality_filter=modality_filter,
                                metadata_filter=metadata_filter,
                                use_rerank=use_rerank,
                                use_vision=use_vision,
                                enforce_guardrails=enforce_guardrails,
                                answer_model=selected_model,
                                memory=[],
                            )
                            answer = result.get("answer", "")
                            context_docs = result.get("context_docs", [])
                            grounded = bool(result.get("grounded", False))
                            latency = float(result.get("latency", {}).get("total_s", 0.0))
                            usage = result.get("usage", {})
                            model_used = result.get("model_used", selected_model)
                            retrieval_scores = result.get("retrieval_scores", [])
                            est_cost = estimate_cost_usd(usage, model_used, price_table)
                            update_model_telemetry(
                                st.session_state.model_telemetry,
                                model_used,
                                latency,
                                usage,
                                grounded,
                                est_cost,
                            )
                            record_query_log(
                                st.session_state.query_logs,
                                case["question"],
                                model_used,
                                grounded,
                                result.get("latency", {}),
                                usage,
                                est_cost,
                                context_docs,
                                retrieval_scores,
                            )

                            expected_answer = case["expected_answer"].lower()
                            answer_ok = expected_answer in answer.lower()
                            if answer_ok:
                                answer_hits += 1

                            expected_source = case["expected_source"].lower()
                            recall_ok = None
                            if expected_source:
                                recall_total += 1
                                recall_ok = any(expected_source in str(doc.get("source", "")).lower() for doc in context_docs)
                                if recall_ok:
                                    recall_hits += 1

                            if grounded:
                                grounded_hits += 1
                            total_latency += latency

                            rows.append(
                                {
                                    "question": case["question"],
                                    "answer_match": answer_ok,
                                    "retrieval_recall": recall_ok,
                                    "grounded": grounded,
                                    "latency_s": latency,
                                    "model": model_used,
                                    "answer_preview": answer[:140],
                                }
                            )

                    n = len(cases)
                    answer_rate = answer_hits / n if n else 0.0
                    recall_rate = (recall_hits / recall_total) if recall_total else 0.0
                    grounded_rate = grounded_hits / n if n else 0.0
                    avg_latency = total_latency / n if n else 0.0
                    st.session_state.eval_rows = rows
                    st.session_state.eval_metrics = {
                        "answer_rate": answer_rate,
                        "recall_rate": recall_rate,
                        "grounded_rate": grounded_rate,
                        "avg_latency": avg_latency,
                    }

        if st.session_state.eval_metrics:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Answer Match", f"{float(st.session_state.eval_metrics.get('answer_rate', 0.0)):.1%}")
            c2.metric("Retrieval Recall", f"{float(st.session_state.eval_metrics.get('recall_rate', 0.0)):.1%}")
            c3.metric("Grounding Rate", f"{float(st.session_state.eval_metrics.get('grounded_rate', 0.0)):.1%}")
            c4.metric("Avg Latency", f"{float(st.session_state.eval_metrics.get('avg_latency', 0.0)):.2f}s")
        if st.session_state.eval_rows:
            st.dataframe(st.session_state.eval_rows, use_container_width=True)

    with tab_analytics:
        render_analytics(
            engine,
            st.session_state.query_logs,
            st.session_state.model_telemetry,
            st.session_state.eval_rows,
            st.session_state.eval_metrics,
        )


if __name__ == "__main__":
    main()
