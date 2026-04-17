import os
import hashlib
import shutil
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import streamlit as st
import requests
import json
from typing import Optional
from groq import Groq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
import traceback
import time
import tempfile


# ========== EMBEDDINGS WRAPPER ==========
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()


@st.cache_resource
def get_embeddings():
    """Carga embeddings una sola vez y las cachea."""
    return SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


from utils import validate_pdf, get_cache_stats, clear_cache


# ========== AUTO CONFIG ==========
def auto_config(num_pages: int, total_chars: int) -> dict:
    """Determina configuración óptima según tamaño del PDF."""
    # PDFs pequeños (1-5 páginas, <15k chars): lectura completa
    if num_pages <= 5 or total_chars < 15000:
        return {
            "temperature": 0.2,
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "k_retrieval": 10,
            "profile": "compact"
        }
    # PDFs medianos (6-30 páginas)
    elif num_pages <= 30 or total_chars < 100000:
        return {
            "temperature": 0.25,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "k_retrieval": 8,
            "profile": "medium"
        }
    # PDFs grandes (31-100 páginas)
    elif num_pages <= 100:
        return {
            "temperature": 0.3,
            "chunk_size": 800,
            "chunk_overlap": 150,
            "k_retrieval": 6,
            "profile": "large"
        }
    # PDFs muy grandes (100+ páginas)
    else:
        return {
            "temperature": 0.3,
            "chunk_size": 600,
            "chunk_overlap": 120,
            "k_retrieval": 5,
            "profile": "xlarge"
        }


# ========== TRADUCCIONES ==========
TRANSLATIONS = {
    "es": {
        "greeting": "Hola, soy Agente PDF.",
        "greeting_sub": "Sube un PDF y hazme preguntas sobre su contenido.",
        "chat_placeholder": "Escribe tu pregunta sobre el PDF...",
        "no_pdf_yet": "Adjunta un PDF para comenzar",
        "pdf_loaded": "PDF cargado",
        "config": "⚙️ Configuración",
        "model": "Modelo",
        "temperature": "Temperatura",
        "chunk_size": "Tamaño chunks",
        "chunks_k": "Chunks a recuperar",
        "clear_cache": "🗑️ Limpiar caché",
        "cache_cleared": "✅ Caché limpiado",
        "processing": "Pensando...",
        "error": "Error",
        "no_text": "No se pudo extraer texto del PDF",
        "prompt_instruction": "Responde en español.",
        "prompt_template": """Eres un asistente experto. Responde SIEMPRE en español.
Responde basándote ÚNICAMENTE en el contexto proporcionado.
Sé exhaustivo y completo. Si la pregunta pide listar elementos, personajes, temas o datos, revisa TODO el contexto e incluye TODOS sin omitir ninguno.
Si no encuentras la respuesta en el contexto, dilo claramente.

Contexto:
{context}

Pregunta: {question}

Respuesta detallada:""",
        "strategy_full": "Contenido completo",
        "strategy_rag": "RAG + MMR",
        "details": "Detalles",
        "chunks_legend": "📄 ≤20 chunks → contenido completo  ·  🔍 >20 chunks → búsqueda RAG+MMR",
        "new_chat": "🔄 Nueva conversación",
        "attach_pdf": "📎",
        "mic_listening": "🔴 Escuchando...",
        "mic_start": "🎤",
        "mic_error": "Tu navegador no soporta reconocimiento de voz",
        "mic_no_result": "No se detectó voz, intenta de nuevo",
    },
    "en": {
        "greeting": "Hi, I'm PDF Agent.",
        "greeting_sub": "Upload a PDF and ask me questions about its content.",
        "chat_placeholder": "Ask a question about the PDF...",
        "no_pdf_yet": "Attach a PDF to get started",
        "pdf_loaded": "PDF loaded",
        "config": "⚙️ Settings",
        "model": "Model",
        "temperature": "Temperature",
        "chunk_size": "Chunk size",
        "chunks_k": "Chunks to retrieve",
        "clear_cache": "🗑️ Clear cache",
        "cache_cleared": "✅ Cache cleared",
        "processing": "Thinking...",
        "error": "Error",
        "no_text": "Could not extract text from PDF",
        "prompt_instruction": "Answer in English.",
        "prompt_template": """You are an expert assistant. ALWAYS answer in English.
Answer based ONLY on the provided context.
Be thorough and complete. If the question asks to list elements, characters, topics or data, review ALL the context and include ALL of them without omitting any.
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Detailed answer:""",
        "strategy_full": "Full content",
        "strategy_rag": "RAG + MMR",
        "details": "Details",
        "chunks_legend": "📄 ≤20 chunks → full content  ·  🔍 >20 chunks → RAG+MMR search",
        "new_chat": "🔄 New chat",
        "attach_pdf": "📎",
        "mic_listening": "🔴 Listening...",
        "mic_start": "🎤",
        "mic_error": "Your browser does not support speech recognition",
        "mic_no_result": "No speech detected, try again",
    },
    "ru": {
        "greeting": "Привет, я PDF Агент.",
        "greeting_sub": "Загрузите PDF и задавайте мне вопросы по его содержанию.",
        "chat_placeholder": "Задайте вопрос о PDF...",
        "no_pdf_yet": "Прикрепите PDF для начала",
        "pdf_loaded": "PDF загружен",
        "config": "⚙️ Настройки",
        "model": "Модель",
        "temperature": "Температура",
        "chunk_size": "Размер чанков",
        "chunks_k": "Чанков к извлечению",
        "clear_cache": "🗑️ Очистить кэш",
        "cache_cleared": "✅ Кэш очищен",
        "processing": "Думаю...",
        "error": "Ошибка",
        "no_text": "Не удалось извлечь текст из PDF",
        "prompt_instruction": "Отвечай на русском языке.",
        "prompt_template": """Ты — экспертный ассистент. ВСЕГДА отвечай на русском языке.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Будь исчерпывающим и полным. Если вопрос просит перечислить элементы, персонажей, темы или данные, просмотри ВЕСЬ контекст и включи ВСЁ без пропусков.
Если ты не можешь найти ответ в контексте, скажи об этом прямо.

Контекст:
{context}

Вопрос: {question}

Подробный ответ:""",
        "strategy_full": "Полное содержание",
        "strategy_rag": "RAG + MMR",
        "details": "Детали",
        "chunks_legend": "📄 ≤20 чанков → полное содержание  ·  🔍 >20 чанков → поиск RAG+MMR",
        "new_chat": "🔄 Новый чат",
        "attach_pdf": "📎",
        "mic_listening": "🔴 Слушаю...",
        "mic_start": "🎤",
        "mic_error": "Ваш браузер не поддерживает распознавание речи",
        "mic_no_result": "Речь не обнаружена, попробуйте снова",
    },
}

def t(key):
    lang = st.session_state.get("lang", "es")
    return TRANSLATIONS.get(lang, TRANSLATIONS["es"]).get(key, key)


# ========== PAGE CONFIG ==========
st.set_page_config(
    page_title="Agente PDF",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ========== CSS DARK MODERNO ==========
st.markdown("""
<style>
    /* ===== FONDO OSCURO ===== */
    .stApp {
        background-color: #1a1a2e;
    }
    .main .block-container {
        max-width: 820px;
        padding-top: 1rem;
        padding-bottom: 0;
    }

    /* Ocultar footer */
    footer { display: none !important; }

    /* ===== GREETING CENTRADO ===== */
    .greeting {
        text-align: center;
        padding: 4rem 0 2rem 0;
    }
    .greeting-icon {
        font-size: 3rem;
        margin-bottom: 0.5rem;
    }
    .greeting h2 {
        color: #e0e0e0;
        font-size: 1.6rem;
        font-weight: 600;
        margin: 0 0 0.4rem 0;
    }
    .greeting p {
        color: #666;
        font-size: 0.95rem;
        margin: 0;
    }

    /* ===== BARRA SUPERIOR ===== */
    .top-bar {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 0.3rem 0;
    }
    .top-bar-left {
        display: flex;
        align-items: center;
        gap: 8px;
    }
    .top-bar-left .logo {
        font-size: 1.3rem;
    }
    .top-bar-left .brand {
        color: #e0e0e0;
        font-weight: 600;
        font-size: 1rem;
    }
    /* ===== LANG PILL BUTTONS ===== */
    /* Target the top columns row containing brand + lang buttons */
    .main .block-container > div:first-child .stButton > button,
    div[data-testid="stColumns"] .stButton > button {
        padding: 2px 14px !important;
        border-radius: 16px !important;
        font-size: 12px !important;
        font-weight: 700 !important;
        min-height: 0 !important;
        height: 30px !important;
        line-height: 1 !important;
    }
    div[data-testid="stColumns"] .stButton > button[kind="primary"] {
        background: #4a6cf7 !important;
        color: #fff !important;
        border: none !important;
    }
    div[data-testid="stColumns"] .stButton > button[kind="secondary"] {
        background: #2a2a3e !important;
        color: #888 !important;
        border: 1px solid #333 !important;
    }

    /* ===== PDF CHIP ===== */
    .pdf-chip {
        display: inline-flex;
        align-items: center;
        background: #22223a;
        color: #888;
        padding: 5px 14px;
        border-radius: 18px;
        font-size: 0.82rem;
        border: 1px solid #2e2e4a;
    }
    .pdf-dot {
        width: 7px; height: 7px;
        border-radius: 50%;
        margin-right: 8px;
    }
    .pdf-dot-green { background: #4CAF50; }
    .pdf-dot-gray { background: #555; }
    .pdf-name { color: #bbb; font-weight: 500; margin-left: 4px; }

    /* ===== INJECTED MODEL SELECT IN CHAT BAR ===== */
    .injected-model-select {
        background: #2a2a3e;
        color: #aaa;
        border: 1px solid #3a3a5a;
        border-radius: 14px;
        padding: 3px 10px;
        font-size: 0.75rem;
        font-weight: 600;
        cursor: pointer;
        outline: none;
        appearance: none;
        -webkit-appearance: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='10' height='6'%3E%3Cpath d='M0 0l5 6 5-6z' fill='%23888'/%3E%3C/svg%3E");
        background-repeat: no-repeat;
        background-position: right 8px center;
        padding-right: 24px;
        height: 28px;
        flex-shrink: 0;
        transition: all 0.15s;
    }
    .injected-model-select:hover {
        border-color: #4a6cf7;
        color: #ccc;
    }
    .injected-model-select:focus {
        border-color: #4a6cf7;
        box-shadow: 0 0 0 2px rgba(74,108,247,0.2);
    }
    .injected-model-select option {
        background: #1a1a2e;
        color: #ccc;
    }

    /* ===== CHAT MESSAGES ===== */
    div[data-testid="stChatMessage"] {
        background-color: #22223a !important;
        border: 1px solid #2e2e4a;
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }

    /* ===== SIDEBAR ===== */
    section[data-testid="stSidebar"] {
        background-color: #16162a;
    }
    section[data-testid="stSidebar"] * {
        color: #ccc;
    }

    /* ===== FILE UPLOADER COMPACT ===== */
    div[data-testid="stFileUploader"] {
        background: #22223a;
        border-radius: 12px;
        border: 1px dashed #3a3a5a;
        padding: 0.5rem;
    }
    div[data-testid="stFileUploader"] label {
        color: #888 !important;
        font-size: 0.85rem;
    }

    /* ===== CHAT INPUT ===== */
    div[data-testid="stChatInput"] {
        background-color: #22223a !important;
        border: 1px solid #3a3a5a !important;
        border-radius: 16px !important;
        padding: 8px 12px 6px 12px !important;
    }
    div[data-testid="stChatInput"] textarea {
        background-color: transparent !important;
        border: none !important;
        color: #e0e0e0 !important;
        border-radius: 0 !important;
        padding: 4px 0 !important;
    }
    /* ===== MIC INSIDE CHAT BAR ===== */
    .injected-mic {
        background: transparent;
        border: none;
        font-size: 1.15rem;
        cursor: pointer;
        padding: 4px 6px;
        margin-right: 2px;
        border-radius: 50%;
        transition: all 0.2s;
        color: #888;
        display: flex;
        align-items: center;
        justify-content: center;
        width: 34px;
        height: 34px;
        flex-shrink: 0;
    }
    .injected-mic:hover {
        background: rgba(74,108,247,0.15);
        color: #4a6cf7;
    }
    .injected-mic.recording {
        color: #e74c3c;
        animation: mic-pulse 1s infinite;
    }
    @keyframes mic-pulse {
        0%, 100% { box-shadow: 0 0 0 0 rgba(231,76,60,0.4); }
        50% { box-shadow: 0 0 0 8px rgba(231,76,60,0); }
    }
</style>
""", unsafe_allow_html=True)


# ========== SESSION STATE ==========
if "lang" not in st.session_state:
    st.session_state.lang = "es"
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "pdf_hash" not in st.session_state:
    st.session_state.pdf_hash = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = None
if "modelo" not in st.session_state:
    st.session_state.modelo = "kiwi_kiwi/gemma-4-uncensores:e4b"


# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown(f"### {t('config')}")
    st.divider()
    if st.button(t("clear_cache"), use_container_width=True):
        clear_cache()
        st.success(t("cache_cleared"))
    st.divider()
    if st.button(t("new_chat"), use_container_width=True):
        old_hash = st.session_state.pdf_hash

        st.session_state.messages = []
        st.session_state.pdf_content = None
        st.session_state.pdf_name = None
        st.session_state.pdf_hash = None

        if old_hash:
            old_db = f"./chroma_db/{old_hash}"
            if os.path.exists(old_db):
                shutil.rmtree(old_db, ignore_errors=True)

        st.rerun()


# ========== TOP BAR ==========
lang_codes = ["ru", "en", "es"]
current_lang = st.session_state.lang

# Brand + Language pills en una sola fila
top_cols = st.columns([4, 1, 1, 1])
with top_cols[0]:
    st.markdown('<div class="top-bar-left"><span class="logo">📄</span><span class="brand">Agente PDF</span></div>', unsafe_allow_html=True)

for i, code in enumerate(lang_codes):
    with top_cols[i + 1]:
        btn_type = "primary" if current_lang == code else "secondary"
        if st.button(code.upper(), key=f"lang_{code}", type=btn_type, use_container_width=True):
            if st.session_state.lang != code:
                st.session_state.lang = code
                st.rerun()

st.markdown("---")


# ========== GREETING (solo si no hay mensajes) ==========
if not st.session_state.messages:
    st.markdown(f"""
    <div class="greeting">
        <div class="greeting-icon">📄</div>
        <h2>{t('greeting')}</h2>
        <p>{t('greeting_sub')}</p>
    </div>
    """, unsafe_allow_html=True)


# ========== PDF CHIP + UPLOAD ==========
if st.session_state.pdf_name:
    st.markdown(f"""
    <div class="pdf-chip">
        <span class="pdf-dot pdf-dot-green"></span>
        {t('pdf_loaded')}: <span class="pdf-name">{st.session_state.pdf_name}</span>
    </div>
    """, unsafe_allow_html=True)
else:
    st.markdown(f"""
    <div class="pdf-chip">
        <span class="pdf-dot pdf-dot-gray"></span>
        {t('no_pdf_yet')}
    </div>
    """, unsafe_allow_html=True)

pdf_file = st.file_uploader(
    t("attach_pdf"),
    type="pdf",
    label_visibility="collapsed",
    key="pdf_uploader"
)

if pdf_file is not None:
    new_content = pdf_file.getvalue()

    if new_content:
        new_hash = hashlib.sha256(new_content).hexdigest()

        if st.session_state.pdf_hash != new_hash:
            is_valid, msg = validate_pdf(new_content, pdf_file.name)
            if is_valid:
                old_hash = st.session_state.pdf_hash

                st.session_state.pdf_content = new_content
                st.session_state.pdf_name = pdf_file.name
                st.session_state.pdf_hash = new_hash
                st.session_state.messages = []

                if old_hash:
                    old_db = f"./chroma_db/{old_hash}"
                    if os.path.exists(old_db):
                        shutil.rmtree(old_db, ignore_errors=True)

                st.rerun()
            else:
                st.error(f"❌ {msg}")


# ========== HISTORIAL DE CHAT ==========
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "details" in msg:
            with st.expander(f"📊 {t('details')}"):
                st.markdown(msg["details"])





# ========== FUNCIÓN RAG ==========
def process_question(pregunta: str):
    """Procesa la pregunta contra el PDF cargado."""
    if st.session_state.pdf_content is None:
        return f"⚠️ {t('no_pdf_yet')}", None

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(st.session_state.pdf_content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        documents = loader.load()

        if not documents:
            os.unlink(tmp_path)
            return f"❌ {t('no_text')}", None

        # Auto-detect optimal config
        total_chars = sum(len(doc.page_content) for doc in documents)
        cfg = auto_config(len(documents), total_chars)

        # Llamadas a GROQ en lugar de Ollama. Se espera que la API key
        # se configure como variable de entorno `GROQ_API_KEY` en Render.
        def call_groq_model(prompt_text: str, model: Optional[str] = None, temperature: float = 0.2) -> str:
            api_key = os.environ.get("GROQ_API_KEY")
            if not api_key:
                raise RuntimeError("GROQ_API_KEY no está establecida en el entorno")
            if model is None:
                model = os.environ.get("GROQ_MODEL", st.session_state.modelo)

            # Instanciar cliente Groq (usa api_key desde entorno)
            client = Groq(api_key=api_key)

            messages = [{"role": "user", "content": prompt_text}]

            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_completion_tokens=1024,
                top_p=1,
                stream=True,
                stop=None,
            )

            # Acumular contenido del stream
            output = ""
            for chunk in completion:
                try:
                    # Intentar ruta tipo delta (stream)
                    delta = chunk.choices[0].delta
                    if delta is not None:
                        # delta puede ser objeto o dict
                        if hasattr(delta, "content"):
                            output += delta.content or ""
                        elif isinstance(delta, dict):
                            output += delta.get("content", "") or ""
                        continue

                    # Si no hay delta, intentar message completo
                    msg = chunk.choices[0].message
                    if msg is not None:
                        if hasattr(msg, "content"):
                            output += msg.content or ""
                        elif isinstance(msg, dict):
                            output += msg.get("content", "") or ""
                except Exception:
                    # Fallback for dict-shaped chunk
                    try:
                        c = (chunk.get("choices", [{}])[0].get("delta", {}) or {}).get("content", "")
                        output += c or ""
                    except Exception:
                        pass

            return output

        embeddings = get_embeddings()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        template = t("prompt_template")

        prompt = PromptTemplate(template=template, input_variables=["context", "question"])

        UMBRAL_CHUNKS = 20
        estrategia = ""

        if len(chunks) <= UMBRAL_CHUNKS:
            estrategia = t("strategy_full")
            todo = format_docs(chunks)
            # Construir prompt de texto y llamar a GROQ
            prompt_text = prompt.template.format(context=todo, question=pregunta)
            respuesta = None
            for intento in range(3):
                try:
                    respuesta = call_groq_model(prompt_text, model=os.environ.get("GROQ_MODEL"), temperature=cfg["temperature"])
                    break
                except Exception:
                    if intento < 2:
                        time.sleep(2)
                    else:
                        raise
        else:
            estrategia = t("strategy_rag")
            db_dir = f"./chroma_db/{st.session_state.pdf_hash}"

            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_dir
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={
                    "k": cfg["k_retrieval"],
                    "fetch_k": min(cfg["k_retrieval"] * 3, len(chunks)),
                    "lambda_mult": 0.5
                }
            )
            # Recuperar documentos relevantes y construir prompt
            try:
                # Algunas implementaciones exponen `get_relevant_documents`
                relevant = retriever.get_relevant_documents(pregunta)
            except Exception:
                try:
                    relevant = retriever(pregunta)
                except Exception:
                    relevant = []

            contexto = format_docs(relevant)
            prompt_text = prompt.template.format(context=contexto, question=pregunta)
            respuesta = None
            for intento in range(3):
                try:
                    respuesta = call_groq_model(prompt_text, model=os.environ.get("GROQ_MODEL"), temperature=cfg["temperature"])
                    break
                except Exception:
                    if intento < 2:
                        time.sleep(2)
                    else:
                        raise

        os.unlink(tmp_path)

        details = (
            f"**{t('model')}:** `{st.session_state.modelo}`  \n"
            f"**Config:** {cfg['profile']} · T={cfg['temperature']} · chunk={cfg['chunk_size']}  \n"
            f"**Strategy:** {estrategia}  \n"
            f"**Chunks:** {len(chunks)} · **Pages:** {len(documents)} · **Chars:** {total_chars:,}"
        )

        return respuesta, details

    except Exception as e:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass
        return f"❌ {t('error')}: {str(e)}", None


# ========== MIC INJECTED INTO CHAT BAR (Web Speech API) ==========
lang_map = {"es": "es-ES", "en": "en-US", "ru": "ru-RU"}
mic_lang = lang_map.get(st.session_state.lang, "es-ES")

import urllib.parse

# Helper: render HTML/JS via data URL inside an iframe (replaces deprecated st.components.v1.html)
def render_html_via_iframe(html: str, height: int = 0):
    data_url = "data:text/html;charset=utf-8," + urllib.parse.quote(html)
    st.iframe(data_url, height=height)

mic_inject_js = f"""
<script>
(function() {{
    var CHECK_INTERVAL = 300;
    var MAX_ATTEMPTS = 30;
    var attempts = 0;

    function injectMic() {{
        var doc = window.parent.document;
        // Already injected?
        if (doc.getElementById('injectedMicBtn')) return;

        // Find the chat input container
        var chatInput = doc.querySelector('div[data-testid="stChatInput"]');
        if (!chatInput) {{
            attempts++;
            if (attempts < MAX_ATTEMPTS) setTimeout(injectMic, CHECK_INTERVAL);
            return;
        }}

        // Find the send button
        var sendBtn = chatInput.querySelector('button[data-testid="stChatInputSubmitButton"]');
        if (!sendBtn) {{
            // Fallback: find any button inside
            sendBtn = chatInput.querySelector('button');
        }}
        if (!sendBtn) {{
            attempts++;
            if (attempts < MAX_ATTEMPTS) setTimeout(injectMic, CHECK_INTERVAL);
            return;
        }}

        // Create mic button
        var micBtn = doc.createElement('button');
        micBtn.id = 'injectedMicBtn';
        micBtn.className = 'injected-mic';
        micBtn.type = 'button';
        micBtn.innerHTML = '🎤';
        micBtn.title = 'Voice input';

        // Insert before the send button
        sendBtn.parentNode.insertBefore(micBtn, sendBtn);

        // Speech Recognition logic
        var recognition = null;
        var isRecording = false;

        micBtn.addEventListener('click', function(e) {{
            e.preventDefault();
            e.stopPropagation();

            var W = window.parent;
            if (!('webkitSpeechRecognition' in W) && !('SpeechRecognition' in W)) {{
                alert("{t('mic_error')}");
                return;
            }}

            if (isRecording && recognition) {{
                recognition.stop();
                return;
            }}

            var SR = W.SpeechRecognition || W.webkitSpeechRecognition;
            recognition = new SR();
            recognition.lang = "{mic_lang}";
            recognition.interimResults = false;
            recognition.maxAlternatives = 1;
            recognition.continuous = false;

            recognition.onstart = function() {{
                isRecording = true;
                micBtn.classList.add('recording');
                micBtn.innerHTML = '🔴';
            }};

            recognition.onresult = function(event) {{
                var transcript = event.results[0][0].transcript;
                var textarea = doc.querySelector('textarea[data-testid="stChatInputTextArea"]');
                if (textarea) {{
                    var nativeSetter = Object.getOwnPropertyDescriptor(W.HTMLTextAreaElement.prototype, 'value').set;
                    nativeSetter.call(textarea, transcript);
                    textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
                    // Focus so user can just press Enter or click send
                    textarea.focus();
                }}
            }};

            recognition.onend = function() {{
                isRecording = false;
                micBtn.classList.remove('recording');
                micBtn.innerHTML = '🎤';
            }};

            recognition.onerror = function() {{
                isRecording = false;
                micBtn.classList.remove('recording');
                micBtn.innerHTML = '🎤';
            }};

            recognition.start();
        }});
    }}

    // Start trying to inject
    if (document.readyState === 'complete') {{
        setTimeout(injectMic, 500);
    }} else {{
        window.addEventListener('load', function() {{ setTimeout(injectMic, 500); }});
    }}
}})();
</script>
"""
render_html_via_iframe(mic_inject_js, height=0)

# ========== MODEL SELECT (injected into chat bar via JS) ==========
MODELS = {
    "kiwi_kiwi/gemma-4-uncensores:e4b": "🧠 Gemma 4"
}

# Sync from query params if user changed model via injected select
qp = st.query_params
if "model" in qp:
    new_model = qp["model"]
    if new_model in MODELS and new_model != st.session_state.modelo:
        st.session_state.modelo = new_model

modelo = st.session_state.modelo

# Build options HTML for injection
options_html = ""
for mid, mlabel in MODELS.items():
    selected = "selected" if mid == modelo else ""
    options_html += f'<option value="{mid}" {selected}>{mlabel}</option>'

model_inject_js = f"""
<script>
(function() {{
    var CHECK_INTERVAL = 300;
    var MAX_ATTEMPTS = 30;
    var attempts = 0;

    function injectModelSelect() {{
        var doc = window.parent.document;
        if (doc.getElementById('injectedModelSelect')) return;

        var chatInput = doc.querySelector('div[data-testid="stChatInput"]');
        if (!chatInput) {{
            attempts++;
            if (attempts < MAX_ATTEMPTS) setTimeout(injectModelSelect, CHECK_INTERVAL);
            return;
        }}

        var textarea = chatInput.querySelector('textarea');
        if (!textarea) return;

        // Create select element
        var sel = doc.createElement('select');
        sel.id = 'injectedModelSelect';
        sel.className = 'injected-model-select';
        sel.innerHTML = `{options_html}`;

        sel.addEventListener('change', function() {{
            var url = new URL(window.parent.location.href);
            url.searchParams.set('model', sel.value);
            window.parent.location.href = url.toString();
        }});

        // Insert before the send button (next to mic)
        var sendBtn = chatInput.querySelector('button[data-testid="stChatInputSubmitButton"]');
        if (!sendBtn) sendBtn = chatInput.querySelector('button');

        if (sendBtn) {{
            // Find the mic button if exists and insert before it
            var micBtn = doc.getElementById('injectedMicBtn');
            if (micBtn) {{
                micBtn.parentNode.insertBefore(sel, micBtn);
            }} else {{
                sendBtn.parentNode.insertBefore(sel, sendBtn);
            }}
        }}
    }}

    if (document.readyState === 'complete') {{
        setTimeout(injectModelSelect, 700);
    }} else {{
        window.addEventListener('load', function() {{ setTimeout(injectModelSelect, 700); }});
    }}
}})();
</script>
"""
render_html_via_iframe(model_inject_js, height=0)

# ========== CHAT INPUT ==========
if pregunta := st.chat_input(t("chat_placeholder")):
    # Guardar y mostrar mensaje del usuario
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)

    # Generar respuesta
    with st.chat_message("assistant"):
        with st.spinner(t("processing")):
            respuesta, details = process_question(pregunta)
            st.markdown(respuesta)
            if details:
                with st.expander(f"📊 {t('details')}"):
                    st.markdown(details)

    # Guardar en historial
    msg_data = {"role": "assistant", "content": respuesta}
    if details:
        msg_data["details"] = details
    st.session_state.messages.append(msg_data)
