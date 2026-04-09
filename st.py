import os
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

import streamlit as st
from langchain_ollama import OllamaLLM
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
import hashlib
import shutil

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
    if num_pages <= 5 or total_chars < 15000:
        return {
            "temperature": 0.2,
            "chunk_size": 1500,
            "chunk_overlap": 300,
            "k_retrieval": 10,
            "profile": "compact"
        }
    elif num_pages <= 30 or total_chars < 100000:
        return {
            "temperature": 0.25,
            "chunk_size": 1000,
            "chunk_overlap": 200,
            "k_retrieval": 8,
            "profile": "medium"
        }
    elif num_pages <= 100:
        return {
            "temperature": 0.3,
            "chunk_size": 800,
            "chunk_overlap": 150,
            "k_retrieval": 6,
            "profile": "large"
        }
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
        "prompt_template": """Eres un asistente experto. Responde SIEMPRE en español. Responde basándote ÚNICAMENTE en el contexto proporcionado. Sé exhaustivo y completo. Si la pregunta pide listar elementos, personajes, temas o datos, revisa TODO el contexto e incluye TODOS sin omitir ninguno. Si no encuentras la respuesta en el contexto, dilo claramente. Contexto: {context} Pregunta: {question} Respuesta detallada:""",
        "strategy_full": "Contenido completo",
        "strategy_rag": "RAG + MMR",
        "details": "Detalles",
        "chunks_legend": "📄 ≤20 chunks → contenido completo · 🔍 >20 chunks → búsqueda RAG+MMR",
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
        "prompt_template": """You are an expert assistant. ALWAYS answer in English. Answer based ONLY on the provided context. Be thorough and complete. If the question asks to list elements, characters, topics or data, review ALL the context and include ALL of them without omitting any. If you cannot find the answer in the context, say so clearly. Context: {context} Question: {question} Detailed answer:""",
        "strategy_full": "Full content",
        "strategy_rag": "RAG + MMR",
        "details": "Details",
        "chunks_legend": "📄 ≤20 chunks → full content · 🔍 >20 chunks → RAG+MMR search",
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
        "prompt_template": """Ты — экспертный ассистент. ВСЕГДА отвечай на русском языке. Отвечай ТОЛЬКО на основе предоставленного контекста. Будь исчерпывающим и полным. Если вопрос просит перечислить элементы, персонажей, темы или данные, просмотри ВЕСЬ контекст и включи ВСЁ без пропусков. Если ты не можешь найти ответ в контексте, скажи об этом прямо. Контекст: {context} Вопрос: {question} Подробный ответ:""",
        "strategy_full": "Полное содержание",
        "strategy_rag": "RAG + MMR",
        "details": "Детали",
        "chunks_legend": "📄 ≤20 чанков → полное содержание · 🔍 >20 чанков → поиск RAG+MMR",
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
/* Estilos omitidos para brevedad, asumiendo los mismos del usuario */
</style>
""", unsafe_allow_html=True)

# ========== SESSION STATE ==========
if "lang" not in st.session_state: st.session_state.lang = "es"
if "pdf_content" not in st.session_state: st.session_state.pdf_content = None
if "pdf_name" not in st.session_state: st.session_state.pdf_name = None
if "pdf_hash" not in st.session_state: st.session_state.pdf_hash = None
if "messages" not in st.session_state: st.session_state.messages = []
if "modelo" not in st.session_state: st.session_state.modelo = "kiwi_kiwi/gemma-4-uncensores:e4b"

# ========== SIDEBAR ==========
with st.sidebar:
    st.markdown(f"### {t('config')}")
    st.divider()
    if st.button(t("clear_cache"), use_container_width=True):
        clear_cache()
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db", ignore_errors=True)
        st.success(t("cache_cleared"))
    st.divider()
    if st.button(t("new_chat"), use_container_width=True):
        st.session_state.messages = []
        st.session_state.pdf_content = None
        st.session_state.pdf_name = None
        st.session_state.pdf_hash = None
        st.rerun()

# ========== TOP BAR ==========
lang_codes = ["ru", "en", "es"]
current_lang = st.session_state.lang
top_cols = st.columns([4, 1, 1, 1])
with top_cols[0]:
    st.markdown('

 <span class="logo">📄</span> <span class="brand">Agente PDF</span> 

', unsafe_allow_html=True)
for i, code in enumerate(lang_codes):
    with top_cols[i + 1]:
        btn_type = "primary" if current_lang == code else "secondary"
        if st.button(code.upper(), key=f"lang_{code}", type=btn_type, use_container_width=True):
            if st.session_state.lang != code:
                st.session_state.lang = code
                st.rerun()
st.markdown("---")

# ========== GREETING ==========
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
        <div class="pdf-dot pdf-dot-green"></div>
        {t('pdf_loaded')}: <span class="pdf-name">{st.session_state.pdf_name}</span>
    </div>
    """, unsafe_allow_html=True)

pdf_file = st.file_uploader(
    t("attach_pdf"),
    type="pdf",
    label_visibility="collapsed",
    key="pdf_uploader"
)

if pdf_file is not None:
    new_content = pdf_file.read()
    new_hash = hashlib.sha256(new_content).hexdigest()
    
    if new_content and (st.session_state.pdf_hash != new_hash):
        is_valid, msg = validate_pdf(new_content, pdf_file.name)
        if is_valid:
            st.session_state.pdf_content = new_content
            st.session_state.pdf_name = pdf_file.name
            st.session_state.pdf_hash = new_hash
            st.session_state.messages = []
            
            # Limpiar DB vectorial anterior si existe para evitar mezclas
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db", ignore_errors=True)
                
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
            
        total_chars = sum(len(doc.page_content) for doc in documents)
        cfg = auto_config(len(documents), total_chars)
        
        llm = OllamaLLM(
            model=st.session_state.modelo,
            temperature=cfg["temperature"],
            base_url="http://localhost:11434"
        )
        
        embeddings = get_embeddings()
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=cfg["chunk_size"],
            chunk_overlap=cfg["chunk_overlap"],
            separators=["

", "
", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        
        def format_docs(docs):
            return "

".join(doc.page_content for doc in docs)
            
        template = t("prompt_template")
        prompt = PromptTemplate(template=template, input_variables=["context", "question"])
        
        UMBRAL_CHUNKS = 20
        if len(chunks) <= UMBRAL_CHUNKS:
            estrategia = t("strategy_full")
            contexto = format_docs(chunks)
            chain = prompt | llm
            respuesta = chain.invoke({"context": contexto, "question": pregunta})
        else:
            estrategia = t("strategy_rag")
            # Usar carpeta específica por hash para evitar colisiones
            db_dir = f"./chroma_db_{st.session_state.pdf_hash}"
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory=db_dir
            )
            retriever = vectorstore.as_retriever(
                search_type="mmr",
                search_kwargs={"k": cfg["k_retrieval"]}
            )
            
            chain = (
                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                | prompt
                | llm
            )
            respuesta = chain.invoke(pregunta)
            
        os.unlink(tmp_path)
        details = f"**{t('model')}:** `{st.session_state.modelo}`
**Strategy:** {estrategia}
**Chunks:** {len(chunks)}"
        return respuesta, details
    except Exception as e:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        return f"❌ {t('error')}: {str(e)}", None

# ========== CHAT INPUT ==========
if pregunta := st.chat_input(t("chat_placeholder")):
    st.session_state.messages.append({"role": "user", "content": pregunta})
    with st.chat_message("user"):
        st.markdown(pregunta)
    
    with st.chat_message("assistant"):
        with st.spinner(t("processing")):
            respuesta, details = process_question(pregunta)
            st.markdown(respuesta)
            if details:
                with st.expander(f"📊 {t('details')}"):
                    st.markdown(details)
    
    st.session_state.messages.append({"role": "assistant", "content": respuesta, "details": details})
