import os
# Evitar que httpx (usado por ollama/langchain-ollama) enrute a través del proxy del sistema
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')

import streamlit as st
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
import traceback
import time


# Wrapper de embeddings usando sentence-transformers
class SentenceTransformerEmbeddings(Embeddings):
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts, convert_to_numpy=True).tolist()
    
    def embed_query(self, text):
        return self.model.encode(text, convert_to_numpy=True).tolist()

from utils import (
    validate_pdf,
    get_cache_stats,
    clear_cache
)

# Configuración página
st.set_page_config(
    page_title="Agente PDF",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ========== SISTEMA DE IDIOMAS ==========
TRANSLATIONS = {
    "es": {
        "page_title": "Agente PDF",
        "title": "📄 Agente PDF - Demo",
        "subtitle": "*Sube un PDF y haz preguntas sobre su contenido*",
        "config": "⚙️ Configuración",
        "language": "🌐 Idioma:",
        "model": "🤖 Modelo Ollama:",
        "model_help": "Selecciona el modelo disponible en tu Ollama",
        "temperature": "🌡️ Temperatura:",
        "temp_help": "Menor = más determinístico, Mayor = más creativo",
        "chunk_size": "📏 Tamaño de chunks:",
        "chunks_retrieve": "🔍 Chunks a recuperar:",
        "clear_cache": "🗑️ Limpiar caché",
        "cache_cleared": "✅ Caché limpiado",
        "cache_title": "📊 Caché",
        "pdfs_saved": "PDFs guardados",
        "space_used": "Espacio usado",
        "cache_hits": "Hits de caché",
        "tab_upload": "📥 Carga PDF",
        "tab_questions": "❓ Preguntas",
        "tab_info": "ℹ️ Info",
        "upload_title": "Sube tu PDF",
        "select_pdf": "Selecciona un PDF:",
        "max_size": "Máximo tamaño: 50MB",
        "load_btn": "📄 Cargar",
        "no_pdf": "❌ Selecciona un PDF primero",
        "pdf_loaded": "✅ <b>PDF cargado correctamente</b>",
        "file_label": "Archivo",
        "size_label": "Tamaño",
        "load_error": "❌ Error al cargar",
        "tech_details": "📋 Detalles técnicos",
        "current_pdf": "📚 PDF actual",
        "ask_title": "Haz tu pregunta",
        "upload_first": "⚠️ Primero sube un PDF en la pestaña '📥 Carga PDF'",
        "working_with": "📄 Trabajando con",
        "your_question": "Tu pregunta:",
        "question_placeholder": "Ej: ¿Cuál es el tema principal?",
        "question_help": "Haz una pregunta sobre el contenido del PDF",
        "search_btn": "🔍 Buscar",
        "processing": "⏳ Procesando pregunta...",
        "loading_pdf": "📖 Cargando PDF...",
        "no_text": "❌ No se pudo extraer texto del PDF",
        "init_model": "🤖 Inicializando modelo...",
        "gen_embeddings": "🔢 Generando embeddings...",
        "splitting": "✂️ Dividiendo texto...",
        "chunks_created": "chunks creados de",
        "pages": "páginas",
        "small_doc": "📄 Documento pequeño",
        "full_content": "usando contenido completo",
        "creating_index": "🗺️ Creando índice vectorial...",
        "building_chain": "🔗 Construyendo cadena RAG...",
        "generating": "💭 Generando respuesta...",
        "retrying": "⚠️ Reintentando...",
        "no_response": "❌ No se pudo obtener respuesta",
        "response": "✅ Respuesta:",
        "search_details": "🔍 Detalles de la búsqueda",
        "model_label": "Modelo",
        "temp_label": "Temperatura",
        "chunks_retrieved": "Chunks recuperados",
        "total_chunks": "Total de chunks",
        "total_pages": "Total de páginas",
        "strategy_label": "Estrategia",
        "strategy_full": "Contenido completo (doc pequeño)",
        "strategy_rag": "RAG con búsqueda MMR",
        "error": "❌ Error",
        "write_question": "⚠️ Escribe una pregunta",
        "info_title": "ℹ️ Información",
        "how_works": "### 📚 ¿Cómo funciona?",
        "how_steps": """1. **Carga**: Sube un PDF desde tu computadora
2. **Procesamiento**: Se divide en chunks y se generan embeddings
3. **Búsqueda**: Tu pregunta busca los chunks más relevantes
4. **Respuesta**: El LLM responde basándose en esos chunks""",
        "models_title": "### 🔧 Modelos disponibles",
        "models_list": """- **Gemma3** (recomendado): Rápido y ligero (3.3 GB)
- **Llama3-Groq-Tool-Use**: Excelente para reasoning (4.7 GB)
- **DeepSeek R1**: Alto rendimiento (5.2 GB)
- **Qwen2.5-Coder**: Especializado en código (4.7 GB)""",
        "tips_title": "### ⚡ Tips",
        "tips_list": """- Sé específico en tus preguntas
- Usa palabras clave del documento
- Aumenta temperatura para respuestas creativas
- Reduce temperatura para respuestas más precisas""",
        "reqs_title": "### 🚀 Requisitos",
        "reqs_list": """- Ollama ejecutándose en localhost:11434
- Modelos disponibles en Ollama (usa `ollama list` para verificar)""",
        "stats_title": "### 📊 Estadísticas",
        "chunks_legend_title": "### 🧩 Lógica de Chunks",
        "chunks_legend": """El sistema usa una **estrategia adaptativa** según el tamaño del documento:

| Tamaño | Estrategia | Descripción |
|--------|-----------|-------------|
| **≤ 20 chunks** | 📄 Contenido completo | Se envía TODO el texto al modelo. Ideal para PDFs cortos — no se pierde información. |
| **> 20 chunks** | 🔍 RAG + MMR | Se buscan los chunks más relevantes y diversos usando Maximum Marginal Relevance. Ideal para PDFs largos. |

**¿Qué es un chunk?** Un fragmento del texto del PDF (por defecto ~1000 caracteres). El documento se divide en chunks para procesamiento.

**¿Qué es MMR?** Maximum Marginal Relevance — un algoritmo que balancea **relevancia** con **diversidad**, evitando traer chunks repetitivos.""",
        "prompt_instruction": "Responde en español.",
        "pdfs_cached": "PDFs cacheados",
        "space": "Espacio",
        "hits": "Hits",
    },
    "en": {
        "page_title": "PDF Agent",
        "title": "📄 PDF Agent - Demo",
        "subtitle": "*Upload a PDF and ask questions about its content*",
        "config": "⚙️ Settings",
        "language": "🌐 Language:",
        "model": "🤖 Ollama Model:",
        "model_help": "Select the model available in your Ollama",
        "temperature": "🌡️ Temperature:",
        "temp_help": "Lower = more deterministic, Higher = more creative",
        "chunk_size": "📏 Chunk size:",
        "chunks_retrieve": "🔍 Chunks to retrieve:",
        "clear_cache": "🗑️ Clear cache",
        "cache_cleared": "✅ Cache cleared",
        "cache_title": "📊 Cache",
        "pdfs_saved": "Saved PDFs",
        "space_used": "Space used",
        "cache_hits": "Cache hits",
        "tab_upload": "📥 Upload PDF",
        "tab_questions": "❓ Questions",
        "tab_info": "ℹ️ Info",
        "upload_title": "Upload your PDF",
        "select_pdf": "Select a PDF:",
        "max_size": "Max size: 50MB",
        "load_btn": "📄 Load",
        "no_pdf": "❌ Select a PDF first",
        "pdf_loaded": "✅ <b>PDF loaded successfully</b>",
        "file_label": "File",
        "size_label": "Size",
        "load_error": "❌ Error loading",
        "tech_details": "📋 Technical details",
        "current_pdf": "📚 Current PDF",
        "ask_title": "Ask your question",
        "upload_first": "⚠️ First upload a PDF in the '📥 Upload PDF' tab",
        "working_with": "📄 Working with",
        "your_question": "Your question:",
        "question_placeholder": "Ex: What is the main topic?",
        "question_help": "Ask a question about the PDF content",
        "search_btn": "🔍 Search",
        "processing": "⏳ Processing question...",
        "loading_pdf": "📖 Loading PDF...",
        "no_text": "❌ Could not extract text from PDF",
        "init_model": "🤖 Initializing model...",
        "gen_embeddings": "🔢 Generating embeddings...",
        "splitting": "✂️ Splitting text...",
        "chunks_created": "chunks created from",
        "pages": "pages",
        "small_doc": "📄 Small document",
        "full_content": "using full content",
        "creating_index": "🗺️ Creating vector index...",
        "building_chain": "🔗 Building RAG chain...",
        "generating": "💭 Generating response...",
        "retrying": "⚠️ Retrying...",
        "no_response": "❌ Could not get a response",
        "response": "✅ Response:",
        "search_details": "🔍 Search details",
        "model_label": "Model",
        "temp_label": "Temperature",
        "chunks_retrieved": "Chunks retrieved",
        "total_chunks": "Total chunks",
        "total_pages": "Total pages",
        "strategy_label": "Strategy",
        "strategy_full": "Full content (small doc)",
        "strategy_rag": "RAG with MMR search",
        "error": "❌ Error",
        "write_question": "⚠️ Write a question",
        "info_title": "ℹ️ Information",
        "how_works": "### 📚 How does it work?",
        "how_steps": """1. **Upload**: Upload a PDF from your computer
2. **Processing**: It's split into chunks and embeddings are generated
3. **Search**: Your question searches for the most relevant chunks
4. **Answer**: The LLM answers based on those chunks""",
        "models_title": "### 🔧 Available models",
        "models_list": """- **Gemma3** (recommended): Fast and light (3.3 GB)
- **Llama3-Groq-Tool-Use**: Excellent for reasoning (4.7 GB)
- **DeepSeek R1**: High performance (5.2 GB)
- **Qwen2.5-Coder**: Code specialist (4.7 GB)""",
        "tips_title": "### ⚡ Tips",
        "tips_list": """- Be specific in your questions
- Use keywords from the document
- Increase temperature for creative answers
- Decrease temperature for precise answers""",
        "reqs_title": "### 🚀 Requirements",
        "reqs_list": """- Ollama running on localhost:11434
- Models available in Ollama (use `ollama list` to check)""",
        "stats_title": "### 📊 Statistics",
        "chunks_legend_title": "### 🧩 Chunk Logic",
        "chunks_legend": """The system uses an **adaptive strategy** based on document size:

| Size | Strategy | Description |
|------|----------|-------------|
| **≤ 20 chunks** | 📄 Full content | ALL text is sent to the model. Ideal for short PDFs — no information is lost. |
| **> 20 chunks** | 🔍 RAG + MMR | The most relevant and diverse chunks are searched using Maximum Marginal Relevance. Ideal for long PDFs. |

**What is a chunk?** A fragment of the PDF text (default ~1000 chars). The document is split into chunks for processing.

**What is MMR?** Maximum Marginal Relevance — an algorithm that balances **relevance** with **diversity**, avoiding repetitive chunks.""",
        "prompt_instruction": "Answer in English.",
        "pdfs_cached": "Cached PDFs",
        "space": "Space",
        "hits": "Hits",
    },
    "ru": {
        "page_title": "PDF Агент",
        "title": "📄 PDF Агент - Демо",
        "subtitle": "*Загрузите PDF и задавайте вопросы по его содержанию*",
        "config": "⚙️ Настройки",
        "language": "🌐 Язык:",
        "model": "🤖 Модель Ollama:",
        "model_help": "Выберите модель, доступную в вашем Ollama",
        "temperature": "🌡️ Температура:",
        "temp_help": "Ниже = более детерминированный, Выше = более креативный",
        "chunk_size": "📏 Размер чанков:",
        "chunks_retrieve": "🔍 Чанков для извлечения:",
        "clear_cache": "🗑️ Очистить кэш",
        "cache_cleared": "✅ Кэш очищен",
        "cache_title": "📊 Кэш",
        "pdfs_saved": "Сохранённые PDF",
        "space_used": "Занято места",
        "cache_hits": "Попаданий кэша",
        "tab_upload": "📥 Загрузка PDF",
        "tab_questions": "❓ Вопросы",
        "tab_info": "ℹ️ Инфо",
        "upload_title": "Загрузите ваш PDF",
        "select_pdf": "Выберите PDF:",
        "max_size": "Максимальный размер: 50МБ",
        "load_btn": "📄 Загрузить",
        "no_pdf": "❌ Сначала выберите PDF",
        "pdf_loaded": "✅ <b>PDF успешно загружен</b>",
        "file_label": "Файл",
        "size_label": "Размер",
        "load_error": "❌ Ошибка загрузки",
        "tech_details": "📋 Технические детали",
        "current_pdf": "📚 Текущий PDF",
        "ask_title": "Задайте ваш вопрос",
        "upload_first": "⚠️ Сначала загрузите PDF во вкладке '📥 Загрузка PDF'",
        "working_with": "📄 Работаем с",
        "your_question": "Ваш вопрос:",
        "question_placeholder": "Пр: Какова основная тема?",
        "question_help": "Задайте вопрос о содержании PDF",
        "search_btn": "🔍 Поиск",
        "processing": "⏳ Обработка вопроса...",
        "loading_pdf": "📖 Загрузка PDF...",
        "no_text": "❌ Не удалось извлечь текст из PDF",
        "init_model": "🤖 Инициализация модели...",
        "gen_embeddings": "🔢 Генерация эмбеддингов...",
        "splitting": "✂️ Разделение текста...",
        "chunks_created": "чанков создано из",
        "pages": "страниц",
        "small_doc": "📄 Небольшой документ",
        "full_content": "используется полное содержание",
        "creating_index": "🗺️ Создание векторного индекса...",
        "building_chain": "🔗 Построение RAG-цепочки...",
        "generating": "💭 Генерация ответа...",
        "retrying": "⚠️ Повтор попытки...",
        "no_response": "❌ Не удалось получить ответ",
        "response": "✅ Ответ:",
        "search_details": "🔍 Детали поиска",
        "model_label": "Модель",
        "temp_label": "Температура",
        "chunks_retrieved": "Извлечено чанков",
        "total_chunks": "Всего чанков",
        "total_pages": "Всего страниц",
        "strategy_label": "Стратегия",
        "strategy_full": "Полное содержание (малый док.)",
        "strategy_rag": "RAG с поиском MMR",
        "error": "❌ Ошибка",
        "write_question": "⚠️ Напишите вопрос",
        "info_title": "ℹ️ Информация",
        "how_works": "### 📚 Как это работает?",
        "how_steps": """1. **Загрузка**: Загрузите PDF с вашего компьютера
2. **Обработка**: Текст разделяется на чанки и генерируются эмбеддинги
3. **Поиск**: Ваш вопрос ищет наиболее релевантные чанки
4. **Ответ**: LLM отвечает на основе этих чанков""",
        "models_title": "### 🔧 Доступные модели",
        "models_list": """- **Gemma3** (рекомендуется): Быстрая и лёгкая (3.3 ГБ)
- **Llama3-Groq-Tool-Use**: Отличная для рассуждений (4.7 ГБ)
- **DeepSeek R1**: Высокая производительность (5.2 ГБ)
- **Qwen2.5-Coder**: Специалист по коду (4.7 ГБ)""",
        "tips_title": "### ⚡ Советы",
        "tips_list": """- Будьте конкретны в вопросах
- Используйте ключевые слова из документа
- Увеличьте температуру для креативных ответов
- Уменьшите температуру для точных ответов""",
        "reqs_title": "### 🚀 Требования",
        "reqs_list": """- Ollama работает на localhost:11434
- Модели доступны в Ollama (используйте `ollama list` для проверки)""",
        "stats_title": "### 📊 Статистика",
        "chunks_legend_title": "### 🧩 Логика чанков",
        "chunks_legend": """Система использует **адаптивную стратегию** в зависимости от размера документа:

| Размер | Стратегия | Описание |
|--------|-----------|----------|
| **≤ 20 чанков** | 📄 Полное содержание | ВЕСЬ текст отправляется модели. Идеально для коротких PDF — информация не теряется. |
| **> 20 чанков** | 🔍 RAG + MMR | Ищутся наиболее релевантные и разнообразные чанки с помощью Maximum Marginal Relevance. Идеально для длинных PDF. |

**Что такое чанк?** Фрагмент текста PDF (по умолчанию ~1000 символов). Документ разделяется на чанки для обработки.

**Что такое MMR?** Maximum Marginal Relevance — алгоритм, балансирующий **релевантность** и **разнообразие**, избегая повторяющихся чанков.""",
        "prompt_instruction": "Отвечай на русском языке.",
        "pdfs_cached": "PDF в кэше",
        "space": "Место",
        "hits": "Попаданий",
    },
}

# Función helper para traducciones
def t(key):
    lang = st.session_state.get("lang", "es")
    return TRANSLATIONS.get(lang, TRANSLATIONS["es"]).get(key, key)

# CSS personalizado
st.markdown("""
<style>
    .main {background-color: #f5f5f5;}
    .stButton>button {
        background-color: #0066cc;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #e8e8e8;
    }
    .success-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 5px;
        color: #155724;
    }
    /* Estilos para botones de idioma en la barra superior */
    div[data-testid="stHorizontalBlock"].lang-bar .stButton>button {
        border-radius: 50%;
        width: 40px;
        height: 40px;
        padding: 0;
        font-size: 13px;
        font-weight: bold;
        min-height: 0;
        line-height: 1;
    }
</style>
""", unsafe_allow_html=True)

# Inicializar session state
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None
if "lang" not in st.session_state:
    st.session_state.lang = "es"

# ========== BARRA SUPERIOR CON TÍTULO + IDIOMAS ==========
top_left, top_right = st.columns([4, 1])
with top_left:
    st.title(t("title"))
with top_right:
    lang_codes = ["es", "en", "ru"]
    lang_labels_top = ["ES", "EN", "RU"]
    cols = st.columns(3)
    for i, (code, label) in enumerate(zip(lang_codes, lang_labels_top)):
        with cols[i]:
            btn_type = "primary" if st.session_state.lang == code else "secondary"
            if st.button(label, key=f"lang_{code}", type=btn_type, use_container_width=True):
                if st.session_state.lang != code:
                    st.session_state.lang = code
                    st.rerun()

st.markdown(t("subtitle"))

# Sidebar
with st.sidebar:
    st.header(t("config"))
    
    modelo = st.selectbox(
        t("model"),
        ["gemma3:latest", "llama3-groq-tool-use:latest", "deepseek-r1:8b", "qwen2.5-coder:7b"],
        help=t("model_help")
    )
    
    temperatura = st.slider(
        t("temperature"),
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        help=t("temp_help")
    )
    
    chunk_size = st.number_input(
        t("chunk_size"),
        min_value=200,
        max_value=2000,
        value=1000,
        step=100
    )
    
    k_retrieval = st.number_input(
        t("chunks_retrieve"),
        min_value=1,
        max_value=20,
        value=8,
        step=1
    )
    
    st.divider()
    
    if st.button(t("clear_cache"), use_container_width=True):
        clear_cache()
        st.success(t("cache_cleared"))
    
    st.divider()
    
    # Estadísticas
    stats = get_cache_stats()
    st.subheader(t("cache_title"))
    st.metric(t("pdfs_saved"), stats["total_files"])
    st.metric(t("space_used"), f"{stats['cache_size']} MB")
    st.metric(t("cache_hits"), stats["hits"])

# Tabs principales
tab1, tab2, tab3 = st.tabs([t("tab_upload"), t("tab_questions"), t("tab_info")])

with tab1:
    st.subheader(t("upload_title"))
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        pdf_file = st.file_uploader(
            t("select_pdf"),
            type="pdf",
            help=t("max_size")
        )
    
    with col2:
        if st.button(t("load_btn"), use_container_width=True):
            if pdf_file is None:
                st.error(t("no_pdf"))
            else:
                try:
                    # Leer archivo una sola vez
                    pdf_content = pdf_file.read()
                    
                    # Validar PDF
                    is_valid, msg = validate_pdf(pdf_content, pdf_file.name)
                    if not is_valid:
                        st.error(f"❌ {msg}")
                    else:
                        with st.spinner(t("loading_pdf")):
                            # Usar el contenido ya leído
                            st.session_state.pdf_content = pdf_content
                            st.session_state.pdf_name = pdf_file.name
                            
                            st.markdown(f"""
                            <div class="success-box">
                            {t("pdf_loaded")}<br>
                            {t("file_label")}: {pdf_file.name}<br>
                            {t("size_label")}: {len(st.session_state.pdf_content)} bytes
                            </div>
                            """, unsafe_allow_html=True)
                            
                except Exception as e:
                    st.error(f"{t('load_error')}: {str(e)}")
                    with st.expander(t("tech_details")):
                        st.code(traceback.format_exc())
    
    # Mostrar PDF cargado
    if st.session_state.pdf_content:
        st.info(f"{t('current_pdf')}: **{st.session_state.pdf_name}**")

with tab2:
    st.subheader(t("ask_title"))
    
    if st.session_state.pdf_content is None:
        st.warning(t("upload_first"))
    else:
        st.info(f"{t('working_with')}: **{st.session_state.pdf_name}**")
        
        pregunta = st.text_input(
            t("your_question"),
            placeholder=t("question_placeholder"),
            help=t("question_help")
        )
        
        col1, col2 = st.columns([3, 1])
        
        with col2:
            buscar = st.button(t("search_btn"), use_container_width=True)
        
        if buscar and pregunta:
            with st.spinner(t("processing")):
                try:
                    # Procesar PDF a bytes
                    from langchain_community.document_loaders import PyPDFLoader
                    import tempfile
                    
                    # Guardar temporalmente
                    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                        tmp.write(st.session_state.pdf_content)
                        tmp_path = tmp.name
                    
                    # Cargar PDF
                    st.write(t("loading_pdf"))
                    loader = PyPDFLoader(tmp_path)
                    documents = loader.load()
                    
                    if not documents:
                        st.error(t("no_text"))
                        os.unlink(tmp_path)
                    else:
                        # Inicializar LLM
                        st.write(t("init_model"))
                        llm = OllamaLLM(
                            model=modelo,
                            temperature=temperatura,
                            base_url="http://localhost:11434"
                        )
                        
                        # Inicializar embeddings
                        st.write(t("gen_embeddings"))
                        embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                        
                        # Text splitter
                        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=200,
                            separators=["\n\n", "\n", ". ", " ", ""]
                        )
                        
                        # Crear chunks
                        st.write(t("splitting"))
                        chunks = splitter.split_documents(documents)
                        st.write(f"📊 {len(chunks)} {t('chunks_created')} {len(documents)} {t('pages')}")
                        
                        # Función para formatear documentos
                        def format_docs(docs):
                            return "\n\n".join(doc.page_content for doc in docs)
                        
                        # Prompt template con instrucción de idioma
                        lang_instruction = t("prompt_instruction")
                        template = f"""Eres un asistente experto. {lang_instruction}
Responde basándote ÚNICAMENTE en el contexto proporcionado.
Sé exhaustivo y completo en tu respuesta. Si la pregunta pide listar elementos, personajes, temas o datos, revisa TODO el contexto cuidadosamente e incluye TODOS los que aparezcan sin omitir ninguno.

Si no encuentras la respuesta en el contexto, dilo claramente.

Contexto:
{{context}}

Pregunta: {{question}}

Respuesta detallada:"""
                        
                        prompt = PromptTemplate(
                            template=template,
                            input_variables=["context", "question"]
                        )
                        
                        # Decidir estrategia según tamaño del documento
                        UMBRAL_CHUNKS = 20  # Si tiene pocos chunks, pasar todo al LLM
                        estrategia_usada = ""
                        
                        if len(chunks) <= UMBRAL_CHUNKS:
                            # Documento pequeño: pasar TODO el contenido
                            st.write(f"{t('small_doc')} ({len(chunks)} chunks) → {t('full_content')}")
                            estrategia_usada = t("strategy_full")
                            todo_el_contexto = format_docs(chunks)
                            
                            rag_chain = prompt | llm
                            
                            # Invocar con reintentos
                            st.write(t("generating"))
                            respuesta = None
                            for intento in range(3):
                                try:
                                    respuesta = rag_chain.invoke({
                                        "context": todo_el_contexto,
                                        "question": pregunta
                                    })
                                    break
                                except Exception as retry_err:
                                    if intento < 2:
                                        st.write(f"{t('retrying')} ({intento + 1}/3)")
                                        time.sleep(2)
                                    else:
                                        raise retry_err
                        else:
                            # Documento grande: usar RAG con búsqueda vectorial MMR
                            estrategia_usada = t("strategy_rag")
                            st.write(t("creating_index"))
                            vectorstore = Chroma.from_documents(
                                documents=chunks,
                                embedding=embeddings,
                                persist_directory="./chroma_db"
                            )
                            retriever = vectorstore.as_retriever(
                                search_type="mmr",
                                search_kwargs={
                                    "k": k_retrieval,
                                    "fetch_k": min(k_retrieval * 3, len(chunks)),
                                    "lambda_mult": 0.5
                                }
                            )
                            
                            st.write(t("building_chain"))
                            rag_chain = (
                                {"context": retriever | format_docs, "question": RunnablePassthrough()}
                                | prompt
                                | llm
                            )
                            
                            # Invocar con reintentos
                            st.write(t("generating"))
                            respuesta = None
                            for intento in range(3):
                                try:
                                    respuesta = rag_chain.invoke(pregunta)
                                    break
                                except Exception as retry_err:
                                    if intento < 2:
                                        st.write(f"{t('retrying')} ({intento + 1}/3)")
                                        time.sleep(2)
                                    else:
                                        raise retry_err
                        
                        if respuesta is None:
                            st.error(t("no_response"))
                        else:
                            # Mostrar resultado
                            st.markdown(f"""
                            <div class="success-box">
                            <b>{t('response')}</b>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            st.write(respuesta)
                        
                        # Detalles técnicos
                        with st.expander(t("search_details")):
                            st.write(f"**{t('model_label')}:** {modelo}")
                            st.write(f"**{t('temp_label')}:** {temperatura}")
                            st.write(f"**{t('strategy_label')}:** {estrategia_usada}")
                            st.write(f"**{t('chunks_retrieved')}:** {k_retrieval}")
                            st.write(f"**{t('total_chunks')}:** {len(chunks)}")
                            st.write(f"**{t('total_pages')}:** {len(documents)}")
                        
                        # Limpiar
                        os.unlink(tmp_path)
                    
                except Exception as e:
                    st.error(f"{t('error')}: {str(e)}")
                    with st.expander(t("tech_details")):
                        st.code(traceback.format_exc())
        
        elif buscar and not pregunta:
            st.warning(t("write_question"))

with tab3:
    st.subheader(t("info_title"))
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown(t("how_works"))
        st.markdown(t("how_steps"))
        st.markdown(t("models_title"))
        st.markdown(t("models_list"))
    
    with col2:
        st.markdown(t("tips_title"))
        st.markdown(t("tips_list"))
        st.markdown(t("reqs_title"))
        st.markdown(t("reqs_list"))
        st.markdown(t("stats_title"))
        
        stats = get_cache_stats()
        st.info(f"""
        **{t('pdfs_cached')}:** {stats['total_files']}
        **{t('space')}:** {stats['cache_size']} MB
        **{t('hits')}:** {stats['hits']}
        """)
    
    # Leyenda de lógica de chunks
    st.divider()
    st.markdown(t("chunks_legend_title"))
    st.markdown(t("chunks_legend"))

