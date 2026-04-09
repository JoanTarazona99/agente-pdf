import os
# Evitar que httpx (usado por ollama/langchain-ollama) enrute a través del proxy del sistema
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')

import streamlit as st
import pyttsx3
import speech_recognition as sr
import streamlit_audiorec as st_audiorec
import traceback
import tempfile

from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from langchain_core.embeddings import Embeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


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

# Configurar página
st.set_page_config(
    page_title="🎙️ Agente PDF con Voz",
    page_icon="🎤",
    layout="wide"
)

# --------------------------
# 🔊 Función para hablar (TTS)
# --------------------------
def speak(text, language="es"):
    """Reproduce texto en voz."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        
        # Seleccionar voz en español si está disponible
        voices = engine.getProperty('voices')
        spanish_voice_found = False
        
        for voice in voices:
            if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                spanish_voice_found = True
                break
        
        if not spanish_voice_found and voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.say(text)
        engine.runAndWait()
        
    except Exception as e:
        st.warning(f"⚠️ No se pudo reproducir audio: {str(e)}")

# --------------------------
# 🗣️ Función para convertir audio en texto (STT)
# --------------------------
def audio_to_text(audio_bytes):
    """Convierte audio a texto usando Google Speech Recognition."""
    try:
        recognizer = sr.Recognizer()
        
        # Guardar audio temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(audio_bytes)
            tmp_wav_path = tmp_wav.name
        
        # Procesar audio
        with sr.AudioFile(tmp_wav_path) as source:
            audio = recognizer.record(source)
        
        # Reconocer en español
        try:
            return recognizer.recognize_google(audio, language="es-ES")
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
            
    except Exception as e:
        st.error(f"❌ Error procesando audio: {str(e)}")
        return None

# --------------------------
# 🎨 Interfaz Streamlit
# --------------------------

st.markdown("""
    <h1 style="color: #1f77b4;">🎙️ Agente PDF con Voz</h1>
    <p>Sube un PDF, haz preguntas por voz y recibe respuestas habladas 🎤🔊</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    modelo = st.selectbox(
        "🤖 Modelo Ollama",
        ["gemma3:latest", "llama3-groq-tool-use:latest", "deepseek-r1:8b", "qwen2.5-coder:7b"]
    )
    
    temperatura = st.slider(
        "🌡️ Temperatura:",
        min_value=0.0,
        max_value=1.0,
        value=0.3
    )
    
    enable_tts = st.checkbox("🔊 Activar voz en respuestas", value=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Limpiar caché"):
        clear_cache()
        st.success("✅ Caché limpiada")
        st.rerun()

# Inicializar session state
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = None
if "pdf_name" not in st.session_state:
    st.session_state.pdf_name = None

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar PDF")
    uploaded_file = st.file_uploader(
        "Sube un archivo PDF",
        type="pdf",
        label_visibility="collapsed"
    )
    
    if uploaded_file:
        try:
            is_valid, msg = validate_pdf(uploaded_file.read(), uploaded_file.name)
            if is_valid:
                # Releer archivo
                st.session_state.pdf_content = uploaded_file.read()
                st.session_state.pdf_name = uploaded_file.name
                st.success("✅ PDF cargado")
        except Exception as e:
            st.error(f"Error: {e}")

with col2:
    st.subheader("📊 Información")
    stats = get_cache_stats()
    st.metric("Archivos cacheados", stats["total_files"])
    st.metric("Tamaño caché", f"{stats['cache_size_mb']} MB")

st.markdown("---")

# Sección de entrada de voz
st.subheader("🎤 Opciones de Pregunta")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Opción 1: Graba por micrófono**")
    wav_audio_data = st_audiorec.st_audiorec()

with col2:
    st.write("**Opción 2: Escribe manualmente**")
    query_manual = st.text_input(
        "Escribe tu pregunta aquí",
        label_visibility="collapsed",
        placeholder="¿Cuál es...?"
    )

# Determinar pregunta
query = ""

if wav_audio_data:
    with st.spinner("🎙️ Procesando audio..."):
        texto_audio = audio_to_text(wav_audio_data)
        
        if texto_audio:
            st.info(f"✅ Detectado: *{texto_audio}*")
            query = texto_audio
        else:
            st.warning("⚠️ No se pudo reconocer el audio. Intenta nuevamente.")

if query_manual and query_manual.strip():
    query = query_manual

# Procesar pregunta y PDF
st.markdown("---")

if st.button("🔍 Obtener Respuesta", use_container_width=True):
    
    # Validaciones
    if not st.session_state.pdf_content:
        st.error("❌ Por favor sube un PDF")
        st.stop()
    
    if not query or not query.strip():
        st.error("❌ Por favor haz una pregunta (voz o texto)")
        st.stop()
    
    # Procesar
    with st.spinner("⏳ Cargando y procesando PDF..."):
        try:
            # Guardar PDF temporalmente
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(st.session_state.pdf_content)
                tmp_path = tmp.name
            
            # Cargar PDF
            st.write("📖 Cargando PDF...")
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            if not documents:
                st.error("❌ No se pudo extraer texto del PDF")
                os.unlink(tmp_path)
            else:
                # Inicializar LLM
                st.write("🤖 Inicializando modelo...")
                llm = OllamaLLM(
                    model=modelo,
                    temperature=temperatura,
                    base_url="http://localhost:11434"
                )
                
                # Inicializar embeddings
                st.write("🔢 Generando embeddings...")
                embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
                
                # Text splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )
                
                # Crear chunks
                st.write("✂️ Dividiendo texto...")
                chunks = splitter.split_documents(documents)
                st.write(f"📊 {len(chunks)} chunks de {len(documents)} páginas")
                
                # Vectorstore
                st.write("🗺️ Creando índice...")
                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    persist_directory="./chroma_db"
                )
                retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # Cadena RAG
                template = """Eres un asistente experto. Responde basándote ÚNICAMENTE en el contexto.

Contexto:
{context}

Pregunta: {question}

Respuesta:"""
                
                prompt = PromptTemplate(
                    template=template,
                    input_variables=["context", "question"]
                )
                
                def format_docs(docs):
                    return "\n\n".join(doc.page_content for doc in docs)
                
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                )
                
                # Generar respuesta
                st.write("💭 Generando respuesta...")
                respuesta = rag_chain.invoke(query)
                
                # Mostrar
                st.markdown("""
                <div style="background-color: #d4edda; padding: 15px; border-radius: 5px;">
                <b>✅ Respuesta:</b>
                </div>
                """, unsafe_allow_html=True)
                
                st.write(respuesta)
                
                # Reproducir si está habilitado
                if enable_tts:
                    st.info("🔊 Reproduciendo respuesta...")
                    speak(respuesta)
                    st.success("✅ Respuesta reproducida")
                
                # Limpiar
                os.unlink(tmp_path)
                
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("📋 Detalles"):
                st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    🎙️ Agente PDF con Voz v2.0 | 25/03/2026
</div>
""", unsafe_allow_html=True)

# Configurar página
st.set_page_config(
    page_title="🎙️ Agente PDF con Voz",
    page_icon="🎤",
    layout="wide"
)

# --------------------------
# 🔊 Función para hablar (TTS)
# --------------------------
def speak(text, language="es"):
    """Reproduce texto en voz."""
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 160)
        
        # Seleccionar voz en español si está disponible
        voices = engine.getProperty('voices')
        spanish_voice_found = False
        
        for voice in voices:
            if "spanish" in voice.name.lower() or "es" in voice.id.lower():
                engine.setProperty('voice', voice.id)
                spanish_voice_found = True
                break
        
        if not spanish_voice_found and voices:
            engine.setProperty('voice', voices[0].id)
        
        engine.say(text)
        engine.runAndWait()
        
    except Exception as e:
        st.warning(f"⚠️ No se pudo reproducir audio: {str(e)}")

# --------------------------
# 🗣️ Función para convertir audio en texto (STT)
# --------------------------
def audio_to_text(audio_bytes):
    """Convierte audio a texto usando Google Speech Recognition."""
    try:
        recognizer = sr.Recognizer()
        
        # Guardar audio temporalmente
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_wav:
            tmp_wav.write(audio_bytes)
            tmp_wav_path = tmp_wav.name
        
        # Procesar audio
        with sr.AudioFile(tmp_wav_path) as source:
            audio = recognizer.record(source)
        
        # Reconocer en español
        try:
            return recognizer.recognize_google(audio, language="es-ES")
        except sr.UnknownValueError:
            return None
        except sr.RequestError:
            return None
            
    except Exception as e:
        st.error(f"❌ Error procesando audio: {str(e)}")
        return None

# --------------------------
# 🎨 Interfaz Streamlit
# --------------------------

# Header
st.markdown("""
    <h1 style="color: #1f77b4;">🎙️ Agente PDF con Voz</h1>
    <p>Sube un PDF, haz preguntas por voz y recibe respuestas habladas 🎤🔊</p>
""", unsafe_allow_html=True)

st.markdown("---")

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuración")
    
    ollama_host = st.text_input(
        "🔗 Host de Ollama",
        value="http://localhost:11434"
    )
    
    model_name = st.selectbox(
        "🤖 Modelo LLM",
        ["gemma3:latest", "llama3-groq-tool-use:latest", "deepseek-r1:8b", "qwen2.5-coder:7b"]
    )
    
    enable_tts = st.checkbox("🔊 Activar voz en respuestas", value=True)
    
    st.markdown("---")
    
    if st.button("🗑️ Limpiar caché"):
        clear_cache()
        st.success("✅ Caché limpiada")
        st.rerun()

# Two-column layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📤 Cargar PDF")
    uploaded_file = st.file_uploader(
        "Sube un archivo PDF",
        type="pdf",
        label_visibility="collapsed"
    )

with col2:
    st.subheader("📊 Información")
    cache_info = get_cache_info()
    st.metric("Archivos cacheados", cache_info["cached_files"])
    st.metric("Tamaño caché", f"{cache_info['cache_size_mb']} MB")

st.markdown("---")

# Sección de entrada de voz
st.subheader("🎤 Opciones de Pregunta")

col1, col2 = st.columns([1, 1])

with col1:
    st.write("**Opción 1: Graba por micrófono**")
    wav_audio_data = st_audiorec.st_audiorec()

with col2:
    st.write("**Opción 2: Escribe manualmente**")
    query_manual = st.text_input(
        "Escribe tu pregunta aquí",
        label_visibility="collapsed",
        placeholder="¿Cuál es...?"
    )

# Determinar pregunta
query = ""
prediction_source = None

if wav_audio_data:
    with st.spinner("🎙️ Procesando audio..."):
        texto_audio = audio_to_text(wav_audio_data)
        
        if texto_audio:
            st.info(f"✅ Detectado: *{texto_audio}*")
            query = texto_audio
            prediction_source = "voz"
        else:
            st.warning("⚠️ No se pudo reconocer el audio. Intenta nuevamente.")

if query_manual and query_manual.strip():
    query = query_manual
    prediction_source = "texto"

# Procesar pregunta y PDF
st.markdown("---")

if st.button("🔍 Obtener Respuesta", use_container_width=True):
    
    # Validaciones
    if not uploaded_file:
        st.error("❌ Por favor sube un PDF")
        st.stop()
    
    if not query or not query.strip():
        st.error("❌ Por favor haz una pregunta (voz o texto)")
        st.stop()
    
    # Validar PDF
    file_bytes = uploaded_file.read()
    is_valid, validation_msg = validate_pdf(file_bytes, uploaded_file.name)
    st.info(validation_msg)
    
    if not is_valid:
        st.stop()
    
    # Procesar
    with st.spinner("⏳ Cargando y procesando PDF..."):
        try:
            # Cargar vectorstore con caché
            db, metadata = load_or_create_vectorstore(
                file_bytes,
                uploaded_file.name,
                use_cache=True
            )
            
            st.success(f"✅ PDF procesado - {metadata.get('pages', '?')} páginas")
            
            # Conectar con LLM
            with st.spinner("🤖 Generando respuesta..."):
                try:
                    llm = OllamaLLM(
                        model=model_name,
                        base_url=ollama_host
                    )
                    
                    qa_chain = RetrievalQA.from_chain_type(
                        llm=llm,
                        retriever=db.as_retriever(search_kwargs={"k": 4})
                    )
                    
                    response = qa_chain.invoke({"query": query})
                    answer = format_response(response)
                    
                    # Mostrar respuesta
                    st.markdown("---")
                    st.success("✅ Respuesta Generada")
                    
                    st.markdown(f"""
                    <div style="background-color: #e8f4f8; padding: 20px; border-radius: 10px;">
                        <p><strong>📄:</strong> {uploaded_file.name}</p>
                        <p><strong>❓:</strong> {query}</p>
                        <hr>
                        <p><strong>🤖 Respuesta:</strong></p>
                        <p style="font-size: 1.1em; line-height: 1.6;">{answer}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Reproducir respuesta si está habilitado
                    if enable_tts:
                        st.info("🔊 Reproduciendo respuesta...")
                        speak(answer)
                        st.success("✅ Respuesta reproducida")
                    
                    # Mostrar código (copiar)
                    with st.expander("📋 Copiar respuesta"):
                        st.code(answer, language="text")
                    
                except ConnectionError:
                    st.error(f"""
                    ❌ No se can conectar con Ollama
                    
                    Verifica:
                    - Host: {ollama_host}
                    - `ollama serve` ejecutándose
                    - `ollama pull {model_name}` instalado
                    """)
                    
        except ValueError as e:
            st.error(f"❌ Error procesando PDF: {str(e)}")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            with st.expander("🔧 Detalles"):
                st.code(traceback.format_exc())

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 0.9em;">
    🎙️ Agente PDF con Voz v1.0 | 25/03/2026
</div>
""", unsafe_allow_html=True)
