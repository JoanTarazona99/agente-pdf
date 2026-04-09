import os
# Evitar que httpx (usado por ollama/langchain-ollama) enrute a través del proxy del sistema
os.environ.setdefault('NO_PROXY', 'localhost,127.0.0.1')

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import tempfile

from langchain_ollama import OllamaLLM
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

# ==========================================
# Startup/Shutdown Events
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Iniciando API Agente PDF...")
    cache_stats = get_cache_stats()
    print(f"📊 Caché: {cache_stats['total_files']} archivos, {cache_stats['cache_size_mb']} MB")
    yield
    # Shutdown
    print("🛑 Cerrando API Agente PDF...")

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="🤖 Agente PDF API",
    description="API para hacer preguntas sobre archivos PDF usando IA",
    version="2.0.0",
    lifespan=lifespan
)

# Agregar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Rutas
# ==========================================

@app.get("/")
async def root():
    """Ruta raíz con información de la API."""
    return {
        "titulo": "🤖 Agente PDF API",
        "version": "2.0.0",
        "descripcion": "Lee PDFs y responde preguntas sobre su contenido",
        "endpoint": "/preguntar",
        "documentacion": "/docs",
        "cache": get_cache_stats()
    }

@app.get("/health")
async def health_check():
    """Verificar si el servidor está funcionando."""
    return {
        "status": "✅ Sistema operacional",
        "version": "2.0.0"
    }

@app.post("/preguntar")
async def preguntar_pdf(
    pregunta: str = Form(..., description="Pregunta sobre el PDF"),
    pdf: UploadFile = File(..., description="Archivo PDF a analizar"),
    modelo: str = Form(default="gemma3:latest", description="Modelo LLM a usar (disponible en ollama list)"),
    temperatura: float = Form(default=0.3, description="Temperatura (0.0-1.0)"),
    host_ollama: str = Form(default="http://localhost:11434", description="Host de Ollama")
):
    """
    Hacer una pregunta sobre un PDF.
    
    **Parámetros:**
    - pregunta: Tu pregunta sobre el contenido del PDF
    - pdf: El archivo PDF a analizar
    - modelo: Modelo LLM (default: gemma3:latest)
    - temperatura: Temperatura LLM (default: 0.3)
    - host_ollama: URL del servidor Ollama (default: http://localhost:11434)
    
    **Respuesta:**
    - respuesta: La respuesta a tu pregunta
    - documento: Nombre del PDF
    - metadata: Información sobre el procesamiento
    """
    
    try:
        # ========== VALIDACIONES ==========
        if not pregunta or not pregunta.strip():
            raise HTTPException(status_code=400, detail="❌ La pregunta no puede estar vacía")
        
        if temperatura < 0.0 or temperatura > 1.0:
            raise HTTPException(status_code=400, detail="❌ Temperatura debe estar entre 0.0 y 1.0")
        
        # Leer contenido del PDF
        file_content = await pdf.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="❌ El archivo está vacío")
        
        # Validar PDF
        is_valid, validation_msg = validate_pdf(file_content, pdf.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_msg)
        
        # ========== PROCESAR PDF ==========
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_content)
            tmp_path = tmp.name
        
        try:
            # Cargar PDF
            loader = PyPDFLoader(tmp_path)
            documents = loader.load()
            
            if not documents:
                raise HTTPException(status_code=400, detail="❌ No se pudo extraer texto del PDF")
            
            # Dividir en chunks
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            chunks = splitter.split_documents(documents)
            
            # ========== EMBEDDINGS Y VECTORSTORE ==========
            embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
            
            vectorstore = Chroma.from_documents(
                documents=chunks,
                embedding=embeddings,
                persist_directory="./chroma_db"
            )
            retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
            
            # ========== CONECTAR CON LLM ==========
            try:
                llm = OllamaLLM(
                    model=modelo,
                    temperature=temperatura,
                    base_url=host_ollama
                )
                
                # Crear prompt
                template = """Eres un asistente experto. Responde basándote ÚNICAMENTE en el contexto proporcionado.

Si no encuentras la respuesta en el contexto, di claramente: "No encontré esa información en el documento."

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
                
                # Crear cadena RAG
                rag_chain = (
                    {"context": retriever | format_docs, "question": RunnablePassthrough()}
                    | prompt
                    | llm
                )
                
                # ========== GENERAR RESPUESTA ==========
                respuesta = rag_chain.invoke(pregunta)
                
                # ========== RETORNAR RESPUESTA ==========
                return JSONResponse(content={
                    "status": "✅ Éxito",
                    "respuesta": respuesta,
                    "documento": pdf.filename,
                    "pregunta": pregunta,
                    "modelo": modelo,
                    "temperatura": temperatura,
                    "metadata": {
                        "paginas": len(documents),
                        "chunks": len(chunks),
                        "tamaño_mb": len(file_content) / (1024 * 1024)
                    }
                }, status_code=200)
                
            except ConnectionError:
                raise HTTPException(
                    status_code=500,
                    detail=f"❌ No se puede conectar con Ollama en {host_ollama}. Verifica que esté ejecutándose."
                )
            except Exception as e:
                raise HTTPException(
                    status_code=500,
                    detail=f"❌ Error del modelo: {str(e)}"
                )
        
        finally:
            # Limpiar archivo temporal
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"❌ Error inesperado: {str(e)}"
        )

@app.get("/cache")
async def cache_info():
    """Obtener información sobre la caché."""
    return {
        "cache": get_cache_stats()
    }

@app.delete("/cache")
async def delete_cache():
    """Limpiar toda la caché."""
    resultado = clear_cache()
    return {
        "status": resultado
    }

# ==========================================
# Ejecutar servidor
# ==========================================

if __name__ == "__main__":
    import uvicorn
    print("""
    ╔═══════════════════════════════════════════╗
    ║  🚀 Agente PDF API v2.0                   ║
    ║  http://localhost:8000                    ║
    ║  📚 Docs: http://localhost:8000/docs      ║
    ╚═══════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)

# ==========================================
# Startup/Shutdown Events
# ==========================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("🚀 Iniciando API Agente PDF...")
    print(f"📊 Caché: {get_cache_info()}")
    yield
    # Shutdown
    print("🛑 Cerrando API Agente PDF...")

# ==========================================
# FastAPI App
# ==========================================

app = FastAPI(
    title="🤖 Agente PDF API",
    description="API para hacer preguntas sobre archivos PDF usando IA",
    version="1.0.0",
    lifespan=lifespan
)

# Agregar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==========================================
# Rutas
# ==========================================

@app.get("/")
async def root():
    """Ruta raíz con información de la API."""
    return {
        "titulo": "🤖 Agente PDF API",
        "version": "1.0.0",
        "descripcion": "Lee PDFs y responde preguntas sobre su contenido",
        "endpoint": "/preguntar",
        "documentacion": "/docs",
        "cache": get_cache_info()
    }

@app.get("/health")
async def health_check():
    """Verificar si el servidor está funcionando."""
    return {
        "status": "✅ Sistema operacional",
        "version": "1.0.0"
    }

@app.post("/preguntar")
async def preguntar_pdf(
    pregunta: str = Form(..., description="Pregunta sobre el PDF"),
    pdf: UploadFile = File(..., description="Archivo PDF a analizar"),
    modelo: str = Form(default="gemma3:latest", description="Modelo LLM a usar"),
    host_ollama: str = Form(default="http://localhost:11434", description="Host de Ollama")
):
    """
    Hacer una pregunta sobre un PDF.
    
    **Parámetros:**
    - pregunta: Tu pregunta sobre el contenido del PDF
    - pdf: El archivo PDF a analizar
    - modelo: Modelo LLM (default: gemma3:latest)
    - host_ollama: URL del servidor Ollama (default: http://localhost:11434)
    
    **Respuesta:**
    - respuesta: La respuesta a tu pregunta
    - fuentes: Información sobre pages/chunks procesados
    """
    
    try:
        # ========== VALIDACIONES ==========
        if not pregunta or not pregunta.strip():
            raise HTTPException(status_code=400, detail="❌ La pregunta no puede estar vacía")
        
        # Leer contenido del PDF
        file_content = await pdf.read()
        
        if not file_content:
            raise HTTPException(status_code=400, detail="❌ El archivo está vacío")
        
        # Validar PDF
        is_valid, validation_msg = validate_pdf(file_content, pdf.filename)
        if not is_valid:
            raise HTTPException(status_code=400, detail=validation_msg)
        
        # ========== PROCESAR PDF ==========
        db, metadata = load_or_create_vectorstore(
            file_content,
            pdf.filename,
            use_cache=True
        )
        
        if not db:
            raise HTTPException(status_code=500, detail="❌ Error procesando el PDF")
        
        # ========== CONECTAR CON LLM ==========
        try:
            llm = OllamaLLM(
                model=modelo,
                base_url=host_ollama
            )
            
            # Crear cadena RAG
            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=db.as_retriever(search_kwargs={"k": 4})
            )
            
            # ========== GENERAR RESPUESTA ==========
            respuesta_data = qa_chain.invoke({"query": pregunta})
            respuesta = format_response(respuesta_data)
            
            # ========== RETORNAR RESPUESTA ==========
            return JSONResponse(content={
                "status": "✅ Éxito",
                "respuesta": respuesta,
                "documento": pdf.filename,
                "pregunta": pregunta,
                "modelo": modelo,
                "fuentes": {
                    "paginas": metadata.get("pages", "?"),
                    "fragmentos": metadata.get("chunks", "?"),
                    "tamaño_mb": metadata.get("size_mb", "?")
                }
            }, status_code=200)
            
        except ConnectionError as e:
            raise HTTPException(
                status_code=500,
                detail=f"❌ No se puede conectar con Ollama en {host_ollama}. Verifica que esté ejecutándose."
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"❌ Error del modelo: {str(e)}"
            )
        
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(
            status_code=400,
            detail=f"❌ Error procesando PDF: {str(e)}"
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"❌ Error inesperado: {str(e)}"
        )

@app.get("/cache")
async def cache_info():
    """Obtener información sobre la caché de vectores."""
    return {
        "cache": get_cache_info()
    }

@app.delete("/cache")
async def clear_cache_endpoint():
    """Limpiar toda la caché."""
    from utils import clear_cache
    resultado = clear_cache()
    return {
        "status": resultado
    }

# ==========================================
# Ejecutar servidor
# ==========================================

if __name__ == "__main__":
    import uvicorn
    print("""
    🚀 Iniciando servidor FastAPI...
    📚 Documentación: http://localhost:8000/docs
    🔗 API: http://localhost:8000/preguntar
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
