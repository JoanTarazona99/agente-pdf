# 📘 Agente PDF - Lector y Respondedor de PDFs con IA

Un sistema interactivo que lee archivos PDF y responde preguntas sobre su contenido utilizando técnicas avanzadas de **RAG (Retrieval-Augmented Generation)** con LLMs locales.

## 🎯 Características

- 📄 **Carga de PDFs**: Sube cualquier PDF y el sistema lo procesa automáticamente
- 🤖 **Respuestas Inteligentes**: Responde preguntas específicas sobre el contenido del PDF
- 🎙️ **Interfaz de Voz**: Realiza preguntas por voz y obtén respuestas habladas (en `staaudio.py`)
- ⚡ **RAG Avanzado**: Usa embeddings y recuperación semántica para respuestas precisas
- 🌐 **Múltiples Interfaces**: Streamlit (web) + FastAPI (API REST)
- 💾 **Caché Inteligente**: Almacena vectores para procesamiento más rápido

## 🛠️ Requisitos Previos

- Python 3.9+
- Ollama ejecutándose en `localhost:11434` con modelo `kiwi_kiwi/gemma-4-uncensores:e4b` instalado
- Git (opcional)

### Instalar Ollama y Descargar Modelo

```bash
# Instala Ollama desde: https://ollama.ai

# Descarga el modelo (ejemplo)
ollama pull kiwi_kiwi/gemma-4-uncensores:e4b

# Inicia el servidor Ollama (en otra terminal)
ollama serve
```

## 📦 Instalación

### 1. Clonar/Descargar Repositorio

```bash
cd agente_pdf
```

### 2. Crear Entorno Virtual

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 4. Configurar Variables de Entorno (opcional)

Crea un archivo `.env` en la raíz:

```env
OLLAMA_HOST=http://localhost:11434
PDF_MAX_SIZE_MB=50
CACHE_DIR=./cache
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 🚀 Uso

### Opción 1: Interfaz Web Streamlit (Recomendado)

```bash
streamlit run st.py
```

Abre `http://localhost:8501` en tu navegador.

**Características:**
- Carga de PDFs con arrastrar y soltar
- Respuestas rápidas
- Interfaz intuitiva

### Opción 2: Interfaz con Voz

```bash
streamlit run staaudio.py
```

**Características:**
- Graba preguntas por micrófono
- Respuestas en voz (TTS)
- Escritura opcional de preguntas

### Opción 3: API FastAPI

```bash
python -m uvicorn stn8n:app --reload --host 0.0.0.0 --port 8000
```

Accede a `http://localhost:8000/docs` para ver la documentación interactiva.

**Endpoint:**
```bash
POST /preguntar
- pregunta (string): Tu pregunta
- pdf (file): Archivo PDF
```

**Ejemplo con cURL:**
```bash
curl -X POST "http://localhost:8000/preguntar" \
  -F "pregunta=¿Cuál es el tema principal?" \
  -F "pdf=@mi_documento.pdf"
```

### Opción 4: Script Python Simple

```bash
python agente.py
```

Edita `agente.py` para cambiar la ruta del PDF y la pregunta.

## 📂 Estructura del Proyecto

```
agente_pdf/
├── requirements.txt           # Dependencias del proyecto
├── README.md                  # Este archivo
├── .env                       # Variables de entorno
│
├── 🎨 INTERFACES
├── st.py                      # Streamlit - Interfaz web principal
├── staaudio.py                # Streamlit con voz y audio
├── stn8n.py                   # FastAPI - API REST
├── agente.py                  # Script Python básico
│
├── 🛠️ UTILITIES
├── utils.py                   # Funciones compartidas y caché
│
├── 📚 RAG AVANZADO
├── rag-tutorial-v2/
│   ├── populate_database.py   # Cargar PDFs en BD vectorial
│   ├── query_data.py          # Consultar BD vectorial
│   ├── get_embedding_function.py
│   ├── test_rag.py
│   └── data/                  # PDFs para procesamiento batch
│
├── 💾 CACHÉ Y DATOS
├── cache/                     # Vectores FAISS cacheados
├── __pycache__/
└── venv/                      # Entorno virtual
```

## ⚙️ Configuración Avanzada

### Cambiar Modelo de Lenguaje

En los archivos `.py`, busca la línea:

```python
llm = OllamaLLM(model="kiwi_kiwi/gemma-4-uncensores:e4b")
```

Y cámbiala a otro modelo disponible:

```bash
# Ver modelos disponibles
ollama list

# Descargar nuevo modelo
ollama pull llama2  # o mistral, neural-chat, etc.
```

### Ajustar Tamaño de Chunks

En `st.py`, `staaudio.py`, `stn8n.py`:

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,      # Reducir para respuestas más precisas
    chunk_overlap=50     # Aumentar para mejor contexto
)
```

### Cambiar Modelo de Embeddings

```python
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-mpnet-base-v2"  # Más grande, mejor precisión
)
```

## 🐛 Solución de Problemas

### Error: "ConnectionError: Could not connect to Ollama"
- ✅ Verifica que Ollama esté ejecutándose: `ollama serve`
- ✅ Comprueba que el modelo existe: `ollama list`

### Error: "ModuleNotFoundError"
- ✅ Asegúrate de activar el venv: `source venv/bin/activate`
- ✅ Instala dependencias: `pip install -r requirements.txt`

### Error: "PDF no se procesa"
- ✅ Verifica que el PDF no esté corrupto
- ✅ Reduce `chunk_size` en el splitter
- ✅ Aumenta timeout en configuración

### Lentitud en primera carga
- ✅ El primer PDF tarda más porque genera embeddings
- ✅ Los siguientes son más rápidos (caché automático)
- ✅ Considera usar GPUs si tienes CUDA disponible

## 📊 Rendimiento

| Interfaz | Velocidad | Memoria | Mejor para |
|----------|-----------|---------|-----------|
| Streamlit (`st.py`) | ⭐⭐⭐⭐⭐ | Bajo | Demostración, producción |
| Audio (`staaudio.py`) | ⭐⭐ | Alto | Accesibilidad, voz |
| FastAPI (`stn8n.py`) | ⭐⭐⭐⭐ | Normal | Integración, escalabilidad |
| Script Python | ⭐⭐⭐ | Bajo | Testing, automatización |

## 🔐 Consideraciones de Seguridad

- Valida tamaño máximo de archivos
- Limpia archivos temporales
- No almacenes PDFs sensibles sin encriptar
- Usa HTTPS en producción

## 📝 Roadmap

- [ ] Caché persistente mejorado
- [ ] Soporte para múltiples PDFs simultáneamente
- [ ] Análisis de documentos (tabla de contenidos, índice)
- [ ] Exportar respuestas a PDF
- [ ] Modo oscuro en UI

## 📄 Licencia

Proyecto de demostración.

---

**¿Dudas o mejoras?** Modifica los archivos según tus necesidades. El código es simple y extensible.

**Última actualización:** 25 de marzo de 2026
