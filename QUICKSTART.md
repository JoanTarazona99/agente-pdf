# 🚀 INICIO RÁPIDO - Agente PDF

## ⚡ 5 Pasos para Empezar

### 1️⃣ Requisitos Previos
- Python 3.9+
- Ollama instalado desde https://ollama.ai

### 2️⃣ Instalación Automática
```bash
python setup.py
```

Si tienes problemas, instala manualmente:
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# o
source venv/bin/activate  # Linux/Mac

pip install -r requirements.txt
```

### 3️⃣ Descargar Modelo LLM
En otra terminal:
```bash
ollama pull gemma3:latest
```

### 4️⃣ Iniciar Ollama
```bash
ollama serve
```

### 5️⃣ Ejecutar la Aplicación

**Opción A: Streamlit (Interfaz Web - Recomendado)**
```bash
streamlit run st.py
```
Abre: http://localhost:8501

**Opción B: Con Voz y Audio**
```bash
streamlit run staaudio.py
```

**Opción C: API FastAPI**
```bash
python -m uvicorn stn8n:app --reload
```
Documentación: http://localhost:8000/docs

---

## 🔧 Solución de Problemas

### "ConnectionError: Could not connect to Ollama"
```bash
# Terminal 1: Inicia Ollama
ollama serve

# Terminal 2: Verifica que funciona
ollama list

# Terminal 3: Ejecuta la app
streamlit run st.py
```

### "Module not found" o "ModuleNotFoundError"
```bash
# Asegúrate de activar el venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Reinstala dependencias
pip install -r requirements.txt
```

### PDF no se procesa
- ✓ Descarga el PDF directamente (no copias)
- ✓ Verifica que no sea protegido
- ✓ Intenta con un PDF de texto plano (no imagen)

### Respuestas lentas
- Primera carga: Normal, genera embeddings
- Recargas: Deberían ser rápidas (usa caché)
- Limpia caché si algo se corrompe: botón 🗑️

---

## 📚 Próximos Pasos

1. Lee [README.md](README.md) para documentación completa
2. Experimenta con diferentes modelos: `ollama pull llama2`
3. Ajusta `CHUNK_SIZE` en [utils.py](utils.py) para más/menos detalle
4. Crea un frontend personalizado con tu API FastAPI

---

**¿Listo?** Abre http://localhost:8501 🎉
