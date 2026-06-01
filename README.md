# 📄 Agente PDF

Agente conversacional que permite chatear con documentos PDF. Sube un PDF y haz preguntas sobre su contenido mediante un chat potenciado por IA.

🔗 **Demo en vivo:** [agente-pdf.vercel.app](https://agente-pdf.vercel.app)

## ✨ Funcionalidades

- 📂 Carga y procesamiento de documentos PDF
- 💬 Chat interactivo con el contenido del PDF
- 🤖 Respuestas generadas por LLM con contexto del documento
- ⚡ Interfaz rápida con TypeScript + Vite
- 🌐 Desplegado en Vercel (frontend) y API serverless

## 🛠️ Stack Tecnológico

| Capa | Tecnología |
|------|------------|
| Frontend | TypeScript · Vite · CSS |
| Backend / API | Python · FastAPI |
| IA / LLM | LLM API (OpenRouter) |
| Despliegue | Vercel |
| CI/CD | GitHub Actions |

## 🚀 Uso

Accede directamente a la demo: [agente-pdf.vercel.app](https://agente-pdf.vercel.app)

O ejecuta localmente:

```bash
# Clonar el repositorio
git clone https://github.com/JoanTarazona99/agente-pdf.git
cd agente-pdf

# Instalar dependencias del frontend
npm install

# Iniciar en modo desarrollo
npm run dev
```

## 📁 Estructura

```
agente-pdf/
├── api/    # Backend Python / endpoints
├── src/    # Frontend TypeScript
└── index.html
```

## 👤 Autor

**Joan Tarazona** · [GitHub](https://github.com/JoanTarazona99)
