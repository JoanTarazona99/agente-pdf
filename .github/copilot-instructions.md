---
name: copilot-instructions
description: |-
  Instrucciones del proyecto "Agente PDF" para Copilot. Usar cuando se trabaja
  en funcionalidades relacionadas con carga y consulta de PDFs, embeddings,
  vector store y la interfaz React. Frases clave: PDF, embeddings, vector-store,
  groq, Vercel, despliegue, configuración de API.
applyTo: "**/*"
---

Propósito
- Esta guía ayuda a Copilot a actuar como asistente de desarrollo específico
  para el proyecto "Agente PDF" ubicado en la raíz del workspace.

Contexto del proyecto
- Framework: Vite + React + TypeScript.
- Carpetas clave: `src/`, `lib/`, `api/`, `components/`, `utils/`.
- Funcionalidad principal: subir PDFs, extraer texto, generar embeddings,
  almacenar vectores y permitir búsqueda semántica desde una UI tipo chat.

Cómo ayudar
- Prioriza cambios pequeños y seguros que respeten la estructura existente.
- Cuando implementes nuevas funciones, crea tests mínimos si procede y
  actualiza `package.json` solo si es necesario.
- Prefiere soluciones que mantengan la compatibilidad con despliegue en Vercel.

Convenciones y estilo
- Usa TypeScript con tipos explícitos en las interfaces públicas.
- Mantén componentes React como funciones pequeñas y reutilizables.
- Evita cambios globales en dependencias sin consultarlo primero.

Pautas de seguridad y secretos
- Nunca incluir claves API o secretos en el código. Sugiere usar variables de
  entorno (`process.env`) y documenta qué variables son necesarias.

Comandos útiles (sugerir al usuario ejecutar)
- Instalar dependencias: `npm install` o `pnpm install`.
- Desarrollo local: `npm run dev`.
- Build para producción: `npm run build`.

Preguntas frecuentes para el desarrollador
- ¿Necesitas soporte para un proveedor de embeddings nuevo? — Solicita la
  configuración y añade la abstracción en `lib/embeddings.ts`.
- ¿Hay que ajustar el vector store? — Revisa `lib/vector-store.ts` primero.

Si necesitas cambios mayores
- Pregunta al usuario antes de reestructurar carpetas, cambiar build tools o
  modificar la estrategia de almacenamiento de vectores.

Fin
