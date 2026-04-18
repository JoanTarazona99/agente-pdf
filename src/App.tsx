import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  FileText, 
  Send, 
  Settings as SettingsIcon, 
  Mic, 
  MicOff, 
  Plus, 
  X, 
  Trash2, 
  AlertCircle,
  Loader2
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { PDFProcessor } from './lib/pdf-processor';
import { EmbeddingsManager } from './lib/embeddings';
import { VectorStore, DocumentChunk } from './lib/vector-store';
import { TRANSLATIONS, Language } from './lib/translations';
import { chatWithGroq, MODELS, ModelId } from './lib/groq';
import { useSpeechRecognition } from './hooks/useSpeechRecognition';

// --- Utils ---
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  details?: string;
}

const PROMPT_TEMPLATES = {
  es: `Eres un asistente experto. Responde SIEMPRE en español.
Responde basándote ÚNICAMENTE en el contexto proporcionado.
Sé exhaustivo y completo. Si la pregunta pide listar elementos, personajes, temas o datos, revisa TODO el contexto e incluye TODOS sin omitir ninguno.
Si no encuentras la respuesta en el contexto, dilo claramente.

Contexto:
{context}

Pregunta: {question}

Respuesta detallada:`,
  en: `You are an expert assistant. ALWAYS answer in English.
Answer based ONLY on the provided context.
Be thorough and complete. If the question asks to list elements, characters, topics or data, review ALL the context and include ALL of them without omitting any.
If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Detailed answer:`,
  ru: `Ты — экспертный ассистент. ВСЕГДА отвечай на русском языке.
Отвечай ТОЛЬКО на основе предоставленного контекста.
Будь исчерпывающим и полным. Если вопрос просит перечислить элементы, персонажей, темы или данные, просмотри ВЕСЬ контекст и включи ВСЁ без пропусков.
Если ты не можешь найти ответ в контексте, скажи об этом прямо.

Контекст:
{context}

Вопрос: {question}

Подробный ответ:`
};

// --- Main Component ---
export default function App() {
  const [lang, setLang] = useState<Language>(() => (typeof window !== 'undefined' ? (localStorage.getItem('lang') as Language) || 'es' : 'es'));
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [pdfName, setPdfName] = useState<string | null>(null);
  const [pdfText, setPdfText] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [apiKey, setApiKey] = useState(() => (typeof window !== 'undefined' ? localStorage.getItem('groq_api_key') || '' : ''));
  const [model, setModel] = useState<ModelId>(() => (typeof window !== 'undefined' ? (localStorage.getItem('groq_model') as ModelId) || 'llama-3.3-70b-versatile' : 'llama-3.3-70b-versatile'));
  const [modelProgress, setModelProgress] = useState<{status: string, progress: number} | null>(null);

  const embeddingsRef = useRef<EmbeddingsManager | null>(null);
  const vectorStoreRef = useRef(new VectorStore());
  const chatEndRef = useRef<HTMLDivElement>(null);

  const t = (key: keyof typeof TRANSLATIONS['es']) => TRANSLATIONS[lang][key] || key;
  
  const { isListening, transcript, startListening, stopListening, isSupported: isSpeechSupported } = useSpeechRecognition(
    lang === 'es' ? 'es-ES' : lang === 'ru' ? 'ru-RU' : 'en-US'
  );

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('lang', lang);
      } catch (e) {
        // ignore write errors (e.g., storage disabled)
      }
    }
  }, [lang]);

  useEffect(() => {
    // Keep local copy for local/dev usage only; serverless proxy will use env var in production
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('groq_api_key', apiKey);
      } catch (e) {
        // ignore write errors
      }
    }
  }, [apiKey]);

  useEffect(() => {
    if (typeof window !== 'undefined') {
      try {
        localStorage.setItem('groq_model', model);
      } catch (e) {
        // ignore write errors
      }
    }
  }, [model]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    if (transcript) setInput(transcript);
  }, [transcript]);

  // Load Embeddings Model
  useEffect(() => {
    const initEmbeddings = async () => {
      try {
        const manager = new EmbeddingsManager((progress: {status: string, progress: number}) => {
          setModelProgress(progress);
        });
        await manager.init();
        embeddingsRef.current = manager;
        setModelProgress(null);
      } catch (err) {
        console.error('Failed to init embeddings:', err);
      }
    };
    initEmbeddings();
  }, []);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    try {
      setIsProcessing(true);
      const info = await PDFProcessor.extractText(file);
      setPdfName(file.name);
      setPdfText(info.text);
      setPageCount(info.pageCount);
      setMessages([]);
      
      vectorStoreRef.current.clear();
      
      // If PDF is large, chunk it and create embeddings
      if (info.text.length > 10000) {
        if (!embeddingsRef.current) {
          throw new Error(t('model_not_ready') || 'El modelo de embeddings no está listo todavía. Espera unos segundos.');
        }

        const chunks = PDFProcessor.chunkText(info.text);
        const embeddedChunks: DocumentChunk[] = [];
        
        for (let i = 0; i < chunks.length; i++) {
          const embedding = await embeddingsRef.current.embed(chunks[i]);
          embeddedChunks.push({
            content: chunks[i],
            embedding,
            metadata: {}
          });
        }
        
        vectorStoreRef.current.addChunks(embeddedChunks);
      }
    } catch (err) {
      console.error(err);
      const msg = (err as any)?.message || 'Error processing PDF';
      alert(msg);
    } finally {
      setIsProcessing(false);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false
  });

  const handleSend = async () => {
    if (!input.trim() || isProcessing || !pdfText) return;

    const question = input.trim();
    setInput('');
    const userMessage: Message = { id: Date.now().toString(), role: 'user', content: question };
    setMessages(prev => [...prev, userMessage]);

    setIsProcessing(true);
    try {
      let context = '';
      let strategy = '';

      if (pdfText.length <= 10000) {
        context = pdfText;
        strategy = t('strategy_full');
      } else {
        strategy = t('strategy_rag');
        if (embeddingsRef.current) {
          const queryEmbedding = await embeddingsRef.current.embed(question);
          const results = vectorStoreRef.current.search(queryEmbedding, 5);
          context = results.map(r => r.content).join('\n\n');
        }
      }

      const systemPrompt = PROMPT_TEMPLATES[lang].replace('{context}', context).replace('{question}', question);
                    // Call chatWithGroq without client API key; library proxies to serverless endpoint in browser
                    const response = await chatWithGroq(model, question, systemPrompt);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response,
        details: `**Strategy:** ${strategy} | **Model:** ${MODELS[model]} | **Pages:** ${pageCount}`
      };
      setMessages(prev => [...prev, assistantMessage]);
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [...prev, {
        id: Date.now().toString(),
        role: 'assistant',
        content: `❌ ${t('error')}: ${err.message || 'Unknown error'}`
      }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const clearChat = () => {
    setMessages([]);
    setPdfName(null);
    setPdfText(null);
    setPageCount(0);
    vectorStoreRef.current.clear();
  };

  return (
    <div className="flex h-screen bg-[#1a1a2e] text-slate-200 overflow-hidden font-sans">
      {/* Sidebar / Settings Drawer */}
      <AnimatePresence>
        {isSettingsOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsSettingsOpen(false)}
              className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              className="fixed right-0 top-0 h-full w-80 bg-[#16162a] z-50 p-6 shadow-2xl border-l border-slate-800"
            >
              <div className="flex items-center justify-between mb-8">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <SettingsIcon size={20} className="text-blue-400" />
                  {t('settings')}
                </h2>
                <button onClick={() => setIsSettingsOpen(false)} className="hover:bg-slate-800 p-1 rounded-full transition-colors">
                  <X size={24} />
                </button>
              </div>

              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">{t('model')}</label>
                  <select 
                    value={model} 
                    onChange={(e) => setModel(e.target.value as ModelId)}
                    className="w-full bg-[#2a2a3e] border border-slate-700 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500 outline-none transition-all"
                  >
                    {Object.entries(MODELS).map(([id, label]) => (
                      <option key={id} value={id}>{label}</option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="block text-sm font-medium text-slate-400 mb-2">Groq API Key</label>
                  <input 
                    type="password"
                    value={apiKey}
                    onChange={(e) => setApiKey(e.target.value)}
                    placeholder="gsk_..."
                    className="w-full bg-[#2a2a3e] border border-slate-700 rounded-lg p-2.5 focus:ring-2 focus:ring-blue-500 outline-none transition-all text-sm"
                  />
                  <p className="mt-2 text-xs text-slate-500">
                    Get your key at <a href="https://console.groq.com/keys" target="_blank" rel="noopener noreferrer" className="text-blue-400 hover:underline">console.groq.com</a>
                  </p>
                </div>

                <div className="pt-4 border-t border-slate-800">
                  <button 
                    onClick={clearChat}
                    className="w-full flex items-center justify-center gap-2 bg-red-500/10 hover:bg-red-500/20 text-red-400 py-2.5 rounded-lg transition-colors border border-red-500/30"
                  >
                    <Trash2 size={18} />
                    {t('new_chat')}
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col relative max-w-4xl mx-auto w-full">
        {/* Header */}
        <header className="p-4 flex items-center justify-between border-b border-slate-800/50">
          <div className="flex items-center gap-3">
            <div className="bg-blue-600 p-2 rounded-lg">
              <FileText size={20} className="text-white" />
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight">{t('title')}</h1>
              {pdfName && (
                <div className="flex items-center gap-1.5 text-xs text-green-400">
                  <div className="w-1.5 h-1.5 rounded-full bg-green-500 animate-pulse" />
                  <span className="truncate max-w-[150px]">{pdfName}</span>
                </div>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-[#2a2a3e] rounded-full p-1 border border-slate-700/50">
              {(['es', 'en', 'ru'] as const).map((l) => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={cn(
                    "px-3 py-1 rounded-full text-xs font-bold transition-all",
                    lang === l ? "bg-blue-600 text-white shadow-lg" : "text-slate-500 hover:text-slate-300"
                  )}
                >
                  {l.toUpperCase()}
                </button>
              ))}
            </div>
            <button 
              onClick={() => setIsSettingsOpen(true)}
              className="p-2 hover:bg-slate-800 rounded-lg transition-colors text-slate-400"
            >
              <SettingsIcon size={20} />
            </button>
          </div>
        </header>

        {/* AI Model Loading Progress */}
        {modelProgress && (
          <div className="absolute top-16 left-0 right-0 z-10 p-4">
            <div className="bg-[#2a2a3e] border border-blue-500/30 rounded-xl p-4 shadow-2xl max-w-sm mx-auto">
              <div className="flex items-center justify-between mb-2">
                <span className="text-sm font-medium text-blue-300">{t('downloading_model')}</span>
                <span className="text-xs text-slate-400">{Math.round(modelProgress.progress)}%</span>
              </div>
              <div className="w-full bg-slate-700 rounded-full h-1.5 overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: `${modelProgress.progress}%` }}
                  className="h-full bg-blue-500"
                />
              </div>
            </div>
          </div>
        )}

        {/* Chat Area */}
        <main className="flex-1 overflow-y-auto p-4 space-y-6 scrollbar-thin scrollbar-thumb-slate-700">
          
          {!messages.length && !pdfName && (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              <motion.div 
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="mb-8 p-6 bg-blue-600/10 rounded-full border border-blue-500/20"
              >
                <FileText size={64} className="text-blue-500" />
              </motion.div>
              <h2 className="text-2xl font-bold mb-2">{t('greeting')}</h2>
              <p className="text-slate-400 max-w-md mb-8">{t('greeting_sub')}</p>
                  <div className="flex justify-center mb-6">
                <div {...getRootProps()} className={cn(
                  "w-full max-w-md p-6 border-2 border-dashed rounded-2xl transition-all cursor-pointer",
                  isDragActive ? "border-blue-500 bg-blue-500/5" : "border-slate-700 hover:border-slate-600 bg-slate-800/20"
                )}>
                  <input {...getInputProps()} />
                  <div className="flex flex-col items-center gap-3">
                    <Plus size={28} className="text-slate-500" />
                    <p className="text-sm text-slate-400">
                      {pdfName ? `Cambiar PDF (${pdfName})` : t('no_pdf_yet')}
                    </p>
                  </div>
                </div>
              </div>           
            </div>
          )}

          <div className="max-w-3xl mx-auto w-full space-y-6 pb-20">
            {messages.map((msg) => (
              <motion.div
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                key={msg.id}
                className={cn(
                  "flex flex-col",
                  msg.role === 'user' ? "items-end" : "items-start"
                )}
              >
                <div className={cn(
                  "max-w-[85%] rounded-2xl p-4 shadow-md",
                  msg.role === 'user' 
                    ? "bg-blue-600 text-white rounded-tr-none" 
                    : "bg-[#2a2a3e] border border-slate-700/50 rounded-tl-none"
                )}>
                  <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{msg.content}</p>
                </div>
                {msg.details && (
                  <div className="mt-2 text-[11px] text-slate-500 flex items-center gap-2 px-1">
                    <AlertCircle size={12} />
                    <span>{msg.details}</span>
                  </div>
                )}
              </motion.div>
            ))}
            {isProcessing && (
              <div className="flex items-center gap-3 text-slate-400">
                <Loader2 size={18} className="animate-spin text-blue-500" />
                <span className="text-sm italic">{t('processing')}</span>
              </div>
            )}
            <div ref={chatEndRef} />
          </div>
        </main>

        {/* Input Bar */}
        <div className="absolute bottom-0 left-0 right-0 p-4 bg-gradient-to-t from-[#1a1a2e] via-[#1a1a2e] to-transparent">
          <div className="max-w-3xl mx-auto">
            {!pdfName && messages.length > 0 && (
              <div className="mb-4 p-3 bg-blue-500/10 border border-blue-500/20 rounded-xl flex items-center gap-3">
                <AlertCircle size={18} className="text-blue-400 shrink-0" />
                <p className="text-sm text-blue-200">{t('no_pdf_yet')}</p>
              </div>
            )}

            <div className="relative group">
              <div className="absolute left-4 top-1/2 -translate-y-1/2 flex items-center gap-2">
                {isSpeechSupported && (
                  <button 
                    onClick={isListening ? stopListening : startListening}
                    className={cn(
                      "p-2 rounded-full transition-all",
                      isListening ? "bg-red-500 text-white animate-pulse" : "text-slate-400 hover:text-white"
                    )}
                  >
                    {isListening ? <MicOff size={20} /> : <Mic size={20} />}
                  </button>
                )}
              </div>

              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={pdfName ? t('chat_placeholder') : t('no_pdf_yet')}
                disabled={!pdfName || isProcessing}
                className={cn(
                  "w-full bg-[#2a2a3e] border border-slate-700/50 rounded-2xl py-4 pl-14 pr-16 min-h-[56px] max-h-32 outline-none focus:ring-2 focus:ring-blue-500/50 transition-all resize-none shadow-xl",
                  !pdfName && "opacity-60 cursor-not-allowed"
                )}
                rows={1}
              />

              <button
                onClick={handleSend}
                disabled={!input.trim() || isProcessing || !pdfName}
                className={cn(
                  "absolute right-3 top-1/2 -translate-y-1/2 p-2.5 rounded-xl transition-all",
                  input.trim() && pdfName && !isProcessing
                    ? "bg-blue-600 text-white hover:bg-blue-500 shadow-lg shadow-blue-500/20"
                    : "bg-slate-700 text-slate-500 cursor-not-allowed"
                )}
              >
                {isProcessing ? <Loader2 size={20} className="animate-spin" /> : <Send size={20} />}
              </button>
            </div>
            
            <p className="text-center text-[10px] text-slate-500 mt-3 uppercase tracking-widest font-bold">
              Powered by Groq & Llama 3
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
