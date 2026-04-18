import { useState, useEffect, useRef, useCallback } from 'react';
import { AnimatePresence, motion } from 'framer-motion';
import {
  Settings as SettingsIcon,
  Send,
  Mic,
  MicOff,
  Sparkles,
  FileText,
  AlertCircle,
  Loader2,
  Bot,
} from 'lucide-react';
import { clsx } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { PDFProcessor } from './lib/pdf-processor';
import { EmbeddingsManager } from './lib/embeddings';
import { VectorStore, DocumentChunk } from './lib/vector-store';
import { TRANSLATIONS, Language } from './lib/translations';
import { chatWithGroq, MODELS, ModelId } from './lib/groq';
import { useSpeechRecognition } from './hooks/useSpeechRecognition';
import { MessageBubble } from './components/MessageBubble';
import { TypingIndicator } from './components/TypingIndicator';
import { PDFDropzone } from './components/PDFDropzone';
import { SettingsDrawer } from './components/SettingsDrawer';

function cn(...inputs: any[]) {
  return twMerge(clsx(inputs));
}

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  details?: string;
  tokens?: number;
  timestamp: number;
}

const PROMPT_TEMPLATES: Record<Language, string> = {
  es: `Eres DocuMind AI, un asistente experto en análisis de documentos. Responde SIEMPRE en español.
Responde basándote ÚNICAMENTE en el contexto proporcionado del PDF.
Sé exhaustivo y completo. Usa formato Markdown cuando sea útil (listas, negrita, encabezados).
Si no encuentras la respuesta en el contexto, dilo claramente.

Contexto del documento:
{context}

Pregunta: {question}`,

  en: `You are DocuMind AI, an expert document analysis assistant. ALWAYS answer in English.
Answer based ONLY on the provided PDF context.
Be thorough and complete. Use Markdown formatting when helpful (lists, bold, headings).
If you cannot find the answer in the context, say so clearly.

Document context:
{context}

Question: {question}`,

  ru: `Ты — DocuMind AI, экспертный ассистент по анализу документов. ВСЕГДА отвечай на русском.
Отвечай ТОЛЬКО на основе предоставленного контекста PDF.
Будь исчерпывающим. Используй Markdown-форматирование там, где это полезно.
Если ты не можешь найти ответ в контексте, скажи об этом прямо.

Контекст документа:
{context}

Вопрос: {question}`,
};

const EXAMPLE_QUESTIONS: Record<Language, string[]> = {
  es: ['¿De qué trata este documento?', '¿Cuáles son los puntos principales?', 'Haz un resumen ejecutivo', '¿Qué conclusiones se presentan?'],
  en: ['What is this document about?', 'What are the main points?', 'Give me an executive summary', 'What conclusions are presented?'],
  ru: ['О чём этот документ?', 'Каковы основные тезисы?', 'Сделай краткое резюме', 'Какие выводы представлены?'],
};

function ls(key: string, fallback: string): string {
  try { return localStorage.getItem(key) || fallback; } catch { return fallback; }
}
function lsSet(key: string, value: string) {
  try { localStorage.setItem(key, value); } catch { /* ignore */ }
}

export default function App() {
  // Persistent state
  const [lang, setLang] = useState<Language>(() => ls('lang', 'es') as Language);
  const [apiKey, setApiKey] = useState(() => ls('groq_api_key', ''));
  const [model, setModel] = useState<ModelId>(() => ls('groq_model', 'llama-3.3-70b-versatile') as ModelId);
  const [apiKeySaved, setApiKeySaved] = useState(!!ls('groq_api_key', ''));

  // UI state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isTyping, setIsTyping] = useState(false);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [modelProgress, setModelProgress] = useState<{ status: string; progress: number } | null>(null);

  // PDF state
  const [pdfName, setPdfName] = useState<string | null>(null);
  const [pdfText, setPdfText] = useState<string | null>(null);
  const [pageCount, setPageCount] = useState(0);
  const [charCount, setCharCount] = useState(0);

  // Refs
  const embeddingsRef = useRef<EmbeddingsManager | null>(null);
  const vectorStoreRef = useRef(new VectorStore());
  const chatEndRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const t = (key: string): string => (TRANSLATIONS[lang] as any)[key] ?? key;

  const { isListening, transcript, startListening, stopListening, isSupported: isSpeechSupported, reset: resetTranscript }
    = useSpeechRecognition(lang === 'es' ? 'es-ES' : lang === 'ru' ? 'ru-RU' : 'en-US');

  // Sync prefs
  useEffect(() => { lsSet('lang', lang); }, [lang]);
  useEffect(() => {
    lsSet('groq_api_key', apiKey);
    setApiKeySaved(!!apiKey);
  }, [apiKey]);
  useEffect(() => { lsSet('groq_model', model); }, [model]);

  // Auto-scroll
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isTyping]);

  // Voice → input
  useEffect(() => {
    if (transcript) setInput(transcript);
  }, [transcript]);

  // Auto-resize textarea
  useEffect(() => {
    const ta = textareaRef.current;
    if (ta) {
      ta.style.height = 'auto';
      ta.style.height = `${Math.min(ta.scrollHeight, 160)}px`;
    }
  }, [input]);

  // Init embeddings
  useEffect(() => {
    const init = async () => {
      try {
        const mgr = new EmbeddingsManager((p) => setModelProgress(p));
        await mgr.init();
        embeddingsRef.current = mgr;
        setModelProgress(null);
      } catch (e) {
        console.error('Embeddings init failed:', e);
        setModelProgress(null);
      }
    };
    init();
  }, []);

  const onDrop = useCallback(async (files: File[]) => {
    const file = files[0];
    if (!file) return;
    setIsProcessing(true);
    try {
      const info = await PDFProcessor.extractText(file);
      setPdfName(file.name);
      setPdfText(info.text);
      setPageCount(info.pageCount);
      setCharCount(info.text.length);
      setMessages([]);
      vectorStoreRef.current.clear();

      if (info.text.length > 8000) {
        if (!embeddingsRef.current?.ready) {
          throw new Error(t('model_not_ready'));
        }
        const chunks = PDFProcessor.chunkText(info.text, 800, 150);
        embeddingsRef.current.buildCorpus(chunks);
        const embeddedChunks: DocumentChunk[] = chunks.map((c) => ({
          content: c,
          embedding: embeddingsRef.current!.embed(c),
          metadata: {},
        }));
        vectorStoreRef.current.addChunks(embeddedChunks);
      }
    } catch (err: any) {
      alert(err?.message || 'Error processing PDF');
    } finally {
      setIsProcessing(false);
    }
  }, [lang]);

  const clearAll = () => {
    setMessages([]);
    setPdfName(null);
    setPdfText(null);
    setPageCount(0);
    setCharCount(0);
    vectorStoreRef.current.clear();
  };

  const handleSend = async (overrideInput?: string) => {
    const question = (overrideInput ?? input).trim();
    if (!question || isProcessing || !pdfText) return;

    setInput('');
    resetTranscript();
    if (isListening) stopListening();

    const userMsg: Message = {
      id: `u-${Date.now()}`,
      role: 'user',
      content: question,
      timestamp: Date.now(),
    };
    setMessages((prev) => [...prev, userMsg]);
    setIsTyping(true);

    try {
      let context = '';
      let strategy = '';

      if (pdfText.length <= 8000) {
        context = pdfText;
        strategy = t('strategy_full');
      } else {
        strategy = t('strategy_rag');
        if (embeddingsRef.current) {
          const qEmb = embeddingsRef.current.embed(question);
          const results = vectorStoreRef.current.search(qEmb, 6);
          context = results.map((r) => r.content).join('\n\n---\n\n');
        }
      }

      const systemPrompt = PROMPT_TEMPLATES[lang]
        .replace('{context}', context)
        .replace('{question}', question);

      const { content, totalTokens } = await chatWithGroq(model, question, systemPrompt, apiKey);

      const assistantMsg: Message = {
        id: `a-${Date.now()}`,
        role: 'assistant',
        content,
        details: `Strategy: ${strategy} · Model: ${MODELS[model]} · Pages: ${pageCount}`,
        tokens: totalTokens,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, assistantMsg]);
    } catch (err: any) {
      const errMsg: Message = {
        id: `e-${Date.now()}`,
        role: 'assistant',
        content: `❌ **${t('error')}:** ${err.message || 'Unknown error'}`,
        timestamp: Date.now(),
      };
      setMessages((prev) => [...prev, errMsg]);
    } finally {
      setIsTyping(false);
    }
  };

  const examples = EXAMPLE_QUESTIONS[lang];
  const hasMessages = messages.length > 0;
  const canSend = !!input.trim() && !isProcessing && !isTyping && !!pdfText;

  return (
    <div className="flex h-screen overflow-hidden" style={{ background: 'var(--bg-primary)' }}>
      {/* Settings Drawer */}
      <AnimatePresence>
        {isSettingsOpen && (
          <SettingsDrawer
            isOpen={isSettingsOpen}
            onClose={() => setIsSettingsOpen(false)}
            apiKey={apiKey}
            onApiKeyChange={setApiKey}
            model={model}
            onModelChange={setModel}
            lang={lang}
            onLangChange={setLang}
            onClearChat={clearAll}
            t={t}
            apiKeySaved={apiKeySaved}
          />
        )}
      </AnimatePresence>

      {/* Main layout */}
      <div className="flex flex-col flex-1 max-w-4xl mx-auto w-full min-h-0">

        {/* ── HEADER ── */}
        <header className="flex-none flex items-center justify-between px-5 py-3.5 border-b border-indigo-900/30 bg-[#0f0f1a]/80 backdrop-blur-xl">
          <div className="flex items-center gap-3">
            {/* Logo */}
            <div className="relative">
              <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-indigo-500 to-purple-600 flex items-center justify-center shadow-lg shadow-indigo-500/25">
                <Bot size={18} className="text-white" />
              </div>
              <span className="absolute -bottom-0.5 -right-0.5 w-2.5 h-2.5 rounded-full bg-emerald-400 border-2 border-[#0f0f1a]" />
            </div>
            <div>
              <h1 className="font-bold text-base leading-tight bg-gradient-to-r from-indigo-300 to-purple-300 bg-clip-text text-transparent">
                DocuMind AI
              </h1>
              <p className="text-[10px] text-slate-600 leading-none">{t('subtitle')}</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {/* Lang switcher */}
            <div className="flex items-center bg-[#161628] border border-indigo-900/40 rounded-full p-0.5">
              {(['es', 'en', 'ru'] as Language[]).map((l) => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={cn(
                    'px-2.5 py-1 rounded-full text-[11px] font-bold transition-all',
                    lang === l
                      ? 'bg-indigo-600 text-white shadow'
                      : 'text-slate-600 hover:text-slate-300'
                  )}
                >
                  {l.toUpperCase()}
                </button>
              ))}
            </div>

            {/* Settings */}
            <button
              onClick={() => setIsSettingsOpen(true)}
              className={cn(
                'relative p-2 rounded-xl transition-all border',
                'border-indigo-900/40 text-slate-400 hover:bg-indigo-900/30 hover:text-white'
              )}
            >
              <SettingsIcon size={17} />
            </button>
          </div>
        </header>

        {/* ── MODEL PROGRESS ── */}
        <AnimatePresence>
          {modelProgress && (
            <motion.div
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              className="flex-none overflow-hidden"
            >
              <div className="px-5 py-2 bg-indigo-950/50 border-b border-indigo-900/30">
                <div className="flex items-center justify-between mb-1.5">
                  <span className="text-[11px] text-indigo-400 font-medium">{t('downloading_model')}</span>
                  <span className="text-[11px] text-slate-500">{Math.round(modelProgress.progress)}%</span>
                </div>
                <div className="h-1 bg-slate-800 rounded-full overflow-hidden">
                  <motion.div
                    className="h-full bg-gradient-to-r from-indigo-500 to-purple-500 rounded-full"
                    animate={{ width: `${modelProgress.progress}%` }}
                  />
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>

        {/* ── CHAT AREA ── */}
        <main className="flex-1 overflow-y-auto min-h-0 px-4 py-5 space-y-5">

          {/* PDF Upload Zone (always visible at top) */}
          <div className="max-w-lg mx-auto">
            {isProcessing ? (
              <div className="flex items-center justify-center gap-3 p-6 bg-[#1e1e35] border border-indigo-500/20 rounded-2xl">
                <Loader2 size={20} className="animate-spin text-indigo-400" />
                <span className="text-sm text-slate-400">{t('uploading')}</span>
              </div>
            ) : (
              <PDFDropzone
                onDrop={onDrop}
                pdfName={pdfName}
                pageCount={pageCount}
                charCount={charCount}
                isProcessing={isProcessing}
                onClear={clearAll}
                noFileLabel={t('no_pdf_yet')}
                changeLabel={t('change_pdf')}
                pagesLabel={t('pages')}
                charsLabel={t('chars')}
                dragActiveLabel={t('drop_active')}
              />
            )}
          </div>

          {/* Welcome / Empty state */}
          {!hasMessages && !isProcessing && (
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.1 }}
              className="flex flex-col items-center justify-center py-10 px-4 text-center"
            >
              {!pdfName ? (
                <>
                  <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-indigo-600/20 to-purple-600/20 border border-indigo-500/20 flex items-center justify-center mb-5">
                    <FileText size={36} className="text-indigo-400" />
                  </div>
                  <h2 className="text-2xl font-bold text-slate-200 mb-2">{t('greeting')}</h2>
                  <p className="text-slate-500 max-w-sm text-[15px] leading-relaxed">{t('greeting_sub')}</p>
                </>
              ) : (
                <>
                  <div className="w-20 h-20 rounded-3xl bg-gradient-to-br from-indigo-600/20 to-purple-600/20 border border-indigo-500/20 flex items-center justify-center mb-5">
                    <Sparkles size={36} className="text-indigo-400" />
                  </div>
                  <h2 className="text-xl font-bold text-slate-200 mb-1">{t('upload_success')}</h2>
                  <p className="text-slate-500 text-sm mb-6">{t('pdf_loaded')} · {pageCount} {t('pages')}</p>

                  {/* Example questions */}
                  <div className="grid grid-cols-2 gap-2 max-w-md w-full">
                    {examples.map((q, i) => (
                      <button
                        key={i}
                        onClick={() => handleSend(q)}
                        className="text-left px-3 py-2.5 rounded-xl bg-[#1e1e35] border border-indigo-900/40 text-sm text-slate-400 hover:border-indigo-500/50 hover:text-indigo-300 hover:bg-indigo-900/20 transition-all"
                      >
                        <span className="text-indigo-600 font-bold mr-1.5">→</span>
                        {q}
                      </button>
                    ))}
                  </div>
                </>
              )}
            </motion.div>
          )}

          {/* Messages */}
          <div className="max-w-3xl mx-auto w-full space-y-5 pb-2">
            <AnimatePresence initial={false}>
              {messages.map((msg) => (
                <motion.div
                  key={msg.id}
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, scale: 0.98 }}
                  transition={{ duration: 0.2 }}
                >
                  <MessageBubble
                    message={msg}
                    copyLabel={t('copy')}
                  />
                </motion.div>
              ))}
            </AnimatePresence>

            {isTyping && (
              <motion.div
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
              >
                <TypingIndicator />
              </motion.div>
            )}

            <div ref={chatEndRef} />
          </div>
        </main>

        {/* ── INPUT BAR ── */}
        <div className="flex-none px-4 py-3 border-t border-indigo-900/30 bg-[#0f0f1a]/90 backdrop-blur-xl">
          <div className="max-w-3xl mx-auto">
            <div className={cn(
              'flex items-end gap-2 px-3 py-2 rounded-2xl border transition-all duration-200',
              'bg-[#1e1e35]',
              !pdfText
                ? 'border-slate-800/50 opacity-60'
                : 'border-indigo-900/50 focus-within:border-indigo-500/60 focus-within:shadow-lg focus-within:shadow-indigo-500/10'
            )}>
              {/* Voice button */}
              {isSpeechSupported && (
                <button
                  onClick={isListening ? stopListening : startListening}
                  disabled={!pdfText}
                  className={cn(
                    'p-2 rounded-xl transition-all shrink-0 mb-0.5',
                    isListening
                      ? 'bg-red-500/20 text-red-400 border border-red-500/40 animate-pulse'
                      : 'text-slate-600 hover:text-slate-300 hover:bg-slate-800/60'
                  )}
                >
                  {isListening ? <MicOff size={18} /> : <Mic size={18} />}
                </button>
              )}

              {/* Textarea */}
              <textarea
                ref={textareaRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={pdfText ? t('chat_placeholder') : t('no_pdf_yet')}
                disabled={!pdfText || isTyping}
                rows={1}
                className="flex-1 bg-transparent outline-none resize-none text-[15px] placeholder-slate-700 text-slate-200 py-1.5 min-h-[36px] max-h-[160px] disabled:cursor-not-allowed"
              />

              {/* Send button */}
              <button
                onClick={() => handleSend()}
                disabled={!canSend}
                className={cn(
                  'p-2 rounded-xl transition-all shrink-0 mb-0.5',
                  canSend
                    ? 'bg-indigo-600 hover:bg-indigo-500 text-white shadow-lg shadow-indigo-500/25'
                    : 'bg-slate-800/60 text-slate-700 cursor-not-allowed'
                )}
              >
                {isTyping
                  ? <Loader2 size={18} className="animate-spin" />
                  : <Send size={18} />}
              </button>
            </div>

            <p className="text-center text-[10px] text-slate-700 mt-2 tracking-widest uppercase font-semibold">
              DocuMind AI · Powered by Groq & Llama 3 · Press Enter to send
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
