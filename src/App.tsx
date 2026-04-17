import { useState, useEffect, useRef, useCallback } from 'react';
import { 
  FileText, 
  Send, 
  Mic, 
  Settings, 
  Plus, 
  Trash2, 
  Loader2, 
  X
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { useDropzone } from 'react-dropzone';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { PDFProcessor } from './lib/pdf-processor';
import { VectorStore } from './lib/vector-store';
import { EmbeddingsManager } from './lib/embeddings';
import { TRANSLATIONS, Language } from './lib/translations';
import { callGroq } from './lib/groq';

// Utility for tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// --- Types ---
interface Message {
  role: 'user' | 'assistant';
  content: string;
  details?: {
    strategy: string;
    chunks: number;
    pages: number;
    model: string;
  };
}

export default function App() {
  // --- State ---
  const [lang, setLang] = useState<Language>('es');
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  const [isLoadingModel, setIsLoadingModel] = useState(true);
  const [pdfName, setPdfName] = useState<string | null>(null);
  const [pdfPages, setPdfPages] = useState<string[]>([]);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('groq_api_key') || '');
  const [model, setModel] = useState('llama-3.3-70b-versatile');
  
  // RAG references
  const vectorStoreRef = useRef<VectorStore>(new VectorStore());
  const chatEndRef = useRef<HTMLDivElement>(null);

  const t = (key: keyof typeof TRANSLATIONS['es']) => TRANSLATIONS[lang][key] || key;

  // --- Initialization ---
  useEffect(() => {
    const initModel = async () => {
      try {
        await EmbeddingsManager.getInstance().init();
        setIsLoadingModel(false);
      } catch (err) {
        console.error("Failed to load embeddings model:", err);
      }
    };
    initModel();
  }, []);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // --- Handlers ---
  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file) return;

    setIsProcessing(true);
    try {
      const { pages } = await PDFProcessor.extractText(file);
      setPdfName(file.name);
      setPdfPages(pages);
      
      const chunks = PDFProcessor.chunkText(pages);
      vectorStoreRef.current.clear();
      await vectorStoreRef.current.addDocuments(chunks);
      
      setMessages([]); // Clear chat for new PDF
    } catch (err) {
      console.error(err);
      alert(t('error'));
    } finally {
      setIsProcessing(false);
    }
  }, [lang]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false
  });

  const handleSend = async (textOverride?: string) => {
    const text = textOverride || input;
    if (!text.trim() || isProcessing) return;
    if (!apiKey) {
      alert(t('api_key_required'));
      setIsSidebarOpen(true);
      return;
    }

    const userMsg: Message = { role: 'user', content: text };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setIsProcessing(true);

    try {
      let context = "";
      let strategy = "";
      const chunks_k = 5;

      // Simple RAG logic
      if (pdfPages.length > 0) {
        // If it's a small PDF, use everything
        const totalChars = pdfPages.join('').length;
        if (totalChars < 10000) {
          context = pdfPages.join('\n\n');
          strategy = t('strategy_full');
        } else {
          const relevantChunks = await vectorStoreRef.current.similaritySearch(text, chunks_k);
          context = relevantChunks.map(c => c.pageContent).join('\n\n');
          strategy = t('strategy_rag');
        }
      }

      const promptTemplate = `
        You are an expert assistant. ANSWER ALWAYS IN ${lang.toUpperCase()}.
        Answer based ONLY on the provided context.
        Be thorough and complete. If the question asks to list elements, characters, topics or data, review ALL the context and include ALL of them without omitting any.
        If you cannot find the answer in the context, say so clearly.

        Context:
        ${context || "No context provided."}

        Question: ${text}

        Detailed answer:
      `;

      const response = await callGroq([{ role: 'user', content: promptTemplate }], apiKey, model);
      
      const assistantMsg: Message = { 
        role: 'assistant', 
        content: response,
        details: {
          strategy,
          chunks: vectorStoreRef.current.totalChunks,
          pages: pdfPages.length,
          model
        }
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (err: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `❌ Error: ${err.message}` }]);
    } finally {
      setIsProcessing(false);
    }
  };

  const handleVoiceInput = () => {
    const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognition) {
      alert(t('mic_error'));
      return;
    }

    const recognition = new SpeechRecognition();
    recognition.lang = lang === 'es' ? 'es-ES' : lang === 'ru' ? 'ru-RU' : 'en-US';
    recognition.start();

    recognition.onresult = (event: any) => {
      const transcript = event.results[0][0].transcript;
      setInput(transcript);
      // Optional: auto-send
      // handleSend(transcript);
    };
  };

  const clearAll = () => {
    setPdfName(null);
    setPdfPages([]);
    setMessages([]);
    vectorStoreRef.current.clear();
  };

  const saveApiKey = (val: string) => {
    setApiKey(val);
    localStorage.setItem('groq_api_key', val);
  };

  return (
    <div className="flex h-screen bg-[#0f111a] text-gray-200 overflow-hidden font-sans">
      {/* --- Sidebar (Settings) --- */}
      <AnimatePresence>
        {isSidebarOpen && (
          <>
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              onClick={() => setIsSidebarOpen(false)}
              className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm"
            />
            <motion.div 
              initial={{ x: '100%' }}
              animate={{ x: 0 }}
              exit={{ x: '100%' }}
              className="fixed right-0 top-0 h-full w-80 bg-[#1a1c2e] z-50 p-6 shadow-2xl border-l border-white/10"
            >
              <div className="flex justify-between items-center mb-8">
                <h2 className="text-xl font-bold flex items-center gap-2">
                  <Settings className="w-5 h-5" /> {t('config')}
                </h2>
                <button onClick={() => setIsSidebarOpen(false)} className="p-2 hover:bg-white/5 rounded-full transition-colors">
                  <X className="w-5 h-5" />
                </button>
              </div>

              <div className="space-y-6">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">{t('groq_key')}</label>
                  <input 
                    type="password" 
                    value={apiKey}
                    onChange={(e) => saveApiKey(e.target.value)}
                    placeholder="gsk_..."
                    className="w-full bg-[#0f111a] border border-white/10 rounded-lg px-4 py-2 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">{t('model')}</label>
                  <select 
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="w-full bg-[#0f111a] border border-white/10 rounded-lg px-4 py-2 focus:outline-none"
                  >
                    <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                    <option value="llama3-8b-8192">Llama 3 8B</option>
                    <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
                  </select>
                </div>

                <div className="pt-4">
                  <button 
                    onClick={clearAll}
                    className="w-full flex items-center justify-center gap-2 py-2 bg-red-500/10 text-red-400 rounded-lg hover:bg-red-500/20 transition-colors border border-red-500/20"
                  >
                    <Trash2 className="w-4 h-4" /> {t('clear_cache')}
                  </button>
                </div>
              </div>
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* --- Main Content --- */}
      <div className="flex-1 flex flex-col relative max-w-5xl mx-auto w-full px-4 sm:px-6">
        
        {/* Header */}
        <header className="py-4 flex items-center justify-between border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center shadow-lg shadow-blue-600/20">
              <FileText className="text-white w-6 h-6" />
            </div>
            <div>
              <h1 className="font-bold text-lg leading-tight">Agente PDF</h1>
              <div className="flex items-center gap-2">
                <span className={cn("w-2 h-2 rounded-full", pdfName ? "bg-green-500" : "bg-gray-600")}></span>
                <span className="text-xs text-gray-500 font-medium">
                  {pdfName ? pdfName : t('no_pdf_yet')}
                </span>
              </div>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <div className="flex bg-white/5 p-1 rounded-lg">
              {(['es', 'en', 'ru'] as Language[]).map(l => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={cn(
                    "px-3 py-1 rounded text-xs font-bold transition-all",
                    lang === l ? "bg-blue-600 text-white shadow-md" : "text-gray-400 hover:text-gray-200"
                  )}
                >
                  {l.toUpperCase()}
                </button>
              ))}
            </div>
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 hover:bg-white/5 rounded-lg transition-colors text-gray-400 hover:text-white"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Chat Messages */}
        <main className="flex-1 overflow-y-auto py-8 space-y-6 scrollbar-thin scrollbar-thumb-white/10">
          {messages.length === 0 ? (
            <div className="h-full flex flex-col items-center justify-center text-center px-4">
              <motion.div 
                initial={{ scale: 0.9, opacity: 0 }}
                animate={{ scale: 1, opacity: 1 }}
                className="mb-6 w-20 h-20 bg-blue-600/10 rounded-full flex items-center justify-center"
              >
                <FileText className="w-10 h-10 text-blue-500" />
              </motion.div>
              <h2 className="text-2xl font-bold text-white mb-2">{t('greeting')}</h2>
              <p className="text-gray-500 max-w-sm mb-8">{t('greeting_sub')}</p>
              
              {!pdfName && (
                <div {...getRootProps()} className={cn(
                  "w-full max-w-md border-2 border-dashed rounded-2xl p-8 transition-all cursor-pointer",
                  isDragActive ? "border-blue-500 bg-blue-500/10" : "border-white/10 hover:border-white/20 bg-white/5"
                )}>
                  <input {...getInputProps()} />
                  <Plus className="w-8 h-8 text-gray-400 mx-auto mb-4" />
                  <p className="font-medium">{t('attach_pdf')}</p>
                  <p className="text-xs text-gray-500 mt-2">PDF files up to 20MB</p>
                </div>
              )}
            </div>
          ) : (
            <div className="space-y-6 pb-20">
              {messages.map((msg, i) => (
                <motion.div 
                  initial={{ y: 10, opacity: 0 }}
                  animate={{ y: 0, opacity: 1 }}
                  key={i} 
                  className={cn(
                    "flex flex-col max-w-[85%]",
                    msg.role === 'user' ? "ml-auto items-end" : "mr-auto items-start"
                  )}
                >
                  <div className={cn(
                    "px-4 py-3 rounded-2xl text-sm leading-relaxed",
                    msg.role === 'user' 
                      ? "bg-blue-600 text-white rounded-tr-none shadow-lg shadow-blue-600/10" 
                      : "bg-[#1a1c2e] text-gray-200 border border-white/5 rounded-tl-none"
                  )}>
                    {msg.content}
                  </div>
                  
                  {msg.details && (
                    <div className="mt-2 text-[10px] text-gray-500 flex gap-3 px-2">
                      <span>{msg.details.strategy}</span>
                      <span>•</span>
                      <span>{msg.details.model}</span>
                      <span>•</span>
                      <span>{msg.details.pages} pgs</span>
                    </div>
                  )}
                </motion.div>
              ))}
              {isProcessing && (
                <div className="flex items-center gap-3 text-gray-500 text-sm animate-pulse">
                  <div className="w-6 h-6 bg-[#1a1c2e] rounded-full flex items-center justify-center">
                    <Loader2 className="w-3 h-3 animate-spin" />
                  </div>
                  {t('processing')}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </main>

        {/* Chat Input Bar */}
        <div className="absolute bottom-6 left-0 right-0 px-4 sm:px-6">
          <div className="max-w-4xl mx-auto">
            {isLoadingModel ? (
              <div className="bg-[#1a1c2e] border border-white/5 rounded-2xl p-4 flex items-center justify-center gap-3 text-sm text-gray-400">
                <Loader2 className="w-4 h-4 animate-spin text-blue-500" />
                Loading AI components...
              </div>
            ) : (
              <div className="relative group">
                <div className="absolute -inset-0.5 bg-gradient-to-r from-blue-600 to-indigo-600 rounded-2xl opacity-20 group-focus-within:opacity-40 transition-opacity blur"></div>
                <div className="relative flex items-center bg-[#1a1c2e] border border-white/10 rounded-2xl p-2 pl-4 shadow-xl">
                  {pdfName && (
                    <div {...getRootProps()} className="mr-2 p-2 hover:bg-white/5 rounded-xl transition-colors cursor-pointer group/upload">
                      <input {...getInputProps()} />
                      <Plus className="w-5 h-5 text-gray-400 group-hover/upload:text-blue-500" />
                    </div>
                  )}
                  
                  <textarea
                    rows={1}
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
                    className="flex-1 bg-transparent border-none focus:ring-0 text-sm py-2 resize-none placeholder-gray-600 disabled:opacity-50"
                  />
                  
                  <div className="flex items-center gap-1">
                    <button 
                      onClick={handleVoiceInput}
                      disabled={!pdfName || isProcessing}
                      className="p-2 hover:bg-white/5 rounded-xl transition-colors text-gray-400 hover:text-red-500 disabled:opacity-30"
                    >
                      <Mic className="w-5 h-5" />
                    </button>
                    <button 
                      onClick={() => handleSend()}
                      disabled={!input.trim() || !pdfName || isProcessing}
                      className="p-2 bg-blue-600 hover:bg-blue-500 disabled:bg-gray-800 disabled:text-gray-600 text-white rounded-xl transition-all shadow-lg shadow-blue-600/20"
                    >
                      <Send className="w-5 h-5" />
                    </button>
                  </div>
                </div>
              </div>
            )}
            <p className="text-[10px] text-center text-gray-600 mt-3 font-medium tracking-wide uppercase">
              Powered by Groq • Client-side RAG
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
