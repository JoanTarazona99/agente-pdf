import React, { useState, useEffect, useRef } from 'react';
import { 
  FileText, 
  Send, 
  Settings, 
  Mic, 
  MicOff, 
  Trash2, 
  ChevronRight,
  Loader2,
  AlertCircle,
  CheckCircle2,
  Search,
  Cpu,
  Database,
  Info
} from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

import { PDFProcessor } from './lib/pdf-processor';
import { EmbeddingsManager } from './lib/embeddings';
import { VectorStore } from './lib/vector-store';
import { TRANSLATIONS, type Language } from './lib/translations';

// Utility for merging tailwind classes
function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

// Types
interface Message {
  role: 'user' | 'assistant';
  content: string;
  details?: {
    model: string;
    strategy: string;
    chunks: number;
    pages: number;
    chars: number;
  };
}

// Speech Recognition Type Helper
type SpeechRecognition = any;
const SpeechRecognitionAPI = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;

export default function App() {
  // State
  const [lang, setLang] = useState<Language>('es');
  const [messages, setMessages] = useState<Message[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [pdfName, setPdfName] = useState<string | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [apiKey, setApiKey] = useState(() => localStorage.getItem('groq_api_key') || '');
  const [selectedModel, setSelectedModel] = useState('llama-3.3-70b-versatile');
  const [modelProgress, setModelProgress] = useState<number | null>(null);
  const [isRecording, setIsRecording] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Refs
  const vectorStoreRef = useRef<VectorStore | null>(null);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const recognitionRef = useRef<SpeechRecognition | null>(null);

  const t = (key: keyof typeof TRANSLATIONS['es']) => TRANSLATIONS[lang][key] || key;

  // Scroll to bottom on messages change
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  // Sync API Key to localStorage
  useEffect(() => {
    localStorage.setItem('groq_api_key', apiKey);
  }, [apiKey]);

  // Initialize Speech Recognition
  useEffect(() => {
    if (SpeechRecognitionAPI) {
      const recognition = new SpeechRecognitionAPI();
      recognition.continuous = false;
      recognition.interimResults = false;
      recognition.lang = lang === 'es' ? 'es-ES' : lang === 'en' ? 'en-US' : 'ru-RU';

      recognition.onresult = (event: any) => {
        const transcript = event.results[0][0].transcript;
        setInputValue(prev => (prev ? prev + ' ' : '') + transcript);
        setIsRecording(false);
      };

      recognition.onerror = () => setIsRecording(false);
      recognition.onend = () => setIsRecording(false);

      recognitionRef.current = recognition;
    }
  }, [lang]);

  const toggleRecording = () => {
    if (!recognitionRef.current) {
      alert(t('mic_error'));
      return;
    }

    if (isRecording) {
      recognitionRef.current.stop();
    } else {
      recognitionRef.current.start();
      setIsRecording(true);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || file.type !== 'application/pdf') return;

    setIsProcessing(true);
    setError(null);
    setPdfName(file.name);
    setMessages([]);

    try {
      const { pages } = await PDFProcessor.extractText(file);
      
      const chunks = PDFProcessor.chunkText(pages, 1000, 200);

      const embeddingsManager = EmbeddingsManager.getInstance();
      
      // Initialize model and show progress
      await embeddingsManager.init((progress) => {
        setModelProgress(Math.round(progress * 100));
      });
      setModelProgress(null);

      const store = new VectorStore();
      await store.addDocuments(chunks);
      vectorStoreRef.current = store;

      setIsProcessing(false);
    } catch (err: any) {
      console.error(err);
      setError(err.message || 'Error processing PDF');
      setIsProcessing(false);
      setPdfName(null);
    }
  };

  const callGroq = async (prompt: string) => {
    if (!apiKey) throw new Error(t('api_key_required'));

    const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${apiKey}`,
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        model: selectedModel,
        messages: [{ role: 'user', content: prompt }],
        temperature: 0.2,
        max_tokens: 1024
      })
    });

    if (!response.ok) {
      const data = await response.json();
      throw new Error(data.error?.message || 'Failed to call Groq API');
    }

    const data = await response.json();
    return data.choices[0].message.content;
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isProcessing) return;
    if (!vectorStoreRef.current) {
      setError(t('no_pdf_yet'));
      return;
    }

    const userQuery = inputValue.trim();
    setInputValue('');
    setMessages(prev => [...prev, { role: 'user', content: userQuery }]);
    setIsProcessing(true);
    setError(null);

    try {
      // 1. Retrieve relevant context
      const searchResults = await vectorStoreRef.current.similaritySearch(userQuery, 5);
      const context = searchResults.map(r => r.pageContent).join('\n\n---\n\n');

      // 2. Build Prompt
      const prompt = `
        ${t('prompt_template')}
        
        CONTEXT:
        ${context}
        
        QUESTION:
        ${userQuery}
      `;

      // 3. Call AI
      const answer = await callGroq(prompt);

      setMessages(prev => [...prev, { 
        role: 'assistant', 
        content: answer,
        details: {
          model: selectedModel,
          strategy: searchResults.length > 0 ? t('strategy_rag') : t('strategy_full'),
          chunks: searchResults.length,
          pages: 0,
          chars: context.length
        }
      }]);
    } catch (err: any) {
      console.error(err);
      setError(err.message);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="flex h-screen bg-[#0f111a] text-slate-200 font-sans overflow-hidden">
      {/* Sidebar Overlay */}
      <AnimatePresence>
        {isSidebarOpen && (
          <motion.div 
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setIsSidebarOpen(false)}
            className="fixed inset-0 bg-black/50 z-40 backdrop-blur-sm lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside 
        initial={false}
        animate={{ 
          width: isSidebarOpen ? 320 : 0,
          opacity: isSidebarOpen ? 1 : 0
        }}
        className="fixed lg:relative z-50 h-full bg-[#161925] border-r border-slate-800/50 flex flex-col overflow-hidden"
      >
        <div className="p-6 w-[320px] flex-1 flex flex-col gap-6">
          <div className="flex items-center justify-between">
            <h2 className="text-xl font-bold flex items-center gap-2">
              <Settings className="w-5 h-5 text-indigo-400" />
              {t('config')}
            </h2>
            <button onClick={() => setIsSidebarOpen(false)} className="lg:hidden text-slate-400 hover:text-white">
              <ChevronRight className="w-6 h-6 rotate-180" />
            </button>
          </div>

          <div className="space-y-4">
            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                Groq API Key
              </label>
              <input 
                type="password"
                value={apiKey}
                onChange={(e) => setApiKey(e.target.value)}
                placeholder="gsk_..."
                className="w-full bg-[#0f111a] border border-slate-700 rounded-xl px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none transition-all"
              />
            </div>

            <div className="space-y-2">
              <label className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
                {t('model')}
              </label>
              <select 
                value={selectedModel}
                onChange={(e) => setSelectedModel(e.target.value)}
                className="w-full bg-[#0f111a] border border-slate-700 rounded-xl px-4 py-2 focus:ring-2 focus:ring-indigo-500 outline-none appearance-none"
              >
                <option value="llama-3.3-70b-versatile">Llama 3.3 70B</option>
                <option value="llama3-8b-8192">Llama 3 8B</option>
                <option value="mixtral-8x7b-32768">Mixtral 8x7B</option>
              </select>
            </div>
          </div>

          <div className="mt-auto pt-6 border-t border-slate-800">
            <button 
              onClick={() => {
                setMessages([]);
                setPdfName(null);
                vectorStoreRef.current = null;
                setError(null);
              }}
              className="w-full flex items-center justify-center gap-2 py-3 px-4 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 transition-colors"
            >
              <Trash2 className="w-4 h-4" />
              {t('clear_cache')}
            </button>
          </div>
        </div>
      </motion.aside>

      {/* Main Content */}
      <main className="flex-1 flex flex-col min-w-0 bg-[#0f111a] relative">
        {/* Top Header */}
        <header className="h-16 border-b border-slate-800/50 flex items-center justify-between px-6 bg-[#0f111a]/80 backdrop-blur-md sticky top-0 z-30">
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 -ml-2 text-slate-400 hover:text-white lg:hidden"
            >
              <Settings className="w-6 h-6" />
            </button>
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 bg-indigo-600 rounded-lg flex items-center justify-center">
                <FileText className="w-5 h-5 text-white" />
              </div>
              <h1 className="font-bold text-lg hidden sm:block tracking-tight">Agente PDF</h1>
            </div>
          </div>

          <div className="flex items-center gap-2 sm:gap-4">
            <div className="hidden sm:flex bg-slate-900/50 p-1 rounded-xl border border-slate-800">
              {(['es', 'en', 'ru'] as Language[]).map((l) => (
                <button
                  key={l}
                  onClick={() => setLang(l)}
                  className={cn(
                    "px-3 py-1 rounded-lg text-xs font-bold transition-all uppercase",
                    lang === l ? "bg-indigo-600 text-white shadow-lg" : "text-slate-500 hover:text-slate-300"
                  )}
                >
                  {l}
                </button>
              ))}
            </div>
            
            <button 
              onClick={() => setIsSidebarOpen(true)}
              className="hidden lg:flex p-2 bg-slate-900 border border-slate-800 rounded-xl text-slate-400 hover:text-white transition-all shadow-sm"
            >
              <Settings className="w-5 h-5" />
            </button>
          </div>
        </header>

        {/* Chat Area */}
        <div className="flex-1 overflow-y-auto px-4 py-8 space-y-6 scrollbar-thin scrollbar-thumb-slate-800 scrollbar-track-transparent">
          {messages.length === 0 ? (
            <div className="max-w-2xl mx-auto flex flex-col items-center justify-center mt-20 text-center animate-in fade-in slide-in-from-bottom-4 duration-700">
              <div className="w-20 h-20 bg-indigo-600/10 rounded-full flex items-center justify-center mb-6">
                <FileText className="w-10 h-10 text-indigo-500" />
              </div>
              <h2 className="text-3xl font-bold text-white mb-2">{t('greeting')}</h2>
              <p className="text-slate-400 text-lg mb-8 max-w-md">{t('greeting_sub')}</p>
              
              {!pdfName ? (
                <label className="group relative flex flex-col items-center justify-center w-full max-w-sm h-40 border-2 border-dashed border-slate-700 rounded-3xl hover:border-indigo-500/50 hover:bg-indigo-500/5 transition-all cursor-pointer overflow-hidden">
                  <div className="flex flex-col items-center justify-center pt-5 pb-6">
                    <Database className="w-10 h-10 text-slate-500 mb-4 group-hover:scale-110 transition-transform duration-300" />
                    <p className="text-sm text-slate-400 group-hover:text-slate-300">{t('attach_pdf')}</p>
                  </div>
                  <input type="file" className="hidden" accept=".pdf" onChange={handleFileUpload} />
                  
                  {isProcessing && (
                    <div className="absolute inset-0 bg-slate-900/90 flex flex-col items-center justify-center p-6 backdrop-blur-sm">
                      <Loader2 className="w-10 h-10 text-indigo-500 animate-spin mb-4" />
                      <p className="text-white font-medium mb-1">{t('processing')}</p>
                      {modelProgress !== null && (
                        <div className="w-full bg-slate-800 rounded-full h-1.5 mt-2 overflow-hidden max-w-[200px]">
                          <motion.div 
                            className="bg-indigo-500 h-full" 
                            initial={{ width: 0 }}
                            animate={{ width: `${modelProgress}%` }}
                          />
                        </div>
                      )}
                    </div>
                  )}
                </label>
              ) : (
                <div className="flex items-center gap-3 px-6 py-4 bg-indigo-600/10 border border-indigo-500/30 rounded-2xl">
                  <CheckCircle2 className="w-5 h-5 text-indigo-400" />
                  <span className="font-medium text-indigo-100">{t('pdf_loaded')}: {pdfName}</span>
                </div>
              )}
            </div>
          ) : (
            <div className="max-w-3xl mx-auto space-y-6">
              {messages.map((msg, i) => (
                <motion.div 
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  key={i} 
                  className={cn(
                    "flex flex-col gap-2",
                    msg.role === 'user' ? "items-end" : "items-start"
                  )}
                >
                  <div className={cn(
                    "max-w-[85%] px-5 py-3.5 rounded-2xl text-[15px] leading-relaxed shadow-sm",
                    msg.role === 'user' 
                      ? "bg-indigo-600 text-white rounded-tr-none" 
                      : "bg-[#1e2330] text-slate-200 border border-slate-800/50 rounded-tl-none"
                  )}>
                    {msg.content}
                  </div>
                  
                  {msg.details && (
                    <div className="flex items-center gap-4 mt-1 px-1">
                      <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-slate-500">
                        <Cpu className="w-3 h-3" />
                        {msg.details.model}
                      </div>
                      <div className="flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-widest text-indigo-400/80">
                        <Search className="w-3 h-3" />
                        {msg.details.strategy}
                      </div>
                    </div>
                  )}
                </motion.div>
              ))}
              
              {isProcessing && (
                <div className="flex items-start gap-3">
                  <div className="bg-[#1e2330] px-5 py-4 rounded-2xl rounded-tl-none border border-slate-800/50">
                    <div className="flex gap-1">
                      <span className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                      <span className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                      <span className="w-2 h-2 bg-indigo-500 rounded-full animate-bounce"></span>
                    </div>
                  </div>
                </div>
              )}
              
              {error && (
                <div className="flex items-center gap-3 bg-red-500/10 border border-red-500/20 p-4 rounded-xl text-red-400 text-sm">
                  <AlertCircle className="w-5 h-5 flex-shrink-0" />
                  {error}
                </div>
              )}
              <div ref={chatEndRef} />
            </div>
          )}
        </div>

        {/* Input Area */}
        <div className="p-4 sm:p-6 bg-gradient-to-t from-[#0f111a] via-[#0f111a] to-transparent">
          <div className="max-w-3xl mx-auto">
            {pdfName && (
              <div className="mb-4 flex items-center justify-between px-2">
                <div className="flex items-center gap-2 text-xs font-medium text-slate-500">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  {pdfName}
                </div>
                <button 
                  onClick={() => {
                    setPdfName(null);
                    vectorStoreRef.current = null;
                  }}
                  className="text-xs text-slate-600 hover:text-red-400 transition-colors"
                >
                  Cambiar PDF
                </button>
              </div>
            )}
            
            <div className="relative group flex items-center gap-3 bg-[#1e2330] border border-slate-700/50 p-2 rounded-2xl focus-within:border-indigo-500/50 focus-within:ring-4 focus-within:ring-indigo-500/5 transition-all shadow-xl">
              <button 
                onClick={toggleRecording}
                className={cn(
                  "p-3 rounded-xl transition-all",
                  isRecording 
                    ? "bg-red-500 text-white animate-pulse" 
                    : "text-slate-500 hover:text-indigo-400 hover:bg-indigo-500/10"
                )}
              >
                {isRecording ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
              </button>
              
              <input 
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => e.key === 'Enter' && handleSendMessage()}
                placeholder={t('chat_placeholder')}
                className="flex-1 bg-transparent py-3 text-slate-200 outline-none placeholder:text-slate-600"
              />
              
              <button 
                onClick={handleSendMessage}
                disabled={!inputValue.trim() || isProcessing || !pdfName}
                className={cn(
                  "p-3 rounded-xl transition-all shadow-lg",
                  inputValue.trim() && !isProcessing && pdfName
                    ? "bg-indigo-600 text-white hover:bg-indigo-500 scale-100" 
                    : "bg-slate-800 text-slate-600 scale-95 opacity-50 cursor-not-allowed"
                )}
              >
                {isProcessing ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </div>
            
            <div className="mt-3 text-center">
              <p className="text-[10px] text-slate-600 font-medium uppercase tracking-widest flex items-center justify-center gap-2">
                <Info className="w-3 h-3" />
                Powered by Groq & Transformers.js · Local Embeddings
              </p>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
