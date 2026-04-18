import { motion } from 'framer-motion';
import { X, Settings, Key, Cpu, Languages, Trash2, ExternalLink, CheckCircle } from 'lucide-react';
import { MODELS, ModelId } from '../lib/groq';
import { Language } from '../lib/translations';

interface Props {
  isOpen: boolean;
  onClose: () => void;
  apiKey: string;
  onApiKeyChange: (key: string) => void;
  model: ModelId;
  onModelChange: (model: ModelId) => void;
  lang: Language;
  onLangChange: (lang: Language) => void;
  onClearChat: () => void;
  t: (key: string) => string;
  apiKeySaved: boolean;
}

export function SettingsDrawer({
  isOpen, onClose, apiKey, onApiKeyChange, model, onModelChange,
  lang, onLangChange, onClearChat, t, apiKeySaved
}: Props) {
  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        onClick={onClose}
        className="fixed inset-0 bg-black/60 z-40 backdrop-blur-sm"
      />

      {/* Drawer */}
      <motion.div
        initial={{ x: '100%' }}
        animate={{ x: 0 }}
        exit={{ x: '100%' }}
        transition={{ type: 'spring', damping: 25, stiffness: 300 }}
        className="fixed right-0 top-0 h-full w-80 bg-[#13132a] z-50 shadow-2xl border-l border-indigo-900/40 flex flex-col"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-5 border-b border-indigo-900/30">
          <div className="flex items-center gap-2.5">
            <div className="p-1.5 bg-indigo-900/50 rounded-lg">
              <Settings size={16} className="text-indigo-400" />
            </div>
            <h2 className="font-semibold text-slate-200">{t('settings')}</h2>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 hover:bg-slate-800 rounded-lg transition-colors text-slate-400 hover:text-white"
          >
            <X size={18} />
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto p-5 space-y-6">
          {/* API Key */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <Key size={12} />
              Groq API Key
            </label>
            <div className="relative">
              <input
                type="password"
                value={apiKey}
                onChange={(e) => onApiKeyChange(e.target.value)}
                placeholder="gsk_..."
                className="w-full bg-[#1e1e35] border border-slate-700/60 rounded-xl px-4 py-2.5 text-sm focus:ring-2 focus:ring-indigo-500/50 outline-none transition-all pr-10 placeholder-slate-600"
              />
              {apiKeySaved && (
                <CheckCircle size={16} className="absolute right-3 top-1/2 -translate-y-1/2 text-emerald-400" />
              )}
            </div>
            <a
              href="https://console.groq.com/keys"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1.5 text-[11px] text-indigo-400 hover:text-indigo-300 transition-colors mt-1"
            >
              <ExternalLink size={11} />
              Get your free API key at console.groq.com
            </a>
          </div>

          {/* Model */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <Cpu size={12} />
              {t('model')}
            </label>
            <div className="space-y-1.5">
              {(Object.entries(MODELS) as [ModelId, string][]).map(([id, label]) => (
                <button
                  key={id}
                  onClick={() => onModelChange(id)}
                  className={`w-full text-left px-3 py-2.5 rounded-xl text-sm transition-all border ${
                    model === id
                      ? 'bg-indigo-900/50 border-indigo-500/50 text-indigo-200'
                      : 'bg-[#1e1e35] border-slate-700/40 text-slate-400 hover:border-slate-600 hover:text-slate-300'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{label}</span>
                    {model === id && (
                      <span className="text-[10px] text-indigo-400 font-bold">ACTIVE</span>
                    )}
                  </div>
                </button>
              ))}
            </div>
          </div>

          {/* Language */}
          <div className="space-y-2">
            <label className="flex items-center gap-2 text-xs font-semibold text-slate-400 uppercase tracking-wider">
              <Languages size={12} />
              {t('language')}
            </label>
            <div className="grid grid-cols-3 gap-2">
              {(['es', 'en', 'ru'] as Language[]).map((l) => (
                <button
                  key={l}
                  onClick={() => onLangChange(l)}
                  className={`py-2 rounded-xl text-sm font-bold transition-all border ${
                    lang === l
                      ? 'bg-indigo-600 border-indigo-500 text-white shadow-lg shadow-indigo-500/20'
                      : 'bg-[#1e1e35] border-slate-700/40 text-slate-400 hover:border-slate-500 hover:text-slate-200'
                  }`}
                >
                  {l === 'es' ? '🇪🇸 ES' : l === 'en' ? '🇺🇸 EN' : '🇷🇺 RU'}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className="p-5 border-t border-indigo-900/30">
          <button
            onClick={() => { onClearChat(); onClose(); }}
            className="w-full flex items-center justify-center gap-2 py-2.5 rounded-xl border border-red-500/30 bg-red-500/10 text-red-400 hover:bg-red-500/20 transition-colors text-sm font-medium"
          >
            <Trash2 size={16} />
            {t('new_chat')}
          </button>
          <p className="text-center text-[10px] text-slate-700 mt-3">DocuMind AI v2.0 · Powered by Groq</p>
        </div>
      </motion.div>
    </>
  );
}
