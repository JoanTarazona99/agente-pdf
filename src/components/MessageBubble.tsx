import { useState } from 'react';
import { Copy, Check, Bot, User, ChevronDown, ChevronUp } from 'lucide-react';
import { clsx } from 'clsx';

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  details?: string;
  tokens?: number;
  timestamp?: number;
}

interface Props {
  message: Message;
  copyLabel: string;
}

function formatMarkdown(text: string): string {
  return text
    .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
    .replace(/\*(.*?)\*/g, '<em>$1</em>')
    .replace(/`([^`]+)`/g, '<code>$1</code>')
    .replace(/^### (.+)$/gm, '<h3>$1</h3>')
    .replace(/^## (.+)$/gm, '<h2>$1</h2>')
    .replace(/^# (.+)$/gm, '<h1>$1</h1>')
    .replace(/^[-•] (.+)$/gm, '<li>$1</li>')
    .replace(/(<li>.*<\/li>)/gs, (match) => `<ul>${match}</ul>`)
    .replace(/\n\n/g, '</p><p>')
    .replace(/^(?!<[h|u|l])/gm, '')
    .trim();
}

export function MessageBubble({ message, copyLabel }: Props) {
  const [copied, setCopied] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const isUser = message.role === 'user';

  const handleCopy = async () => {
    await navigator.clipboard.writeText(message.content);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const time = message.timestamp
    ? new Date(message.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
    : '';

  return (
    <div className={clsx('flex gap-3 fade-in-up', isUser ? 'flex-row-reverse' : 'flex-row')}>
      {/* Avatar */}
      <div className={clsx(
        'w-8 h-8 rounded-full flex items-center justify-center shrink-0 mt-1',
        isUser
          ? 'bg-gradient-to-br from-blue-500 to-indigo-600'
          : 'bg-indigo-900/60 border border-indigo-500/40'
      )}>
        {isUser
          ? <User size={14} className="text-white" />
          : <Bot size={14} className="text-indigo-400" />}
      </div>

      {/* Bubble */}
      <div className={clsx('flex flex-col max-w-[82%]', isUser ? 'items-end' : 'items-start')}>
        <div className={clsx(
          'relative group rounded-2xl px-4 py-3 shadow-lg',
          isUser
            ? 'bg-gradient-to-br from-indigo-600 to-blue-600 text-white rounded-tr-none'
            : 'bg-[#1e1e35] border border-indigo-500/20 text-slate-200 rounded-tl-none'
        )}>
          {/* Content */}
          {isUser ? (
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{message.content}</p>
          ) : (
            <div
              className="message-content text-[15px] leading-relaxed"
              dangerouslySetInnerHTML={{
                __html: `<p>${formatMarkdown(message.content)}</p>`
              }}
            />
          )}

          {/* Copy button (assistant only) */}
          {!isUser && (
            <button
              onClick={handleCopy}
              className="absolute top-2 right-2 opacity-0 group-hover:opacity-100 transition-opacity p-1.5 rounded-lg bg-indigo-950/60 hover:bg-indigo-900/80 text-slate-400 hover:text-white"
              title={copyLabel}
            >
              {copied ? <Check size={13} className="text-green-400" /> : <Copy size={13} />}
            </button>
          )}
        </div>

        {/* Footer */}
        <div className={clsx(
          'flex items-center gap-2 mt-1 px-1',
          isUser ? 'flex-row-reverse' : 'flex-row'
        )}>
          {time && <span className="text-[10px] text-slate-600">{time}</span>}
          {message.details && (
            <button
              onClick={() => setShowDetails(!showDetails)}
              className="flex items-center gap-1 text-[10px] text-slate-600 hover:text-slate-400 transition-colors"
            >
              {showDetails ? <ChevronUp size={10} /> : <ChevronDown size={10} />}
              <span>details</span>
            </button>
          )}
          {message.tokens && (
            <span className="text-[10px] text-slate-700">{message.tokens} tokens</span>
          )}
        </div>

        {/* Details panel */}
        {showDetails && message.details && (
          <div className="mt-1 px-3 py-2 bg-[#161628] border border-indigo-900/30 rounded-lg text-[11px] text-slate-500 max-w-sm">
            {message.details}
          </div>
        )}
      </div>
    </div>
  );
}
