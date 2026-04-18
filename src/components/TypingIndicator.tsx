export function TypingIndicator() {
  return (
    <div className="flex items-end gap-3 max-w-[85%]">
      <div className="w-8 h-8 rounded-full bg-indigo-600/30 border border-indigo-500/40 flex items-center justify-center shrink-0 mb-1">
        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" className="text-indigo-400">
          <path d="M12 2L2 7l10 5 10-5-10-5zM2 17l10 5 10-5M2 12l10 5 10-5" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
      </div>
      <div className="bg-[#1e1e35] border border-indigo-500/20 rounded-2xl rounded-tl-none px-4 py-3">
        <div className="flex items-center gap-1.5">
          <span className="typing-dot w-2 h-2 rounded-full bg-indigo-400 block" />
          <span className="typing-dot w-2 h-2 rounded-full bg-indigo-400 block" />
          <span className="typing-dot w-2 h-2 rounded-full bg-indigo-400 block" />
        </div>
      </div>
    </div>
  );
}
