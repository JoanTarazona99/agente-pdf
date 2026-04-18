import { useDropzone } from 'react-dropzone';
import { motion } from 'framer-motion';
import { FileText, Upload, X, FileCheck } from 'lucide-react';
import { clsx } from 'clsx';

interface Props {
  onDrop: (files: File[]) => void;
  pdfName: string | null;
  pageCount: number;
  charCount: number;
  isProcessing: boolean;
  onClear: () => void;
  noFileLabel: string;
  changeLabel: string;
  pagesLabel: string;
  charsLabel: string;
  dragActiveLabel: string;
}

export function PDFDropzone({
  onDrop, pdfName, pageCount, charCount, isProcessing,
  onClear, noFileLabel, changeLabel, pagesLabel, charsLabel, dragActiveLabel
}: Props) {
  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'application/pdf': ['.pdf'] },
    multiple: false,
    disabled: isProcessing,
  });

  if (pdfName) {
    return (
      <motion.div
        initial={{ opacity: 0, y: -8 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center gap-3 px-4 py-2.5 bg-emerald-950/40 border border-emerald-500/30 rounded-xl"
      >
        <FileCheck size={18} className="text-emerald-400 shrink-0" />
        <div className="flex-1 min-w-0">
          <p className="text-sm font-medium text-emerald-300 truncate">{pdfName}</p>
          <p className="text-[11px] text-emerald-600">
            {pageCount} {pagesLabel} · {charCount.toLocaleString()} {charsLabel}
          </p>
        </div>
        <div className="flex items-center gap-1">
          <div {...getRootProps()}>
            <input {...getInputProps()} />
            <button
              className="p-1.5 rounded-lg text-slate-500 hover:text-slate-300 hover:bg-slate-800/50 transition-colors text-xs"
              title={changeLabel}
            >
              <Upload size={14} />
            </button>
          </div>
          <button
            onClick={(e) => { e.stopPropagation(); onClear(); }}
            className="p-1.5 rounded-lg text-slate-500 hover:text-red-400 hover:bg-red-900/20 transition-colors"
          >
            <X size={14} />
          </button>
        </div>
      </motion.div>
    );
  }

  return (
    <div
      {...getRootProps()}
      className={clsx(
        'relative flex flex-col items-center justify-center gap-3 p-8 border-2 border-dashed rounded-2xl cursor-pointer transition-all duration-300',
        isDragActive
          ? 'border-indigo-400 bg-indigo-500/10 scale-[1.01]'
          : 'border-slate-700/60 hover:border-indigo-500/50 hover:bg-indigo-500/5 bg-slate-900/30'
      )}
    >
      <input {...getInputProps()} />
      <motion.div
        animate={isDragActive ? { scale: 1.1 } : { scale: 1 }}
        className={clsx(
          'w-14 h-14 rounded-2xl flex items-center justify-center transition-colors',
          isDragActive ? 'bg-indigo-500/20' : 'bg-slate-800/60'
        )}
      >
        {isDragActive
          ? <Upload size={24} className="text-indigo-400" />
          : <FileText size={24} className="text-slate-500" />}
      </motion.div>
      <p className="text-sm text-slate-400 text-center font-medium">
        {isDragActive ? dragActiveLabel : noFileLabel}
      </p>
      {!isDragActive && (
        <p className="text-[11px] text-slate-600">PDF · Max 50MB</p>
      )}
    </div>
  );
}
