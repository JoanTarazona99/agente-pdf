import * as pdfjsLib from 'pdfjs-dist';

// Use the worker from the installed package
pdfjsLib.GlobalWorkerOptions.workerSrc = new URL(
  'pdfjs-dist/build/pdf.worker.mjs',
  import.meta.url
).toString();

export interface PDFInfo {
  text: string;
  pageCount: number;
  title?: string;
}

export class PDFProcessor {
  static async extractText(file: File): Promise<PDFInfo> {
    const arrayBuffer = await file.arrayBuffer();
    const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
    const pageCount = pdf.numPages;

    let fullText = '';
    for (let i = 1; i <= pageCount; i++) {
      const page = await pdf.getPage(i);
      const content = await page.getTextContent();
      const pageText = content.items
        .map((item: any) => item.str)
        .join(' ');
      fullText += `\n--- Página ${i} ---\n${pageText}`;
    }

    // Get metadata
    let title: string | undefined;
    try {
      const meta = await pdf.getMetadata();
      title = (meta.info as any)?.Title || undefined;
    } catch (_) {}

    return { text: fullText.trim(), pageCount, title };
  }

  static chunkText(text: string, chunkSize = 1000, overlap = 200): string[] {
    const chunks: string[] = [];
    let start = 0;
    while (start < text.length) {
      const end = Math.min(start + chunkSize, text.length);
      chunks.push(text.slice(start, end));
      start += chunkSize - overlap;
      if (start >= text.length) break;
    }
    return chunks;
  }
}
