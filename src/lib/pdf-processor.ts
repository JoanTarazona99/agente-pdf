import * as pdfjsLib from 'pdfjs-dist';

// Use CDN for worker to avoid bundling issues
pdfjsLib.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

export interface PDFDocumentInfo {
  text: string;
  pageCount: number;
}

export class PDFProcessor {
  static async extractText(file: File): Promise<PDFDocumentInfo> {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjsLib.getDocument({ data: arrayBuffer });
    const pdf = await loadingTask.promise;
    
    let fullText = '';
    const pageCount = pdf.numPages;

    for (let i = 1; i <= pageCount; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map((item: any) => item.str)
        .join(' ');
      fullText += pageText + '\n\n';
    }

    return {
      text: fullText,
      pageCount
    };
  }

  static chunkText(text: string, chunkSize: number = 1000, chunkOverlap: number = 200): string[] {
    const chunks: string[] = [];
    let start = 0;

    while (start < text.length) {
      let end = start + chunkSize;
      
      // If we're not at the end, try to find a better break point (newline or space)
      if (end < text.length) {
        const lastNewline = text.lastIndexOf('\n', end);
        if (lastNewline > start + chunkSize * 0.5) {
          end = lastNewline + 1;
        } else {
          const lastSpace = text.lastIndexOf(' ', end);
          if (lastSpace > start + chunkSize * 0.5) {
            end = lastSpace + 1;
          }
        }
      }

      chunks.push(text.slice(start, end).trim());
      start = end - chunkOverlap;
      
      // Safety check to prevent infinite loop
      if (start >= text.length || end >= text.length) break;
    }

    return chunks.filter(c => c.length > 5);
  }
}
