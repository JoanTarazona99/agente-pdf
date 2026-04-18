import * as pdfjsLib from 'pdfjs-dist';

// Default worker: CDN (fast). If the CDN is unreachable from the browser
// we fall back to a proxied URL using the local proxy at 127.0.0.1:10809.
// The fallback is only applied when the quick HEAD check fails or errors.
const CDN_WORKER_URL = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjsLib.version}/pdf.worker.min.mjs`;

// Set default first (optimistic).
pdfjsLib.GlobalWorkerOptions.workerSrc = CDN_WORKER_URL;

// Check availability of CDN worker with a short HEAD request and
// only if it fails, switch to proxied fallback.
(async () => {
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 3000);

    // Use HEAD to avoid downloading the full worker during the check.
    const res = await fetch(CDN_WORKER_URL, { method: 'HEAD', mode: 'cors', signal: controller.signal });
    clearTimeout(timeout);

    if (!res.ok) {
      // Switch to proxied fallback
      pdfjsLib.GlobalWorkerOptions.workerSrc = `http://127.0.0.1:10809/${CDN_WORKER_URL}`;
      // eslint-disable-next-line no-console
      console.warn('[PDFProcessor] CDN worker unreachable (status ' + res.status + '). Using proxied fallback.');
    }
  } catch (err) {
    // On error (network, CORS, aborted), use proxied fallback.
    pdfjsLib.GlobalWorkerOptions.workerSrc = `http://127.0.0.1:10809/${CDN_WORKER_URL}`;
    // eslint-disable-next-line no-console
    console.warn('[PDFProcessor] Error checking CDN worker; using proxied fallback.', err);
  }
})();

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
