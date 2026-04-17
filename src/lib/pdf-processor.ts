import * as pdfjs from 'pdfjs-dist';

// Standard worker setup for pdfjs
pdfjs.GlobalWorkerOptions.workerSrc = `https://cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.min.js`;

export interface DocumentChunk {
  pageContent: string;
  metadata: {
    pageNumber: number;
    chunkIndex: number;
  };
}

export class PDFProcessor {
  /**
   * Extracts text from a PDF file
   */
  static async extractText(file: File): Promise<{ text: string, pages: string[] }> {
    const arrayBuffer = await file.arrayBuffer();
    const loadingTask = pdfjs.getDocument(new Uint8Array(arrayBuffer));
    const pdf = await loadingTask.promise;
    
    let fullText = '';
    const pages: string[] = [];
    
    for (let i = 1; i <= pdf.numPages; i++) {
      const page = await pdf.getPage(i);
      const textContent = await page.getTextContent();
      const pageText = textContent.items
        .map((item: any) => item.str)
        .join(' ');
      
      fullText += pageText + ' ';
      pages.push(pageText);
    }
    
    return { text: fullText, pages };
  }

  /**
   * Splits text into overlapping chunks for better RAG results
   */
  static chunkText(pages: string[], chunkSize = 800, chunkOverlap = 150): DocumentChunk[] {
    const chunks: DocumentChunk[] = [];
    
    pages.forEach((pageText, pageIdx) => {
      let currentIdx = 0;
      let chunkCount = 0;
      
      while (currentIdx < pageText.length) {
        const chunk = pageText.slice(currentIdx, currentIdx + chunkSize);
        chunks.push({
          pageContent: chunk,
          metadata: {
            pageNumber: pageIdx + 1,
            chunkIndex: chunkCount,
          }
        });
        
        currentIdx += chunkSize - chunkOverlap;
        chunkCount++;
        
        // Safety break
        if (chunkCount > 1000) break;
      }
    });
    
    return chunks;
  }
}
