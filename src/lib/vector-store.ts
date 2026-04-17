import { EmbeddingsManager } from './embeddings';
import { DocumentChunk } from './pdf-processor';

export class VectorStore {
  private chunks: DocumentChunk[] = [];
  private embeddings: number[][] = [];
  private embeddingsManager: EmbeddingsManager;

  constructor() {
    this.embeddingsManager = EmbeddingsManager.getInstance();
  }

  async addDocuments(chunks: DocumentChunk[]) {
    this.chunks = [...this.chunks, ...chunks];
    const texts = chunks.map(c => c.pageContent);
    const newEmbeddings = await this.embeddingsManager.embed(texts);
    this.embeddings = [...this.embeddings, ...newEmbeddings];
  }

  async similaritySearch(query: string, k = 5): Promise<DocumentChunk[]> {
    const queryEmbedding = (await this.embeddingsManager.embed(query))[0];
    
    const scores = this.embeddings.map((emb, idx) => ({
      score: EmbeddingsManager.cosineSimilarity(queryEmbedding, emb),
      chunk: this.chunks[idx]
    }));

    // Sort by score descending and take top k
    return scores
      .sort((a, b) => b.score - a.score)
      .slice(0, k)
      .map(s => s.chunk);
  }

  clear() {
    this.chunks = [];
    this.embeddings = [];
  }

  get totalChunks() {
    return this.chunks.length;
  }
}
