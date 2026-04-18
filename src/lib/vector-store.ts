export interface DocumentChunk {
  content: string;
  embedding: number[];
  metadata: Record<string, any>;
}

export class VectorStore {
  private chunks: DocumentChunk[] = [];

  addChunks(chunks: DocumentChunk[]) {
    this.chunks.push(...chunks);
  }

  clear() {
    this.chunks = [];
  }

  get size() {
    return this.chunks.length;
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    let dot = 0, normA = 0, normB = 0;
    for (let i = 0; i < a.length; i++) {
      dot += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }
    return dot / (Math.sqrt(normA) * Math.sqrt(normB) + 1e-10);
  }

  search(queryEmbedding: number[], topK = 5): DocumentChunk[] {
    if (this.chunks.length === 0) return [];

    const scored = this.chunks.map(chunk => ({
      chunk,
      score: this.cosineSimilarity(queryEmbedding, chunk.embedding)
    }));

    scored.sort((a, b) => b.score - a.score);
    return scored.slice(0, topK).map(s => s.chunk);
  }
}
