export interface DocumentChunk {
  content: string;
  embedding: number[];
  metadata: {
    page?: number;
  };
}

export class VectorStore {
  private chunks: DocumentChunk[] = [];

  addChunks(chunks: DocumentChunk[]) {
    this.chunks = [...this.chunks, ...chunks];
  }

  clear() {
    this.chunks = [];
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
    const magA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
    const magB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
    return dotProduct / (magA * magB);
  }

  search(queryEmbedding: number[], k: number = 5): DocumentChunk[] {
    return [...this.chunks]
      .sort((a, b) => {
        const simA = this.cosineSimilarity(queryEmbedding, a.embedding);
        const simB = this.cosineSimilarity(queryEmbedding, b.embedding);
        return simB - simA;
      })
      .slice(0, k);
  }
}
