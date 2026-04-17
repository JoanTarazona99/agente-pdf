import { pipeline } from '@xenova/transformers';

export class EmbeddingsManager {
  private static instance: EmbeddingsManager;
  private extractor: any = null;
  private modelName = 'Xenova/all-MiniLM-L6-v2';

  private constructor() {}

  static getInstance(): EmbeddingsManager {
    if (!EmbeddingsManager.instance) {
      EmbeddingsManager.instance = new EmbeddingsManager();
    }
    return EmbeddingsManager.instance;
  }

  async init(onProgress?: (progress: number) => void) {
    if (this.extractor) return;

    this.extractor = await pipeline('feature-extraction', this.modelName, {
      progress_callback: (data: any) => {
        if (data.status === 'progress' && onProgress) {
          onProgress(data.progress);
        }
      }
    });
  }

  /**
   * Generates embeddings for a single string or an array of strings
   */
  async embed(text: string | string[]): Promise<number[][]> {
    if (!this.extractor) {
      await this.init();
    }

    const texts = Array.isArray(text) ? text : [text];
    const results: number[][] = [];

    for (const t of texts) {
      const output = await this.extractor(t, { pooling: 'mean', normalize: true });
      results.push(Array.from(output.data));
    }

    return results;
  }

  /**
   * Calculates cosine similarity between two vectors
   */
  static cosineSimilarity(vecA: number[], vecB: number[]): number {
    let dotProduct = 0;
    let normA = 0;
    let normB = 0;
    for (let i = 0; i < vecA.length; i++) {
      dotProduct += vecA[i] * vecB[i];
      normA += vecA[i] * vecA[i];
      normB += vecB[i] * vecB[i];
    }
    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }
}
