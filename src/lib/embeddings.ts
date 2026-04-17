import { pipeline, env } from '@xenova/transformers';

// Configuration for web environment
env.allowLocalModels = false;
env.useBrowserCache = true;

export class EmbeddingsManager {
  private extractor: any = null;
  private onProgress?: (progress: { status: string; progress: number }) => void;

  constructor(onProgress?: (progress: { status: string; progress: number }) => void) {
    this.onProgress = onProgress;
  }

  async init() {
    if (this.extractor) return;

    this.extractor = await pipeline('feature-extraction', 'Xenova/all-MiniLM-L6-v2', {
      progress_callback: (data: any) => {
        if (data.status === 'progress' && this.onProgress) {
          this.onProgress({
            status: data.file,
            progress: data.progress || 0
          });
        }
      }
    });
  }

  /**
   * Embeds a single string into a vector.
   */
  async embed(text: string): Promise<number[]> {
    if (!this.extractor) await this.init();
    
    const output = await this.extractor(text, { pooling: 'mean', normalize: true });
    return Array.from(output.data) as number[];
  }

  /**
   * Embeds multiple strings.
   */
  async embedMany(texts: string[]): Promise<number[][]> {
    const results: number[][] = [];
    for (const text of texts) {
      results.push(await this.embed(text));
    }
    return results;
  }
}
