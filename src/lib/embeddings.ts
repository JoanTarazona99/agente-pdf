export type ProgressCallback = (progress: { status: string; progress: number }) => void;

export class EmbeddingsManager {
  private vocabulary: Map<string, number> = new Map();
  private idf: Map<string, number> = new Map();
  private isReady = false;
  private onProgress: ProgressCallback;

  constructor(onProgress: ProgressCallback) {
    this.onProgress = onProgress;
  }

  async init(): Promise<void> {
    this.onProgress({ status: 'Initializing...', progress: 50 });
    await new Promise(r => setTimeout(r, 100));
    this.isReady = true;
    this.onProgress({ status: 'Ready', progress: 100 });
  }

  private tokenize(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^a-záéíóúàèìòùñüçа-яё\d\s]/gi, ' ')
      .split(/\s+/)
      .filter(t => t.length > 2);
  }

  private termFrequency(tokens: string[]): Map<string, number> {
    const tf = new Map<string, number>();
    for (const token of tokens) {
      tf.set(token, (tf.get(token) || 0) + 1);
    }
    for (const [key, val] of tf.entries()) {
      tf.set(key, val / tokens.length);
    }
    return tf;
  }

  buildCorpus(chunks: string[]) {
    const docFreq = new Map<string, number>();

    for (const chunk of chunks) {
      const tokens = new Set(this.tokenize(chunk));
      for (const token of tokens) {
        docFreq.set(token, (docFreq.get(token) || 0) + 1);
      }
    }

    const N = chunks.length;
    for (const [term, df] of docFreq.entries()) {
      this.idf.set(term, Math.log((N + 1) / (df + 1)) + 1);
    }

    let idx = 0;
    for (const term of docFreq.keys()) {
      this.vocabulary.set(term, idx++);
    }
  }

  embed(text: string): number[] {
    const tokens = this.tokenize(text);
    const tf = this.termFrequency(tokens);
    const size = Math.max(this.vocabulary.size, 1);
    const vector = new Array(size).fill(0);

    for (const [term, tfVal] of tf.entries()) {
      const idx = this.vocabulary.get(term);
      if (idx !== undefined) {
        const idfVal = this.idf.get(term) || 1;
        vector[idx] = tfVal * idfVal;
      }
    }

    const norm = Math.sqrt(vector.reduce((s: number, v: number) => s + v * v, 0)) + 1e-10;
    return vector.map((v: number) => v / norm);
  }

  get ready() {
    return this.isReady;
  }
}
