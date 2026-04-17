import Groq from 'groq-sdk';

export const MODELS = {
  'llama-3.3-70b-versatile': '🧠 Llama 3.3 70B',
  'llama-3.1-8b-instant': '⚡ Llama 3.1 8B',
  'gemma2-9b-it': '💎 Gemma 2 9B',
  'mixtral-8x7b-32768': '🤝 Mixtral 8x7B'
};

export type ModelId = keyof typeof MODELS;
export async function chatWithGroq(
  model: string,
  prompt: string,
  systemPrompt: string,
  opts?: { temperature?: number; max_tokens?: number }
) {
  // In browser: call serverless proxy to avoid exposing API key client-side
  if (typeof window !== 'undefined') {
    const res = await fetch('/api/groq', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ model, prompt, systemPrompt, temperature: opts?.temperature })
    });

    if (!res.ok) {
      let errMsg = res.statusText;
      try {
        const errBody = await res.json();
        errMsg = errBody.error || errMsg;
      } catch (_) {}
      throw new Error(errMsg || 'Groq request failed');
    }

    const data = await res.json();
    return data.text || '';
  }

  // Server-side: use Groq SDK with API key from environment
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error('GROQ_API_KEY not configured in server environment');

  const groq = new Groq({ apiKey });
  const completion = await groq.chat.completions.create({
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt }
    ],
    model: model,
    temperature: opts?.temperature ?? 0.2,
    max_tokens: opts?.max_tokens ?? 1024,
  });

  return completion.choices[0]?.message?.content || '';
}
