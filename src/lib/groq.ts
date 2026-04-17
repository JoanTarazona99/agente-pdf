import Groq from 'groq-sdk';

export const MODELS = {
  'llama-3.3-70b-versatile': '🧠 Llama 3.3 70B',
  'llama-3.1-8b-instant': '⚡ Llama 3.1 8B',
  'gemma2-9b-it': '💎 Gemma 2 9B',
  'mixtral-8x7b-32768': '🤝 Mixtral 8x7B'
};

export type ModelId = keyof typeof MODELS;

export async function chatWithGroq(
  apiKey: string,
  model: string,
  prompt: string,
  systemPrompt: string
) {
  const groq = new Groq({ apiKey, dangerouslyAllowBrowser: true });
  
  const completion = await groq.chat.completions.create({
    messages: [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: prompt }
    ],
    model: model,
    temperature: 0.2,
    max_tokens: 1024,
  });

  return completion.choices[0]?.message?.content || '';
}
