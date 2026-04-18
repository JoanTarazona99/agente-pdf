export const MODELS = {
  'llama-3.3-70b-versatile': 'Llama 3.3 70B Versatile',
  'llama-3.1-8b-instant': 'Llama 3.1 8B Instant',
  'llama3-70b-8192': 'Llama 3 70B',
  'llama3-8b-8192': 'Llama 3 8B',
  'mixtral-8x7b-32768': 'Mixtral 8x7B',
  'gemma2-9b-it': 'Gemma 2 9B',
} as const;

export type ModelId = keyof typeof MODELS;

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export async function chatWithGroq(
  model: ModelId,
  userMessage: string,
  systemPrompt: string,
  apiKey: string,
  history: ChatMessage[] = []
): Promise<{ content: string; totalTokens?: number }> {
  if (!apiKey) {
    throw new Error('No API key provided. Please add your Groq API Key in Settings.');
  }

  const messages: ChatMessage[] = [
    { role: 'system', content: systemPrompt },
    ...history,
    { role: 'user', content: userMessage }
  ];

  const response = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.3,
      max_tokens: 4096,
    }),
  });

  if (!response.ok) {
    const err = await response.json().catch(() => ({}));
    const msg = (err as any)?.error?.message || `HTTP ${response.status}`;
    throw new Error(`Groq API error: ${msg}`);
  }

  const data = await response.json();
  const content = data.choices?.[0]?.message?.content || '';
  const totalTokens = data.usage?.total_tokens;
  return { content, totalTokens };
}
