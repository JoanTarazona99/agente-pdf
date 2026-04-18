import type { VercelRequest, VercelResponse } from '@vercel/node';

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method not allowed' });
  }

  try {
    const { model, messages, apiKey } = req.body ?? {};

    const effectiveApiKey = apiKey?.trim() || process.env.GROQ_API_KEY;

    if (!effectiveApiKey) {
      return res.status(500).json({
        error: 'No hay GROQ_API_KEY en Vercel ni API key enviada por el usuario.',
      });
    }

    const groqResponse = await fetch('https://api.groq.com/openai/v1/chat/completions', {
      method: 'POST',
      headers: {
        Authorization: `Bearer ${effectiveApiKey}`,
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        model,
        messages,
        temperature: 0.3,
        max_tokens: 4096,
      }),
    });

    const data = await groqResponse.json();

    if (!groqResponse.ok) {
      const msg = data?.error?.message || `HTTP ${groqResponse.status}`;
      return res.status(groqResponse.status).json({ error: msg });
    }

    return res.status(200).json({
      content: data.choices?.[0]?.message?.content || '',
      totalTokens: data.usage?.total_tokens,
    });
  } catch (error: any) {
    return res.status(500).json({
      error: error?.message || 'Unexpected server error',
    });
  }
}