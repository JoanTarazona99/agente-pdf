import Groq from 'groq-sdk';

export default async function handler(req: any, res: any) {
  if (req.method !== 'POST') {
    res.setHeader('Allow', 'POST');
    return res.status(405).json({ error: 'Method not allowed' });
  }

  const { model, prompt, systemPrompt, temperature } = req.body || {};

  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: 'GROQ_API_KEY is not configured on the server' });
  }

  try {
    const groq = new Groq({ apiKey });

    const completion = await groq.chat.completions.create({
      messages: [
        { role: 'system', content: systemPrompt || '' },
        { role: 'user', content: prompt || '' }
      ],
      model: model,
      temperature: typeof temperature === 'number' ? temperature : 0.2,
      max_tokens: 1024,
    });

    const text = completion.choices?.[0]?.message?.content || '';
    return res.status(200).json({ text });
  } catch (err: any) {
    const message = err?.message || String(err) || 'Unknown error';
    return res.status(500).json({ error: message });
  }
}
