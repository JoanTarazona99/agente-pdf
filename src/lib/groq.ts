export async function callGroq(
  messages: { role: string; content: string }[],
  apiKey: string,
  model = "llama-3.3-70b-versatile",
  temperature = 0.2
) {
  const response = await fetch("https://api.groq.com/openai/v1/chat/completions", {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${apiKey}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      model,
      messages,
      temperature,
      max_tokens: 1024,
      stream: false,
    }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.error?.message || "Groq API error");
  }

  const data = await response.json();
  return data.choices[0].message.content;
}
