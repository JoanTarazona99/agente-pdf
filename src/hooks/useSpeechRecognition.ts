import { useState, useEffect, useCallback } from 'react';

export const useSpeechRecognition = (lang: string = 'es-ES') => {
  const [isListening, setIsListening] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [error, setError] = useState<string | null>(null);

  const SpeechRecognition = (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
  
  const recognition = SpeechRecognition ? new SpeechRecognition() : null;

  useEffect(() => {
    if (!recognition) {
      setError('Speech Recognition not supported in this browser.');
      return;
    }

    recognition.continuous = false;
    recognition.interimResults = false;
    recognition.lang = lang;

    recognition.onstart = () => setIsListening(true);
    recognition.onend = () => setIsListening(false);
    recognition.onerror = (event: any) => {
      setError(event.error);
      setIsListening(false);
    };
    recognition.onresult = (event: any) => {
      const currentTranscript = event.results[0][0].transcript;
      setTranscript(currentTranscript);
    };
  }, [lang, recognition]);

  const startListening = useCallback(() => {
    if (recognition) {
      setError(null);
      try {
        recognition.start();
      } catch (err) {
        console.error(err);
      }
    }
  }, [recognition]);

  const stopListening = useCallback(() => {
    if (recognition) {
      recognition.stop();
    }
  }, [recognition]);

  return { isListening, transcript, startListening, stopListening, error, isSupported: !!recognition };
};
