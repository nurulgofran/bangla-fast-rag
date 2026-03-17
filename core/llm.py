"""
Groq LLM integration — used for Q1 responses and complex queries.
NOT used in the <100ms hot path.
"""
from groq import Groq

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import GROQ_API_KEY, GROQ_MODEL, GROQ_MAX_TOKENS, GROQ_TEMPERATURE


class GroqLLM:
    """Groq API wrapper for LLM generation."""

    def __init__(self):
        self.client: Groq | None = None
        self.model = GROQ_MODEL

    def _ensure_client(self) -> None:
        if self.client is None:
            if not GROQ_API_KEY:
                raise ValueError(
                    "GROQ_API_KEY not set. Copy .env.example to .env and add your key."
                )
            self.client = Groq(api_key=GROQ_API_KEY)

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int = GROQ_MAX_TOKENS,
        temperature: float = GROQ_TEMPERATURE,
    ) -> str:
        """Generate a response using Groq API."""
        self._ensure_client()

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content.strip()

    def generate_with_history(
        self,
        system_prompt: str,
        history: list[dict],
        user_query: str,
        max_tokens: int = GROQ_MAX_TOKENS,
    ) -> str:
        """Generate response with conversation history."""
        self._ensure_client()

        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_query})

        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=GROQ_TEMPERATURE,
        )
        return response.choices[0].message.content.strip()


# Global instance
groq_llm = GroqLLM()
