"""Real LLM backend: a locally running Ollama server.

This is the provider every agent uses outside of tests. It needs:
  1. `ollama serve` running and reachable (default http://localhost:11434)
  2. A model actually pulled: `ollama pull llama3.2` (or any instruction-
     following model — see README for recommended sizes)

Neither of those is available in every environment (this project was built
in a sandbox where the Ollama model registry itself is network-blocked —
see docs/architecture.md#what-was-and-wasnt-run-here) — that's exactly why
every agent node takes an `LLM` (research_assistant.llm.base.LLM) instead
of importing this class directly. Swap in FakeLLM for anything that can't
reach a real model.
"""

from __future__ import annotations

import ollama


class OllamaLLM:
    """Wraps ollama.Client to satisfy the LLM protocol (invoke(prompt) -> str)."""

    def __init__(
        self,
        model: str = "llama3.2",
        host: str = "http://localhost:11434",
        temperature: float = 0.2,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self._client = ollama.Client(host=host)

    def invoke(self, prompt: str) -> str:
        response = self._client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": self.temperature},
        )
        return response["message"]["content"]
