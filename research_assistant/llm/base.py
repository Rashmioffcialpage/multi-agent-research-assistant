"""LLM provider interface every agent node depends on.

Every node in research_assistant.graph.nodes takes an LLM instance through
this Protocol, never a concrete provider. That's what makes it possible to
run the exact same graph against a deterministic FakeLLM in tests/CI and
against a real Ollama model on a machine that has one pulled — the graph
logic doesn't know or care which one it's talking to.
"""

from __future__ import annotations

from typing import Protocol


class LLM(Protocol):
    """Minimal interface: give it a prompt, get text back."""

    def invoke(self, prompt: str) -> str: ...
