"""Deterministic test double for LLM, NOT a model.

This exists so the graph's actual logic — routing, state updates, the
critic's loop back to the researcher, prompt construction — can be
verified with real assertions in tests and CI, without needing network
access to pull model weights or an API key. It is scripted, not
intelligent: it matches a marker substring in the prompt (every node's
prompt starts with "ROLE: <name>" specifically so this can match on it)
and returns a canned response for that role, advancing through a list of
responses on repeated calls to the same role if one was provided.

Never used by research_assistant.api.app in normal operation — that always
constructs an OllamaLLM. See tests/ for how this drives real assertions
about graph behavior (e.g. tests/test_graph_end_to_end.py exercises the
critic's loop-back path by scripting an INSUFFICIENT verdict followed by
a SUFFICIENT one).
"""

from __future__ import annotations


class FakeLLM:
    def __init__(self, responses: dict[str, str | list[str]], default: str = "") -> None:
        self._responses = responses
        self._default = default
        self._call_index: dict[str, int] = {}
        self.calls: list[str] = []

    def invoke(self, prompt: str) -> str:
        self.calls.append(prompt)
        for marker, response in self._responses.items():
            if marker not in prompt:
                continue
            if isinstance(response, str):
                return response
            i = self._call_index.get(marker, 0)
            self._call_index[marker] = min(i + 1, len(response) - 1)
            return response[i]
        return self._default
