"""Optional web search tool, via the `ddgs` package (DuckDuckGo, no API key).

Not used by default — the graph's default tool set is corpus_search +
calculator, both of which work with zero network access. This exists for
questions the local corpus genuinely can't answer, on a machine with normal
internet access.

Not exercised by any test in this repo: this project was built in a
sandbox whose network policy blocks every search-engine and search-API
host tested (DuckDuckGo's HTML endpoint, its JSON API, Wikipedia's API) —
see docs/architecture.md#what-was-and-wasnt-run-here. `ddgs` is a
maintained, widely used library, and this wrapper is a thin, direct use of
its documented API — but "the library's contract is stable" and "this
call was actually exercised end to end" are different claims, and only
the first one is true here. Treat it as needing real verification on your
own machine before relying on it.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WebResult:
    title: str
    url: str
    snippet: str


def web_search(query: str, max_results: int = 5) -> list[WebResult]:
    from ddgs import DDGS  # imported lazily so this optional path never costs an import at startup

    with DDGS() as ddgs:
        raw_results = list(ddgs.text(query, max_results=max_results))

    return [
        WebResult(
            title=r.get("title", ""),
            url=r.get("href", r.get("link", "")),
            snippet=r.get("body", r.get("snippet", "")),
        )
        for r in raw_results
    ]


def format_web_results(results: list[WebResult]) -> str:
    if not results:
        return "(no web results found)"
    blocks = [f"[{i}] {r.title} ({r.url})\n{r.snippet}" for i, r in enumerate(results, start=1)]
    return "\n\n".join(blocks)
