"""BM25 lexical retrieval over a local markdown corpus.

Chosen as the default (not a dense/embedding retriever) for a concrete
reason, not just convenience: BM25 needs no pretrained weights and no
network access to build or query, so it works identically in this
sandbox, in CI, and on a user's machine with no internet at all — and
hybrid sparse+dense retrieval is standard practice in real RAG systems
anyway (see data/corpus/retrieval-augmented-generation.md), not a
compromise unique to this project. Swapping in a dense retriever later
(e.g. sentence-transformers + FAISS) means adding a second Retriever
implementation beside this one, not replacing it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass(frozen=True)
class Passage:
    doc_id: str
    heading: str
    text: str
    score: float = 0.0


def _split_into_sections(markdown: str) -> list[tuple[str, str]]:
    """Splits a markdown doc into (heading, section_text) chunks on '## ' headers.

    The doc's own '# Title' line plus any text before the first '## '
    becomes one section using the title as its heading.
    """
    lines = markdown.splitlines()
    sections: list[tuple[str, str]] = []
    heading = "Introduction"
    buf: list[str] = []

    def flush() -> None:
        text = "\n".join(buf).strip()
        if text:
            sections.append((heading, text))

    for line in lines:
        if line.startswith("# ") and not buf:
            heading = line[2:].strip()
            continue
        if line.startswith("## "):
            flush()
            heading = line[3:].strip()
            buf = []
            continue
        buf.append(line)
    flush()
    return sections


class BM25Retriever:
    def __init__(self, corpus_dir: Path | str) -> None:
        self.corpus_dir = Path(corpus_dir)
        self._passages: list[Passage] = []
        self._load()
        tokenized = [_tokenize(p.text) for p in self._passages]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _load(self) -> None:
        for path in sorted(self.corpus_dir.glob("*.md")):
            doc_id = path.stem
            markdown = path.read_text(encoding="utf-8")
            for heading, text in _split_into_sections(markdown):
                self._passages.append(Passage(doc_id=doc_id, heading=heading, text=text))

    def search(self, query: str, k: int = 4) -> list[Passage]:
        if self._bm25 is None:
            return []
        scores = self._bm25.get_scores(_tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        results: list[Passage] = []
        for i in ranked[:k]:
            if scores[i] <= 0:
                break
            p = self._passages[i]
            results.append(
                Passage(doc_id=p.doc_id, heading=p.heading, text=p.text, score=float(scores[i]))
            )
        return results

    def __len__(self) -> int:
        return len(self._passages)
