# Retrieval-Augmented Generation (RAG)

RAG combines a retrieval step over an external knowledge source with a
generation step from a language model, so the model's answer is grounded
in retrieved evidence rather than relying only on what it memorized during
pretraining. The core motivation is twofold: language models have a
training cutoff and can't know about anything more recent, and they
hallucinate — produce fluent, confident, wrong answers — especially on
long-tail facts that appeared rarely (or never) in training data. Grounding
generation in retrieved passages, and asking the model to cite them,
reduces both problems without retraining the model itself.

## The basic pipeline

A RAG system has three stages. First, indexing: documents are split into
chunks (a few hundred tokens each is typical — small enough to be specific,
large enough to preserve context) and stored in a retrievable index.
Second, retrieval: given a query, the system finds the most relevant
chunks. Third, generation: the retrieved chunks are inserted into the
model's prompt as context, and the model generates an answer conditioned
on both the query and that context.

## Retrieval methods

Retrieval can be sparse, dense, or hybrid. Sparse retrieval (TF-IDF, BM25)
scores chunks by lexical term overlap with the query — fast, needs no
training or GPU, and is surprisingly strong on queries that share
vocabulary with the source text, but misses semantic matches that use
different words for the same concept. Dense retrieval embeds both the
query and chunks into a shared vector space with a trained embedding model
and ranks by cosine similarity or dot product — it captures semantic
similarity across paraphrases, at the cost of needing an embedding model
and a vector index (see vector-databases.md). Hybrid retrieval runs both
and combines the rankings (e.g., reciprocal rank fusion), which in
practice tends to outperform either alone, since sparse and dense methods
tend to fail on different query types.

## Multi-hop and agentic RAG

Simple RAG does one retrieval pass and generates. Many real questions need
more: a multi-part question might require several distinct retrievals, and
the first retrieval's results might reveal that a different or additional
query is needed. Agentic RAG addresses this by giving the model a
retrieval tool it can call repeatedly and a loop that lets it decide when
it has enough evidence — a critic step evaluates whether the gathered
evidence is sufficient to answer, and if not, triggers another retrieval
with a refined or additional query before generation proceeds. This is
strictly more expensive (more retrieval + more LLM calls) but handles
compound questions single-pass RAG can't.

## Evaluation

RAG systems are evaluated on both retrieval quality (precision/recall of
relevant chunks, e.g. recall@k) and generation quality (faithfulness — does
the answer only claim what the retrieved context supports; answer
relevance; and correctness against a ground-truth answer). Frameworks like
RAGAS and DeepEval automate this with an LLM-as-judge approach, scoring
faithfulness and relevance without requiring hand-labeled ground truth for
every query.
