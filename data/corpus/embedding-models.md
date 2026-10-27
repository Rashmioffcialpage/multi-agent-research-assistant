# Embedding Models

An embedding model maps text (a word, sentence, or document) to a
fixed-length vector of real numbers such that semantically similar inputs
map to nearby vectors, under some distance metric (usually cosine
similarity). This is the foundation dense retrieval, semantic search, and
clustering are built on — it turns "how similar are these two pieces of
text in meaning" into "how close are these two vectors," a question
that's cheap to compute and indexable at scale.

## How they're trained

Most modern text embedding models are trained with contrastive learning:
the model sees pairs (or triplets) of text known to be semantically
related — a query and a relevant passage, two paraphrases, an anchor with
a matched positive and a mismatched negative — and is trained so that
related pairs end up close in embedding space while unrelated pairs end up
far apart. The training signal typically comes from a contrastive loss
(InfoNCE and variants) that pushes the positive pair's similarity up and
similarity to sampled negatives down within the same batch, which is why
batch size and negative sampling strategy matter a lot for embedding model
training quality — more, harder negatives per batch generally produce
better embeddings.

## Sentence-level vs. token-level

A word embedding (word2vec, GloVe) gives one fixed vector per vocabulary
word, with no context sensitivity — "bank" gets the same vector whether it
means a riverbank or a financial institution. Contextual embeddings
(everything built on transformers) instead produce a different vector for
the same word depending on surrounding context. For retrieval, what's
usually needed is a single vector per chunk of text, not one per token —
sentence/document embedding models typically pool token-level
representations (mean pooling over the final layer, or a dedicated \[CLS]-
style token trained specifically to summarize the whole sequence) into one
fixed-size vector per input.

## Symmetric vs. asymmetric search

Some tasks are symmetric — the query and the target are the same kind of
text (finding a similar sentence to a given sentence), and a model trained
for general similarity works well. Search is usually asymmetric: a short
query needs to match a longer passage that answers it, and the two aren't
naturally "similar" as text even when the passage is a perfect answer.
Models intended for retrieval are trained specifically on
query-passage pairs to handle this asymmetry, often with separate prefix
tokens or instructions like "query:" / "passage:" prepended so the model
knows which role it's encoding for a given input.

## Dimensionality and Matryoshka embeddings

A fixed embedding dimension is a fixed cost — every stored vector takes
that much space regardless of how much of that space is actually needed
for a given retrieval task's difficulty. Matryoshka representation
learning trains embeddings so that a *truncated* prefix of the full vector
(the first 256 dimensions of a 1024-dimensional embedding, for instance)
is still a valid, useful embedding on its own — letting a single trained
model serve multiple storage/quality tradeoffs by simply truncating at
query time, instead of needing to train and maintain separate models at
each target dimensionality.
