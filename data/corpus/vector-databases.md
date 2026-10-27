# Vector Databases

A vector database stores high-dimensional embedding vectors and answers
nearest-neighbor queries efficiently — given a query vector, find the k
stored vectors closest to it by some distance metric (cosine similarity,
dot product, or Euclidean distance). They exist because exact
nearest-neighbor search is linear in the number of stored vectors, which
doesn't scale past a few tens of thousands of items; approximate
nearest-neighbor (ANN) search trades a small amount of recall for
sub-linear query time.

## Indexing algorithms

HNSW (Hierarchical Navigable Small World) is the most widely deployed ANN
algorithm today — it builds a multi-layer graph where each node connects to
its approximate nearest neighbors, and search starts at a sparse top layer
and descends, narrowing in on the target region. It gives strong
recall/latency tradeoffs and supports incremental inserts, which is why
FAISS, Qdrant, Weaviate, and Milvus all offer an HNSW index type. IVF
(Inverted File Index) instead partitions the vector space into clusters
(via k-means) and only searches the clusters nearest the query — faster to
build and more memory-efficient than HNSW at very large scale, at some
recall cost, and often combined with product quantization (IVF-PQ) to
compress vectors further.

## FAISS

FAISS (Facebook AI Similarity Search) is a library, not a standalone
database — it provides index implementations (Flat, IVF, HNSW, PQ
variants) and leaves storage, persistence, and serving to the caller. It's
a common choice for a self-contained application that wants ANN search
without running a separate database process: build an index, add vectors,
serialize it to disk, load it at startup. This tradeoff — library vs.
managed service — is the first real decision in choosing a vector store:
FAISS suits a single-process application with a static or batch-updated
corpus; a dedicated vector database (Qdrant, Weaviate, Pinecone, pgvector
as a Postgres extension) suits a system with concurrent writers, the need
for metadata filtering alongside vector search, or multi-service access to
the same index.

## Metadata filtering and hybrid search

Real retrieval queries are rarely pure vector similarity — a user query
often has structured constraints too ("recent," "from this document type,"
"by this author"). Most production vector databases support filtering
alongside ANN search, either pre-filtering the candidate set before vector
search or post-filtering results after. Combining this with sparse
(BM25/TF-IDF) retrieval scores for hybrid search is increasingly standard,
since pure dense retrieval can miss exact-match queries (an ID, a specific
error code, a proper noun) that lexical search handles trivially.

## Choosing dimensionality

Embedding dimensionality is a real cost tradeoff: higher dimensions
generally capture more semantic nuance but linearly increase storage and
search cost. Modern embedding models range from 384 dimensions (small,
fast, e.g. many sentence-transformers models) to 1536+ (OpenAI's
text-embedding-3-large, Cohere's embed-v3) — the right choice depends on
whether retrieval quality or serving cost is the binding constraint for a
given deployment.
