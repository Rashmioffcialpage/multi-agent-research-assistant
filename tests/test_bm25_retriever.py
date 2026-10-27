from research_assistant.retrieval.bm25_retriever import BM25Retriever


def test_loads_all_corpus_documents(retriever: BM25Retriever):
    doc_ids = {p.doc_id for p in retriever.search("the", k=1000)}
    assert len(doc_ids) >= 10  # every corpus doc should be reachable by some query


def test_retrieves_the_relevant_document_for_a_targeted_query(retriever: BM25Retriever):
    results = retriever.search("FedAvg federated averaging rounds clients", k=3)
    assert results, "expected at least one result"
    assert results[0].doc_id == "federated-learning"


def test_results_are_sorted_by_score_descending(retriever: BM25Retriever):
    results = retriever.search("quantization inference memory bandwidth", k=5)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_irrelevant_query_returns_no_results(retriever: BM25Retriever):
    # Genuinely made-up words with no vocabulary overlap with the corpus —
    # an earlier version of this test used "nonexistent_token", which
    # accidentally shares real vocabulary ("token") with
    # embedding-models.md and produced a nonzero score. Caught by actually
    # running the test, not assumed to be right.
    results = retriever.search("qwxzzyfrobnicate wibbleplonk zvorqix", k=5)
    assert results == []


def test_k_limits_result_count(retriever: BM25Retriever):
    results = retriever.search("model training data evaluation", k=2)
    assert len(results) <= 2
