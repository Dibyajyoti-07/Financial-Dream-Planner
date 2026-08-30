import os
from pathlib import Path

from rag.embeddings import LocalEmbeddings  # import before chromadb - avoids a Windows torch/onnxruntime DLL clash

from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

PERSIST_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "financial_planner_kb"
MIN_SIMILARITY = float(os.getenv("RAG_MIN_SIMILARITY", "0.35"))
RERANK_MIN_SCORE = float(os.getenv("RAG_RERANK_MIN_SCORE", "-3.0"))
RERANK_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RRF_K = 60
CANDIDATE_POOL = 10

_store = None
_bm25 = None
_bm25_docs = None
_reranker = None


def _tokenize(text):
    return text.lower().split()


def _load_store():
    global _store, _bm25, _bm25_docs
    if _store is None:
        if not Path(PERSIST_DIR).exists():
            raise FileNotFoundError(f"{PERSIST_DIR} not found - run rag/build_vector_store.py first")
        _store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=LocalEmbeddings(),
            persist_directory=PERSIST_DIR,
        )
        raw = _store.get(include=["documents"])
        _bm25_docs = raw["documents"]
        _bm25 = BM25Okapi([_tokenize(d) for d in _bm25_docs])
    return _store


def _get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(RERANK_MODEL_NAME)
    return _reranker


def is_loaded():
    try:
        _load_store()
        return True
    except FileNotFoundError:
        return False


def warmup():
    _load_store()
    _get_reranker()


def _semantic_search(query, k):
    store = _load_store()
    results = store.similarity_search_with_relevance_scores(query, k=k)
    return [(doc.page_content, score) for doc, score in results]


def _bm25_search(query, k):
    _load_store()
    scores = _bm25.get_scores(_tokenize(query))
    ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
    return [_bm25_docs[i] for i in ranked]


def _reciprocal_rank_fusion(semantic_docs, bm25_docs, rrf_k=RRF_K):
    scores = {}
    for rank, (doc, _score) in enumerate(semantic_docs):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank + 1)
    for rank, doc in enumerate(bm25_docs):
        scores[doc] = scores.get(doc, 0.0) + 1.0 / (rrf_k + rank + 1)
    return sorted(scores.keys(), key=lambda d: scores[d], reverse=True)


def retrieve(query, k=3):
    semantic_results = _semantic_search(query, CANDIDATE_POOL)
    if not any(score >= MIN_SIMILARITY for _doc, score in semantic_results):
        return []

    bm25_results = _bm25_search(query, CANDIDATE_POOL)
    fused = _reciprocal_rank_fusion(semantic_results, bm25_results)
    if not fused:
        return []

    reranker = _get_reranker()
    pairs = [[query, doc] for doc in fused]
    rerank_scores = reranker.predict(pairs)
    ranked = sorted(zip(fused, rerank_scores), key=lambda pair: pair[1], reverse=True)

    return [doc for doc, score in ranked if score >= RERANK_MIN_SCORE][:k]
