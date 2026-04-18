"""Hybrid Search + RRF（BM25 + Dense）

pip install rank-bm25 chromadb sentence-transformers jieba

RRF 公式：RRF(d) = Σ 1 / (k + rank_r(d))，k=60（經驗值）
只看 rank 不看 score，不需 normalization。

實測量級（BEIR benchmark 2025）：
  BM25    NDCG@10 ≈ 0.42
  Dense   NDCG@10 ≈ 0.48
  Hybrid  NDCG@10 ≈ 0.53

Anthropic blog：純 dense 失敗率 5.7%，加 BM25 降到 4.1%。

何時 Hybrid 贏：產品 ID、錯誤碼、程式碼、法律條文、財報。
純對話 QA 用 dense 即可。
"""
from collections import defaultdict
import numpy as np
import jieba
from rank_bm25 import BM25Okapi


def tokenize_zh(text):
    return [t for t in jieba.lcut(text.lower()) if t.strip()]


def build_bm25(corpus):
    corpus_tokens = [tokenize_zh(doc) for doc in corpus]
    return BM25Okapi(corpus_tokens), corpus_tokens


def bm25_search(bm25, query, top_k=20):
    scores = bm25.get_scores(tokenize_zh(query))
    return [(i, float(scores[i])) for i in np.argsort(scores)[::-1][:top_k]]


def rrf_fusion(result_lists, k=60, top_k=5):
    """把多個 ranked list 融合成一個"""
    rrf_scores = defaultdict(float)
    for results in result_lists:
        for rank, (doc_id, _) in enumerate(results):
            rrf_scores[doc_id] += 1.0 / (k + rank + 1)
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]


def hybrid_search(query, bm25_fn, dense_fn, top_k=5):
    """bm25_fn / dense_fn 都回傳 [(doc_id, score), ...]"""
    return rrf_fusion(
        [bm25_fn(query, 20), dense_fn(query, 20)],
        top_k=top_k,
    )
