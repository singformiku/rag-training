"""Reranker 四家對比（同一介面）

為什麼需要 reranker：
- 第一階段 bi-encoder 為求速度妥協了精度
- Reranker 是 cross-encoder，精度高 100×、速度慢 100×
- 典型 pipeline：撈 top-100 → rerank → top-10

決策：
- 中文 + 有 GPU → BGE-v2-m3 自架（免費）
- 多語 SaaS → Cohere/Voyage
- 極低延遲 → Jina
- 要 instruction-following → Voyage rerank-2.5
"""
from sentence_transformers import CrossEncoder


_bge = CrossEncoder("BAAI/bge-reranker-v2-m3")


def rerank_bge(query, docs, top_k=5):
    """本地 BGE，~250ms on T4, 免費"""
    scores = _bge.predict([[query, d] for d in docs])
    return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:top_k]


def rerank_cohere(query, docs, top_k=5):
    """Cohere rerank-v3.5，4096 context，$2/M"""
    import cohere
    co = cohere.ClientV2()
    r = co.rerank(model="rerank-v3.5", query=query, documents=docs, top_n=top_k)
    return [(x.index, x.relevance_score) for x in r.results]


def rerank_voyage(query, docs, top_k=5):
    """Voyage rerank-2.5，8K/32K context，200M 免費後 $0.05/M"""
    import voyageai
    vo = voyageai.Client()
    r = vo.rerank(query=query, documents=docs, model="rerank-2.5", top_k=top_k)
    return [(x.index, x.relevance_score) for x in r.results]


def rerank_jina(query, docs, top_k=5):
    """Jina rerank-v2，極低延遲（~180ms）"""
    import requests
    import os
    r = requests.post(
        "https://api.jina.ai/v1/rerank",
        headers={"Authorization": f"Bearer {os.environ['JINA_API_KEY']}"},
        json={"model": "jina-reranker-v2-base-multilingual",
              "query": query, "documents": docs, "top_n": top_k},
    )
    return [(x["index"], x["relevance_score"]) for x in r.json()["results"]]
