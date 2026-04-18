# Phase 3 · 進階 RAG 技術（2–3 週）

## 檔案清單
- `hybrid_search.py` — BM25 + Dense + RRF
- `rerankers.py` — BGE / Cohere / Voyage / Jina 四家對比
- `query_transformation.py` — HyDE / Decomposition / Multi-query / Step-back
- `contextual_retrieval.py` — Anthropic 旗艦：Contextual Embeddings + Prompt Caching
- `self_query.py` — Metadata filtering（Instructor + Pydantic）

## Long-context RAG vs Traditional RAG 決策表

| 總資料量 | 推薦 |
|----------|------|
| <100K tokens、單次查詢 | Long-context + prompt caching |
| 100–200K、重複查詢 | Long-context + 1h caching |
| 200K–2M、少更新 | RAG 或 Hybrid（RAG 撈 top-30 → 200K context）|
| >2M | RAG 必須 |
| 需要 citation | RAG |
| 資料常更新 | RAG |
| 成本敏感 | RAG（long-context 貴 30–60×）|

成本實測（2M 知識庫）：
- Pure Long-context ~$6/query
- Pure RAG + Haiku ~$0.005
- Hybrid ~$0.30

**99% 生產場景用 RAG 或 Hybrid。**

## Phase 3 驗收標準
- [ ] 跑完 BM25 vs Dense vs Hybrid 對比，畫出 precision/recall 圖
- [ ] 本地跑過 bge-reranker-v2-m3；API 呼叫過 Cohere/Voyage 至少兩家
- [ ] 四種 Query Transformation 都實作一次
- [ ] 用 prompt caching 跑 Contextual Retrieval，對 10 chunks 驗證 cache hit
- [ ] 用 Instructor + Pydantic 實作 self-query（至少 3 個 filter 欄位）
- [ ] 做一次 pure RAG vs hybrid long-context 的 latency & cost 比較
