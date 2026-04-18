# Phase 1 · RAG 基礎實作（1–2 週）

## 檔案清單
- `rag_demo.py` — 最小可跑 RAG demo（Claude Sonnet 4.5 + Voyage-3.5 + ChromaDB）
- `chunking_strategies.py` — 四種 chunking 策略（fixed / recursive / markdown-aware / semantic）
- `embedders.py` — Voyage / BGE 統一抽象層
- `qdrant_store.py` — Qdrant production-grade vector store

## 環境準備
```bash
python -m venv venv && source venv/bin/activate
pip install -r ../requirements.txt

export ANTHROPIC_API_KEY="sk-ant-..."
export VOYAGE_API_KEY="pa-..."   # Voyage 新帳號 200M tokens 免費
```

## 執行
```bash
python rag_demo.py
```

## Phase 1 驗收標準
- [ ] 能在 30 分鐘內從零跑起 `rag_demo.py` 看到答案
- [ ] 能在白板畫出 naive/advanced/agentic 三代 RAG 並講 trade-off
- [ ] 能在 Chroma/Qdrant/sqlite-vec 間切換同一 pipeline
- [ ] 能解釋 Voyage vs OpenAI vs BGE 的選型原則
- [ ] 三種 chunking 策略都跑過一次，能說出何時用哪個
- [ ] 寫 10 題 synthetic Q&A 做 Recall@4 量測
