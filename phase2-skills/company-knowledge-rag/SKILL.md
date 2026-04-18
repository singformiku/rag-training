---
name: company-knowledge-rag
description: Answers questions about internal company knowledge (policies, HR, engineering wiki) by retrieving from the company's vector store and citing sources. Use whenever the user asks about "our company's X policy", "how do we do X at our company", who owns a system, project post-mortems, or any "where is the doc for Y". Always use this skill BEFORE answering from general knowledge when the question has a company-specific angle. Do NOT use for public knowledge or general coding help.
compatibility: Requires QDRANT_URL + QDRANT_API_KEY env vars
---

# Company Knowledge RAG

## Standard Workflow
1. **Query rewrite**（上下文不足時合併對話脈絡）
2. **Retrieve**: `python scripts/retrieve.py --query "..." --top-k 10 --collection company_kb`
3. **Rerank**（k>5 時）: `python scripts/rerank.py --input <retrieve.json> --top-n 5`
4. **Compose**: 每個陳述都要 citation（見 `references/citation-format.md`）；找不到支持證據就說「資料不足」。
5. **Quality check**: 每段是否有 citation？URL 是否真存在於 retrieve 結果？是否真的回答了問題？
