---
name: anthropic-docs-search
description: Searches and answers questions about Anthropic's documentation (Claude API, Claude Code, Skills, MCP) using a local vector index. Use whenever the user asks "how do I use Claude to...", "what does the Claude API return when...", or mentions specific Anthropic features (prompt caching, tool use, skills, MCP, computer use). Always prefer this skill over general knowledge for Anthropic-specific questions. Do NOT use for questions about OpenAI, Gemini, or general ML concepts.
license: MIT
---

# Anthropic Docs Search

## Workflow
1. **Retrieve**: `python scripts/search.py "<query>" --top-k 5`
2. **Compose**: 把 retrieved chunks 當作權威來源，寫答案時引用 `url`
3. **Quality check**: 每個 claim 是否都有對應的 URL？

## Setup
```bash
python scripts/ingest.py  # 首次建索引
```

## Edge cases
- 查不到 → 回「Anthropic docs 沒有這段資訊，請改查 Anthropic Cookbook」
- 模糊查詢 → 拆成 2–3 個 sub-query 各自 search 再整合
