# Phase 4 · Agentic RAG + Claude Skills + MCP 整合（2–3 週）

## 檔案清單
- `agent_rag.py` — Agentic RAG agent loop（完整可執行）
- `anthropic-docs-search/` — 把 RAG 包成 Claude Skill
- `docs-rag-mcp/` — 自建 MCP server（stdio transport）

## 三層決策（什麼時候用什麼）
- 要連「外部系統」（DB、GitHub、Slack）→ **MCP**
- 要教 Claude「怎麼做一件事」→ **Skill**
- 單一 Python 一次用 → **Tool Use**

## MCP 三大 primitives
- **Tools**（可執行，像 POST）
- **Resources**（唯讀，像 GET）
- **Prompts**（可重用模板）

Transport：stdio（本機）、Streamable HTTP（遠端）、SSE（legacy）
最新 spec 2025-11-25 加入 OAuth 2.1、Elicitation、URL mode

## 部署 MCP Server

### macOS (Claude Desktop)
編輯 `~/Library/Application Support/Claude/claude_desktop_config.json`
（參考 `docs-rag-mcp/claude_desktop_config.json`）

**必須絕對路徑**；存檔後完整關閉再重啟 Claude Desktop；
右下角看到 🔨 工具圖示 → 成功。
Debug log 在 `~/Library/Logs/Claude/mcp-server-docs-rag.log`。

## 常見坑
1. `print` 到 stdout → 壞了 JSON-RPC → 全改 `logging` 到 stderr
2. path 沒指到 venv python → `ModuleNotFoundError` → 用絕對路徑
3. JSON 格式錯 → `python -m json.tool claude_desktop_config.json` 驗證

## Production 推薦架構
```
User → Claude Desktop/Code
  ├─ Skills（程序性知識）
  │   ├─ code-review
  │   ├─ rag-procedure
  │   └─ report-template
  └─ MCP Servers（外部連線）
      ├─ github
      ├─ docs-rag（你寫的）
      ├─ postgres
      └─ slack
```

## Phase 4 驗收標準
- [ ] `agent_rag.py` 能跑通，換成自己的文件做 10 題測試
- [ ] 實作 `decompose_question` tool 處理 multi-hop
- [ ] 寫一個 Skill 上傳 Claude Code 並成功觸發
- [ ] 自建 MCP server 掛進 Claude Desktop，能自然對話呼叫
- [ ] 能講清楚 Skills vs MCP vs Tools 的三層決策
