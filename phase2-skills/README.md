# Phase 2 · Claude Skills / SKILL.md（1 週）

## 三個範例
- `git-commit-writer/` — 入門（純 markdown）
- `csv-analyzer/` — 中階（SKILL.md + scripts/profile.py）
- `company-knowledge-rag/` — 進階（RAG 骨架，Phase 4 補完 scripts）

## Skills vs System Prompt vs Tools vs MCP

| 機制 | 何時載入 | Context 成本 | 能執行程式？ | 典型用途 |
|------|----------|--------------|--------------|----------|
| System prompt | 每次全載 | 高 | ❌ | 角色設定 |
| Tools (function call) | schema 恆駐 | 中 | ✅ 透過 tool | 結構化 API |
| MCP | tool schema 恆駐 | 中 | ✅ 透過 server | 跨廠商連外部系統 |
| Skills | 僅 metadata 恆駐 | 極低 | ✅ bundled scripts | 領域知識、工作流程 |

一句話區分：**Tools = 能力，MCP = 連線，Skills = 知識**。三者互補不替代。

## Description 的 4 個黃金法則
1. 必含「做什麼」+「何時用」
2. 第三人稱（不要 "You can use this to..."）
3. 具體關鍵字（副檔名、工具名、使用者口頭禪）
4. 要 pushy（Claude 傾向 undertrigger）：用 "Use this skill whenever..."

## 部署到 Claude Code
```bash
cp -r git-commit-writer ~/.claude/skills/
# 或專案級
cp -r git-commit-writer .claude/skills/
```

## Phase 2 驗收標準
- [ ] 能畫出 Skills vs System Prompt vs Tools vs MCP 四象限圖
- [ ] 能獨立寫出符合 Anthropic 4 黃金法則的 description
- [ ] 寫出 3 個 skill（入門 + 中階 + RAG 骨架）上傳 Claude Code 實測觸發
- [ ] 能解釋 Progressive Disclosure 三層模型
- [ ] 能 clone `anthropics/skills` 的 `skill-creator` 並用它優化自己的 description
