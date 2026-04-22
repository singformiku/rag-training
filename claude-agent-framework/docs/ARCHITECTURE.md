# 架構與設計決策

這份文件解釋框架的設計哲學和關鍵取捨。

---

## 1. 為什麼用 Agentic Loop 而不是預先規劃？

**選擇：動態決策 (每輪讓 Claude 自己決定下一步)**

❌ 不採用：先讓 Claude 生出完整任務計畫，再逐一執行

原因：
- 預先規劃容易在資訊不足時就做出錯誤決定
- 現實世界充滿意外（檔案不存在、API 回傳空值），硬照計畫走會失敗
- Claude 本身的推理能力夠強，讓它在每一步根據最新資訊決策更有彈性

**取捨**：動態決策可能繞路，所以用 `max_iterations` 強制上限，避免無限循環。

---

## 2. 為什麼 Skill 要放在外部 Markdown，而不是寫在 system prompt？

**選擇：SKILL.md 動態載入**

原因：
- **Context 效率**：一次性把所有 skill 塞進 system prompt 會浪費 token。只在需要時載入對應 SKILL.md 比較經濟。
- **可維護性**：非工程師也能編輯 SKILL.md
- **可組合性**：同一個 Agent 可以搭配不同的 skill 組合

**實作方式**：
1. 啟動時掃描 skills 目錄，只把 `name + description` 摘要塞入 system prompt
2. Agent 判斷任務符合某 skill 時，用 `read_file` 工具載入完整 SKILL.md
3. 讀到的指引會進入 memory，後續輪次都能參考

---

## 3. 為什麼工具要有 `requires_confirmation`？

**選擇：危險工具預設需要人工確認**

這對應 Claude 真實產品中的「tool permission」機制。

原因：
- LLM 有機率產生錯誤操作（例如刪錯檔案、執行錯誤的 SQL）
- 使用者對「可逆操作」和「不可逆操作」的容忍度不同
- `read_file` 讀錯沒關係，但 `rm -rf` 執行錯就完了

**實作**：
- `Tool.requires_confirmation = True` 標示高風險工具
- `Agent` 在執行前呼叫 `confirm_callback`（預設為 terminal 問 y/N）
- 可替換成 Slack bot、GUI 彈窗、email 等

---

## 4. Memory 為什麼只存最近 N 條訊息？

**選擇：簡單的滑動窗口 (sliding window)**

原因：
- 對教學用的 framework，實作簡單易懂最重要
- Anthropic API 的 context window 雖然很大 (200K tokens)，但貴
- 真實生產環境應該搭配 **向量檢索** 或 **階層式記憶**

**後續擴充方向**：
- Short-term memory：保留最近 N 輪（目前的實作）
- Working memory：保留本次任務的所有內容
- Long-term memory：用 embedding 存到向量資料庫，需要時檢索

---

## 5. 為什麼要顯式的 Extended Thinking？

**選擇：呼叫 API 時傳入 `thinking` 參數**

原因：
- 複雜任務時讓模型「想清楚再回答」，品質有顯著差異
- Thinking 內容對開發者除錯極有幫助（可以看到模型的推理過程）
- 這是 Claude 獨有的功能，其他模型不一定有等價機制

**代價**：
- Latency 上升
- Output token 變多（thinking 也算 token）
- 小任務不划算

因此 `enable_thinking` 設計成可切換。

---

## 6. Tool Use 為什麼要用 JSON Schema？

**選擇：每個工具都要聲明 `input_schema`**

原因：
- Claude 用 schema 來驗證要生成的參數
- 讓工具對 Claude 而言是「自我說明」的 —— 不需要額外的文件
- 對使用者也是好事：看 schema 就知道怎麼用

這個設計直接對應 Anthropic API 的 tool use 規範。

---

## 7. 與其他框架的差異

| 特性 | 本框架 | LangChain | CrewAI | AutoGen |
|------|--------|-----------|--------|---------|
| 複雜度 | 低 (~600 行) | 高 | 中 | 高 |
| 抽象層 | 薄 | 厚 | 中 | 厚 |
| Agent 數 | 單 Agent | 單/多 | 多 Agent | 多 Agent |
| Claude 專屬特性 | ✅ Skill, Thinking | ❌ | ❌ | ❌ |
| 適合 | 研究、自行擴充 | 快速原型 | 角色協作 | 研究多 Agent |

本框架刻意保持薄、小、可讀，目的是讓你**真的理解每一行在做什麼**，
而不是黑盒子化的「一行啟動一個 Agent」。

---

## 8. 可能的延伸方向

- [ ] **Multi-agent 支援**：一個 Planner Agent + 多個 Worker Agent
- [ ] **Streaming 回應**：即時串流 thinking 和 text
- [ ] **長期記憶**：整合向量資料庫
- [ ] **Tool chain**：工具可以呼叫其他工具
- [ ] **Token 成本追蹤**：累計 usage
- [ ] **錯誤重試機制**：API 失敗時自動 retry
- [ ] **並行工具呼叫**：同一輪的獨立工具呼叫平行執行
- [ ] **MCP (Model Context Protocol) 支援**：串接 Anthropic 生態的外部工具
