# Phase 6 · 畢業專題（1 週）

## Project Spec：`dell-kb-rag`（或你選的主題）

### 題材建議（三選一）
1. **Anthropic docs RAG QA**：公開資料、讀者多、容易被分享
2. **台灣 g0v / 政府開放資料**：差異化、在地化、好故事（中央氣象署 API docs、全國法規資料庫）
3. **Dell-style enterprise ticket assistant**：接近你 ATS 經驗，但用公開資料 mock

### Tech Stack
- Claude Sonnet 4.5（generator）
- Voyage-3.5（embedding）
- Qdrant（vector DB）
- BGE-reranker-v2-m3（rerank，本地）
- Contextual Retrieval（Anthropic prompt caching）
- Langfuse（observability）
- Ragas + 自寫 binary judge（eval）
- Claude Skill + 自建 MCP server 雙包裝

## Repo 結構（recruiter 會看這個）

```
dell-kb-rag/
├── README.md                        # ⭐ 面試官看這個
├── docker-compose.yml               # Langfuse + Qdrant self-host
├── corpus/                          # 你選的領域資料
├── src/
│   ├── rag_pipeline.py              # Contextual + Hybrid + Rerank
│   ├── agent.py                     # Agentic RAG loop
│   ├── judge.py                     # Binary LLM judge
│   └── ragas_runner.py
├── skills/company-knowledge-rag/    # Claude Skill 包裝
│   ├── SKILL.md
│   └── scripts/
├── mcp-server/                      # MCP server 包裝
│   └── server.py
├── eval/
│   ├── golden_set/v1/queries.parquet
│   ├── run_judge.py
│   └── experiments.sqlite           # A/B 結果
├── experiments/
│   └── run_ab.py
├── notebooks/
│   ├── 01_ragas_demo.ipynb
│   ├── 02_error_analysis.ipynb
│   └── 03_ab_results.ipynb
├── .github/workflows/llm-eval.yml
└── docs/
    ├── methodology.md               # 引用 Hamel/Eugene/Anthropic 解釋設計決策
    ├── failure_taxonomy.md          # 你的 open/axial coding 結果
    └── judge_validation_report.md   # κ / TPR / TNR
```

## Blog Post（Hamel 風格）大綱

**標題範例**：
> Building a Production-Grade RAG with Eval-Driven Development:
> Lessons from My Dell Ticket Classifier Experience

1. **Problem statement**（鉤子）：一段故事，為什麼這個 RAG 不 trivial
2. **Architecture diagram**：一張清晰的圖（Excalidraw / Mermaid）
3. **Why this stack**：每個選擇的 trade-off（為什麼 Voyage 而非 OpenAI）
4. **Golden set methodology**：從 100 筆 synthetic + 50 筆 adversarial 起步
5. **LLM judge validation**：binary + critique；κ=0.82；TPR/TNR 表格
6. **A/B results**：12-config leaderboard + Pareto plot
7. **Key findings**：具體數字（faithfulness 從 0.68 → 0.89）
8. **What I'd do differently**：agent eval、production online eval、cost optimization
9. **Code**：GitHub link + Loom 3 分鐘 demo

## 必放 Screenshots
1. Architecture diagram
2. Langfuse trace 截圖（retrieval span + generation span 含 token usage）
3. A/B leaderboard table
4. Cost-quality Pareto scatter plot
5. Confusion matrix（judge vs human）
6. GitHub Actions 紅燈 PR 截圖（證明 CI 有效）
7. Failure mode pivot chart

## 求職 Signal 最大化
- **LinkedIn post**：宣布 repo，tag Hamel / Eugene / Jason Liu
- **Hacker News Show HN**：「Show HN: Production RAG with eval harness, built in 8 weeks」
- **HackerNews / Reddit r/LocalLLaMA**：技術社群擴散
- **面試 30 秒 pitch**：
  > 我在 Dell 4 年做 PAW/ATS/C2 classifier，用 LLM-as-judge 處理 enterprise noisy
  > ticket data，知道 production LLM 的地基是 eval。這 2 個月我把這套方法系統化...
