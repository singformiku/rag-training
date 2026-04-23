# Claude Agent Framework

一個用於研究 AI Agent 架構的教學專案。

> **目的**：讓你從零理解「Agent 是怎麼運作的」，並能自行擴充、實驗、改造。

> **LLM 後端**：OpenAI-compatible 端點 (預設指向 Dell internal GenAI，
> 模型 `gpt-oss-120b`)，以 OAuth `client_credentials` 認證。入口為 `llm_service.py`。

---

## 🎯 核心概念

| 概念 | 對應檔案 | 說明 |
|------|----------|------|
| **Agentic Loop** | `src/agent.py` | Think → Act → Observe 的循環 |
| **Reasoning** | `src/thinking.py` | gpt-oss 的 `reasoning_effort` / `reasoning_content` |
| **Tool Use** | `src/tools/` | OpenAI function-calling |
| **Skills System** | `src/skills/` | 以 `SKILL.md` 的形式注入領域知識 |
| **Memory** | `src/memory.py` | 對話歷史 (system/user/assistant/tool) |
| **Human-in-the-loop** | `src/agent.py` | 高風險操作的確認機制 |
| **LLM Service** | `llm_service.py` | 統一 LLM 入口，處理 OAuth |

---

## 📁 專案結構

```
claude-agent-framework/
├── llm_service.py         # LLM 入口 (OAuth + OpenAI)
├── src/
│   ├── agent.py           # 核心 Agent 類別 (Agentic Loop)
│   ├── thinking.py        # Reasoning 觀察工具
│   ├── memory.py          # 對話記憶 (OpenAI 訊息格式)
│   ├── config.py          # 設定檔 (settings + config)
│   ├── tools/             # 工具系統 (OpenAI function-calling)
│   │   ├── base.py        # Tool 基礎類別
│   │   ├── file_tools.py  # 檔案操作
│   │   ├── bash_tool.py   # Shell 執行
│   │   └── search_tool.py # 網路搜尋
│   └── skills/            # Skill 系統
│       └── loader.py      # 載入 SKILL.md
├── skills/                # 範例 Skills
│   ├── code_review/
│   ├── data_analysis/
│   └── web_research/
├── examples/              # 使用範例
│   ├── 01_basic_agent.py          # 最簡單的 Agent
│   ├── 02_agent_with_tools.py     # 帶工具的 Agent
│   ├── 03_agent_with_skills.py    # 載入 Skill 的 Agent
│   └── 04_multi_step_task.py      # 複雜多步任務
└── tests/
```

---

## 🚀 快速開始

```bash
# 1. 安裝依賴 (requirements.txt 已配置內部 PyPI 以取 aia-auth-client)
pip install -r requirements.txt

# 2. 設定 credentials
cp .env.example .env
# 編輯 .env，填入 LLM_CLIENT_ID / LLM_SECRET 等

# 3. 執行範例
python examples/01_basic_agent.py
```

### 環境變數速查

| 變數 | 用途 |
|------|------|
| `LLM_URL` | OpenAI-compatible 端點 (`.../v1`) |
| `LLM_MODEL` | 模型名，例如 `gpt-oss-120b` |
| `LLM_CLIENT_ID` / `LLM_SECRET` | OAuth client credentials |
| `LLM_MAX_RESPONSE_TOKENS` | 單次回應 token 上限 |
| `PEM_LOCATION` | (選填) 自訂 CA `.pem` 路徑 |
| `MAX_ITERATIONS` | Agent 迴圈上限 |
| `ENABLE_THINKING` / `REASONING_EFFORT` | 控制 gpt-oss reasoning |

---

## 🧠 Agentic Loop 的核心流程

```
┌─────────────────────────────────────────┐
│  User Query                             │
└──────────────┬──────────────────────────┘
               ▼
      ┌────────────────┐
      │  Think         │ ◄─── reasoning_effort
      │  (Reasoning)   │
      └────────┬───────┘
               ▼
      ┌────────────────┐
      │  Decide Action │
      └────────┬───────┘
               ▼
         ┌─────┴─────┐
         │           │
    [Tool Call]  [Final Answer]
         │           │
         ▼           ▼
    ┌────────┐  ┌────────┐
    │ Execute│  │ Return │
    └────┬───┘  └────────┘
         │
    ┌────▼────┐
    │ Observe │
    └────┬────┘
         │
         └─► Back to Think
```

---

## 🔧 如何擴充

### 新增 Tool

```python
from src.tools.base import Tool

class MyTool(Tool):
    name = "my_tool"
    description = "做某件事"
    input_schema = {
        "type": "object",
        "properties": {
            "param": {"type": "string"}
        }
    }

    def execute(self, param: str) -> str:
        return f"Result: {param}"
```

### 新增 Skill

在 `skills/` 下建立資料夾，放入 `SKILL.md`：

```markdown
---
name: my_skill
description: 這個 skill 在什麼情況該被觸發
---

# My Skill

具體的操作指引...
```

---

## 📚 進一步研究

每個模組都有詳細註解，建議閱讀順序：

1. `llm_service.py` — 理解 LLM 接入 (OAuth + OpenAI)
2. `src/agent.py` — 理解主迴圈 (finish_reason / tool_calls)
3. `src/tools/base.py` — 理解 OpenAI function-calling 介面
4. `src/skills/loader.py` — 理解 Skill 如何被動態載入
5. `examples/` — 從簡到繁看實作範例

---

## 📝 授權

MIT
