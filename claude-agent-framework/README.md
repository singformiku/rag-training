# Claude Agent Framework

一個用於研究 AI Agent 架構的教學專案，以 Anthropic Claude 的設計哲學為基礎。

> **目的**：讓你從零理解「Agent 是怎麼運作的」，並能自行擴充、實驗、改造。

---

## 🎯 核心概念

這個框架實作了 Claude 的幾個關鍵設計：

| 概念 | 對應檔案 | 說明 |
|------|----------|------|
| **Agentic Loop** | `src/agent.py` | Think → Act → Observe 的循環 |
| **Extended Thinking** | `src/thinking.py` | 回應前的內部推理 |
| **Tool Use** | `src/tools/` | 讓 Agent 能操作外部世界 |
| **Skills System** | `src/skills/` | 以 `SKILL.md` 的形式注入領域知識 |
| **Memory** | `src/memory.py` | 對話歷史與狀態管理 |
| **Human-in-the-loop** | `src/agent.py` | 高風險操作的確認機制 |

---

## 📁 專案結構

```
claude-agent-framework/
├── src/
│   ├── agent.py           # 核心 Agent 類別 (Agentic Loop)
│   ├── thinking.py        # Extended Thinking 模組
│   ├── memory.py          # 對話記憶
│   ├── config.py          # 設定檔
│   ├── tools/             # 工具系統
│   │   ├── base.py        # Tool 基礎類別
│   │   ├── file_tools.py  # 檔案操作
│   │   ├── bash_tool.py   # Shell 執行
│   │   └── search_tool.py # 網路搜尋
│   └── skills/            # Skill 系統
│       ├── loader.py      # 載入 SKILL.md
│       └── registry.py    # Skill 註冊表
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
# 1. 安裝依賴
pip install -r requirements.txt

# 2. 設定 API Key
cp .env.example .env
# 編輯 .env，填入 ANTHROPIC_API_KEY

# 3. 執行範例
python examples/01_basic_agent.py
```

---

## 🧠 Agentic Loop 的核心流程

```
┌─────────────────────────────────────────┐
│  User Query                             │
└──────────────┬──────────────────────────┘
               ▼
      ┌────────────────┐
      │  Think         │ ◄─── Extended Thinking
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

1. `src/agent.py` — 理解主迴圈
2. `src/tools/base.py` — 理解工具介面
3. `src/skills/loader.py` — 理解 Skill 如何被動態載入
4. `examples/` — 從簡到繁看實作範例

---

## 📝 授權

MIT
