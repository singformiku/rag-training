"""
Agent 核心模組
==============
這是整個框架的心臟。實作 Agentic Loop：

┌─────────────────────────────────────────┐
│  User Query                             │
└──────────────┬──────────────────────────┘
               ▼
      ┌────────────────┐
      │  Think         │ ◄─── (gpt-oss-120b) reasoning_effort
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

關鍵設計：
1. 主迴圈由 ``finish_reason`` 控制：
   - "tool_calls" → 執行工具並進入下一輪
   - "stop"       → 任務完成，回傳最終答案
2. 每一輪都把新的 assistant message / tool result 寫進 Memory，
   下一輪 API 呼叫帶著完整歷史
3. 超過 max_iterations 時強制停止 (避免成本失控)
4. requires_confirmation 的工具會在執行前詢問使用者
"""
from __future__ import annotations

import json
from typing import Any, Callable

from src.config import config, settings
from src.memory import Memory
from src.skills.loader import SkillLoader
from src.tools.base import Tool, ToolRegistry

# rich 用於漂亮的輸出，但非必要。裝不了的話退回純文字。
try:
    from rich.console import Console
    from rich.panel import Panel
    console = Console()
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False

    class _PlainConsole:
        def print(self, *args, **_):
            for a in args:
                print(a)

    def Panel(content, title="", border_style="", **_):  # noqa: N802
        return f"\n── {title} ──\n{content}\n"

    console = _PlainConsole()


class Agent:
    """
    基於 ``LLMService`` (OpenAI 相容 API) 的 AI Agent。

    使用範例::

        agent = Agent(
            system_prompt="你是一個 Python 程式碼審查專家",
            tools=[ReadFileTool(), BashTool()],
        )
        result = agent.run("審查 ./src/main.py 的品質")
    """

    def __init__(
        self,
        system_prompt: str = "",
        tools: list[Tool] | None = None,
        skills_dir: str | None = None,
        max_iterations: int | None = None,
        enable_thinking: bool | None = None,
        reasoning_effort: str | None = None,
        confirm_callback: Callable[[str, dict], bool] | None = None,
        verbose: bool = True,
    ):
        """
        Args:
            system_prompt: Agent 的角色設定
            tools: 可用的工具列表
            skills_dir: Skills 資料夾路徑 (會自動掃描 SKILL.md)
            max_iterations: 最大迭代次數
            enable_thinking: 是否要求模型 reasoning (對應 gpt-oss 的 reasoning_effort)
            reasoning_effort: 覆寫預設的推理強度 (low/medium/high)
            confirm_callback: 高風險工具的確認函式，回傳 True 代表允許執行
            verbose: 是否列印詳細執行過程
        """
        # 延遲 import，避免 test 環境沒裝 llm deps 也能 import src.tools
        from llm_service import llm_service

        config.validate()

        self.llm = llm_service
        self.model = settings.LLM_MODEL
        self.max_iterations = max_iterations or config.max_iterations
        self.enable_thinking = (
            enable_thinking if enable_thinking is not None else config.enable_thinking
        )
        self.reasoning_effort = reasoning_effort or config.reasoning_effort
        self.verbose = verbose
        self.confirm_callback = confirm_callback or self._default_confirm

        # 工具註冊
        self.tool_registry = ToolRegistry()
        if tools:
            self.tool_registry.register_all(tools)

        # Skill 載入
        self.skill_loader: SkillLoader | None = None
        if skills_dir:
            self.skill_loader = SkillLoader(skills_dir)
            self.skill_loader.load_all()

        # 組合 system prompt
        self.system_prompt = self._build_system_prompt(system_prompt)

        # 記憶 — 以 system message 開場 (OpenAI 風格)
        self.memory = Memory()
        if self.system_prompt:
            self.memory.set_system_message(self.system_prompt)

    def _build_system_prompt(self, user_prompt: str) -> str:
        """
        組合完整的 system prompt：
        使用者定義的角色 + Skills 摘要 + 通用指引
        """
        parts = []

        if user_prompt:
            parts.append(user_prompt)

        if self.skill_loader:
            skills_summary = self.skill_loader.get_skills_summary()
            if skills_summary:
                parts.append(skills_summary)

        # 通用指引：教模型如何好好使用這個框架
        parts.append(
            "## 工作準則\n"
            "- 在執行工具前，先說明你的計畫\n"
            "- 遇到不確定的情況，寧可多問一步也不要瞎猜\n"
            "- 任務完成後，給出清楚的總結"
        )

        return "\n\n".join(parts)

    def _default_confirm(self, tool_name: str, tool_input: dict) -> bool:
        """預設的確認函式：在 terminal 問使用者 y/n。"""
        console.print(
            Panel(
                f"[yellow]⚠️  即將執行高風險操作[/yellow]\n\n"
                f"Tool: [bold]{tool_name}[/bold]\n"
                f"Input: {tool_input}",
                title="Human-in-the-loop Confirmation",
                border_style="yellow",
            )
        )
        answer = input("允許執行？(y/N): ").strip().lower()
        return answer == "y"

    # ------------------------------------------------------------------
    # 主迴圈
    # ------------------------------------------------------------------
    def run(self, user_query: str) -> str:
        """
        執行一個完整的任務 —— 整個 Agentic Loop 都在這裡展開。
        """
        if self.verbose:
            console.print(Panel(user_query, title="👤 User Query", border_style="blue"))

        self.memory.add_user_message(user_query)

        final_text = ""

        for iteration in range(self.max_iterations):
            if self.verbose:
                console.print(f"\n[dim]── Iteration {iteration + 1} ──[/dim]")

            # 1. 呼叫 LLM
            completion = self._call_llm()
            choice = completion.choices[0]
            message = choice.message
            finish_reason = choice.finish_reason

            # 2. 存回 memory (assistant message，含可能的 tool_calls)
            self.memory.add_assistant_message(message)

            # 3. 顯示 reasoning + content
            self._display_response(message)

            # 4. 根據 finish_reason 決定下一步
            if finish_reason == "tool_calls" or (message.tool_calls and finish_reason != "stop"):
                # OpenAI 有時會回 "stop" 但仍有 tool_calls (罕見)；以 tool_calls 為準
                self._handle_tool_calls(message.tool_calls or [])
                continue

            if finish_reason == "stop":
                final_text = message.content or ""
                return final_text

            # length / content_filter / 其他
            if self.verbose:
                console.print(
                    f"[yellow]⚠️  停止原因：{finish_reason}[/yellow]"
                )
            final_text = message.content or ""
            return final_text

        # 超過最大迭代次數
        if self.verbose:
            console.print(
                f"[red]⚠️  已達最大迭代次數 ({self.max_iterations})，強制停止[/red]"
            )
        return final_text or "任務未完成：超過最大迭代次數"

    # ------------------------------------------------------------------
    # LLM 呼叫 + 工具處理
    # ------------------------------------------------------------------
    def _call_llm(self) -> Any:
        """呼叫底層 LLMService.complete_with_tools。"""
        tools_payload = (
            self.tool_registry.to_api_format()
            if len(self.tool_registry) > 0
            else None
        )

        return self.llm.complete_with_tools(
            messages=self.memory.get_messages(),
            tools=tools_payload,
            tool_choice="auto" if tools_payload else None,
            reasoning_effort=self.reasoning_effort if self.enable_thinking else None,
        )

    def _handle_tool_calls(self, tool_calls: list) -> None:
        """處理模型回應中的所有 tool_calls。"""
        for call in tool_calls:
            # OpenAI 格式：call.id / call.function.name / call.function.arguments (JSON str)
            tool_call_id = call.id
            tool_name = call.function.name
            raw_args = call.function.arguments or "{}"

            try:
                tool_input = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError as e:
                self.memory.add_tool_result(
                    tool_call_id,
                    f"工具參數 JSON 解析失敗：{e}；原始字串：{raw_args!r}",
                    is_error=True,
                )
                continue

            tool = self.tool_registry.get(tool_name)
            if tool is None:
                self.memory.add_tool_result(
                    tool_call_id,
                    f"錯誤：找不到工具 '{tool_name}'",
                    is_error=True,
                )
                continue

            # 高風險工具需要確認
            if tool.requires_confirmation:
                if not self.confirm_callback(tool_name, tool_input):
                    self.memory.add_tool_result(
                        tool_call_id,
                        "❌ 使用者拒絕執行此操作",
                        is_error=True,
                    )
                    continue

            # 執行
            if self.verbose:
                console.print(
                    f"🔧 [cyan]執行工具:[/cyan] [bold]{tool_name}[/bold] "
                    f"[dim]{tool_input}[/dim]"
                )

            try:
                result = tool.execute(**tool_input)
                is_error = False
            except Exception as e:  # noqa: BLE001
                result = f"工具執行錯誤：{e}"
                is_error = True

            if self.verbose:
                preview = result[:200] + ("..." if len(result) > 200 else "")
                console.print(f"[dim]📤 結果：{preview}[/dim]")

            self.memory.add_tool_result(tool_call_id, result, is_error=is_error)

    # ------------------------------------------------------------------
    # 顯示
    # ------------------------------------------------------------------
    def _display_response(self, message: Any) -> None:
        """把模型回應 (reasoning + content + tool_calls) 印出來。"""
        if not self.verbose:
            return

        # gpt-oss / 部分 OpenAI-compatible server 會附加 reasoning_content
        reasoning = getattr(message, "reasoning_content", None) or getattr(
            message, "reasoning", None
        )
        if reasoning:
            console.print(
                Panel(str(reasoning), title="🧠 Reasoning", border_style="magenta")
            )

        if message.content:
            console.print(
                Panel(message.content, title="🤖 Assistant", border_style="green")
            )

        if message.tool_calls:
            for call in message.tool_calls:
                console.print(
                    f"[cyan]↪ tool_call:[/cyan] [bold]{call.function.name}[/bold] "
                    f"[dim]{call.function.arguments}[/dim]"
                )

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def reset(self) -> None:
        """清空對話記憶，保留 system message。"""
        self.memory.clear(keep_system=True)
