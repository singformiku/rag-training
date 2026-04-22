"""
Agent 核心模組
==============
這是整個框架的心臟。實作了 Claude 的 Agentic Loop：

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

關鍵設計：
1. 主迴圈由 `stop_reason` 控制：當 Claude 回傳 "tool_use" 時繼續，
   回傳 "end_turn" 時代表任務完成
2. 每個工具呼叫都會被記錄到 Memory，下一輪 API 呼叫會帶著完整歷史
3. 超過 max_iterations 時強制停止 (避免成本失控)
4. requires_confirmation 的工具會在執行前詢問使用者
"""
from __future__ import annotations

from typing import Any, Callable
import anthropic

from src.config import config
from src.memory import Memory
from src.tools.base import Tool, ToolRegistry
from src.skills.loader import SkillLoader


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
    一個基於 Claude 的 AI Agent。

    使用範例:
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
        confirm_callback: Callable[[str, dict], bool] | None = None,
        verbose: bool = True,
    ):
        """
        初始化 Agent。

        Args:
            system_prompt: Agent 的角色設定
            tools: 可用的工具列表
            skills_dir: Skills 資料夾路徑 (會自動掃描 SKILL.md)
            max_iterations: 最大迭代次數
            enable_thinking: 是否開啟 Extended Thinking
            confirm_callback: 高風險工具的確認函式，回傳 True 代表允許執行
            verbose: 是否列印詳細執行過程
        """
        config.validate()

        self.client = anthropic.Anthropic(api_key=config.api_key)
        self.model = config.model
        self.max_iterations = max_iterations or config.max_iterations
        self.enable_thinking = (
            enable_thinking if enable_thinking is not None else config.enable_thinking
        )
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

        # 記憶
        self.memory = Memory()

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

        # 通用指引：教 Claude 如何好好使用這個框架
        parts.append(
            "## 工作準則\n"
            "- 在執行工具前，先說明你的計畫\n"
            "- 遇到不確定的情況，寧可多問一步也不要瞎猜\n"
            "- 任務完成後，給出清楚的總結"
        )

        return "\n\n".join(parts)

    def _default_confirm(self, tool_name: str, tool_input: dict) -> bool:
        """
        預設的確認函式：在 terminal 問使用者 y/n。
        可替換成 GUI 彈窗、Slack 訊息等。
        """
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

    def run(self, user_query: str) -> str:
        """
        執行一個完整的任務。

        這是主入口。整個 Agentic Loop 都在這裡展開。
        """
        if self.verbose:
            console.print(Panel(user_query, title="👤 User Query", border_style="blue"))

        self.memory.add_user_message(user_query)

        for iteration in range(self.max_iterations):
            if self.verbose:
                console.print(f"\n[dim]── Iteration {iteration + 1} ──[/dim]")

            # 1. 呼叫 Claude API
            response = self._call_claude()

            # 2. 把 Claude 的回應存進記憶
            self.memory.add_assistant_message(response.content)

            # 3. 印出 Claude 的思考和回應
            self._display_response(response)

            # 4. 根據 stop_reason 決定下一步
            if response.stop_reason == "end_turn":
                # 任務完成，回傳最終答案
                return self._extract_text(response.content)

            if response.stop_reason == "tool_use":
                # 需要執行工具
                self._handle_tool_calls(response.content)
                # 接著進入下一輪迴圈
                continue

            # 其他 stop_reason (max_tokens, stop_sequence 等)
            if self.verbose:
                console.print(f"[yellow]⚠️  停止原因：{response.stop_reason}[/yellow]")
            break

        # 超過最大迭代次數
        if self.verbose:
            console.print(
                f"[red]⚠️  已達最大迭代次數 ({self.max_iterations})，強制停止[/red]"
            )
        return "任務未完成：超過最大迭代次數"

    def _call_claude(self) -> Any:
        """呼叫 Anthropic API。"""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": 4096,
            "system": self.system_prompt,
            "messages": self.memory.get_messages(),
        }

        # 有工具才傳 tools 參數
        if len(self.tool_registry) > 0:
            kwargs["tools"] = self.tool_registry.to_api_format()

        # Extended Thinking (使用 Claude 的內建推理機制)
        # 注意：此功能要求特定模型版本，且會增加 latency 和 cost
        if self.enable_thinking:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": config.thinking_budget,
            }

        return self.client.messages.create(**kwargs)

    def _handle_tool_calls(self, content_blocks: list) -> None:
        """處理 Claude 回應中的所有 tool_use block。"""
        for block in content_blocks:
            if block.type != "tool_use":
                continue

            tool_name = block.name
            tool_input = block.input
            tool_use_id = block.id

            tool = self.tool_registry.get(tool_name)
            if tool is None:
                result = f"錯誤：找不到工具 '{tool_name}'"
                self.memory.add_tool_result(tool_use_id, result, is_error=True)
                continue

            # 高風險工具需要確認
            if tool.requires_confirmation:
                if not self.confirm_callback(tool_name, tool_input):
                    result = "❌ 使用者拒絕執行此操作"
                    self.memory.add_tool_result(tool_use_id, result, is_error=True)
                    continue

            # 執行工具
            if self.verbose:
                console.print(
                    f"🔧 [cyan]執行工具:[/cyan] [bold]{tool_name}[/bold] "
                    f"[dim]{tool_input}[/dim]"
                )

            try:
                result = tool.execute(**tool_input)
                is_error = False
            except Exception as e:
                result = f"工具執行錯誤：{e}"
                is_error = True

            if self.verbose:
                preview = result[:200] + ("..." if len(result) > 200 else "")
                console.print(f"[dim]📤 結果：{preview}[/dim]")

            self.memory.add_tool_result(tool_use_id, result, is_error=is_error)

    def _display_response(self, response: Any) -> None:
        """把 Claude 的回應 (含 thinking 和 text) 印出來。"""
        if not self.verbose:
            return

        for block in response.content:
            if block.type == "thinking":
                console.print(
                    Panel(
                        block.thinking,
                        title="🧠 Extended Thinking",
                        border_style="magenta",
                    )
                )
            elif block.type == "text":
                console.print(
                    Panel(block.text, title="🤖 Claude", border_style="green")
                )

    @staticmethod
    def _extract_text(content_blocks: list) -> str:
        """從 content blocks 中抽出最後的文字回應。"""
        texts = [b.text for b in content_blocks if b.type == "text"]
        return "\n".join(texts)

    def reset(self) -> None:
        """清空對話記憶，重新開始。"""
        self.memory.clear()
