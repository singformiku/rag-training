"""
Example 04: 複雜多步任務
==========================
示範 Agent 處理一個需要多輪工具呼叫才能完成的任務。

觀察點：
- Agent 如何自己拆解任務（不靠外部 planner）
- 中間發生錯誤時如何重新規劃
- 連續 tool_use → tool_result → tool_use 的循環
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.tools import (
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    BashTool,
    CalculatorTool,
)


def auto_approve(tool_name: str, tool_input: dict) -> bool:
    """
    自動批准回呼：範例中為了方便 demo 自動放行所有高風險操作。
    正式環境絕對不要這樣用，請務必保留人工確認。
    """
    print(f"[auto-approve] {tool_name}({tool_input})")
    return True


def main():
    agent = Agent(
        system_prompt=(
            "你是一個獨立作業的工程師 Agent。"
            "收到任務時請：\n"
            "1. 先把任務拆成明確的步驟\n"
            "2. 逐步執行，每步都觀察結果\n"
            "3. 遇到錯誤時主動除錯而不是放棄\n"
            "4. 完成後給出清楚的報告"
        ),
        tools=[
            ReadFileTool(),
            WriteFileTool(),
            ListDirectoryTool(),
            BashTool(),
            CalculatorTool(),
        ],
        max_iterations=15,  # 多步任務，放寬上限
        enable_thinking=True,
        confirm_callback=auto_approve,  # demo 用，正式請拿掉
        verbose=True,
    )

    query = (
        "請幫我做一個小任務：\n"
        "1. 在 /tmp/agent_demo 資料夾中建立三個檔案：a.txt、b.txt、c.txt，"
        "內容分別是 'Hello'、'World'、'!'\n"
        "2. 用 bash 把三個檔案合併成一個 combined.txt\n"
        "3. 讀取 combined.txt 並確認內容是否正確\n"
        "4. 最後回報任務執行狀況"
    )

    result = agent.run(query)

    print("\n" + "=" * 60)
    print("Final Answer:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
