"""
Example 02: 帶工具的 Agent
===========================
示範如何給 Agent 裝備工具，並觀察 Agentic Loop 的運作。

觀察重點：
1. Claude 如何自己決定要呼叫哪個工具
2. 每一輪迭代 Claude 會根據結果調整策略
3. requires_confirmation=True 的工具會被攔下來問你
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.tools import (
    ReadFileTool,
    ListDirectoryTool,
    CalculatorTool,
    BashTool,
)


def main():
    agent = Agent(
        system_prompt=(
            "你是一位好奇心強的研究助理。"
            "擅長用工具探索環境、收集資訊後給出答案。"
            "每次行動前先說明你的計畫。"
        ),
        tools=[
            ReadFileTool(),
            ListDirectoryTool(),
            CalculatorTool(),
            BashTool(),  # 需要確認才會執行
        ],
        enable_thinking=True,
        verbose=True,
    )

    query = (
        "請幫我探索一下目前專案的結構（從當前目錄開始），"
        "然後告訴我 src 底下有幾個 Python 檔，"
        "並算出 7 的 10 次方是多少"
    )

    result = agent.run(query)

    print("\n" + "=" * 60)
    print("Final Answer:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
