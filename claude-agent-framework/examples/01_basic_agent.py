"""
Example 01: 最基本的 Agent
===========================
示範沒有工具、沒有 skill 的純對話 Agent。

重點：即使沒有工具，Agent 本身就能透過推理回答問題。
Extended Thinking 讓你看到它的思考過程。
"""
import sys
from pathlib import Path

# 把專案根目錄加進 sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent


def main():
    agent = Agent(
        system_prompt="你是一位友善、博學的助理。回答時請用繁體中文。",
        enable_thinking=True,  # 開啟延伸思考
        verbose=True,
    )

    query = "幫我比較 ReAct 和 Plan-and-Execute 這兩種 agent 架構的優缺點"
    result = agent.run(query)

    print("\n" + "=" * 60)
    print("Final Answer:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
