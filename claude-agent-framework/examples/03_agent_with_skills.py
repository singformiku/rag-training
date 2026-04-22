"""
Example 03: 載入 Skills 的 Agent
=================================
示範 Skill 系統：Agent 會在 system prompt 中看到所有 skill 的摘要，
當遇到匹配的任務時主動用 read_file 載入完整 SKILL.md 的指引。

這是 Claude 設計最精妙的地方之一 —— 領域知識不是寫死在程式裡，
而是可以用 Markdown 編輯、動態載入。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agent import Agent
from src.tools import ReadFileTool, ListDirectoryTool, BashTool

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    agent = Agent(
        system_prompt=(
            "你是一位資深工程師助理。"
            "當你看到使用者的任務符合某個 skill 的描述時，"
            "請先用 read_file 讀取對應的 SKILL.md 以取得詳細指引，再開始執行。"
        ),
        tools=[
            ReadFileTool(),
            ListDirectoryTool(),
            BashTool(),
        ],
        skills_dir=str(PROJECT_ROOT / "skills"),
        enable_thinking=True,
        verbose=True,
    )

    # 這個任務會觸發 code_review skill
    query = f"幫我審查 {PROJECT_ROOT}/src/tools/bash_tool.py 的程式碼品質"

    result = agent.run(query)

    print("\n" + "=" * 60)
    print("Final Answer:")
    print("=" * 60)
    print(result)


if __name__ == "__main__":
    main()
