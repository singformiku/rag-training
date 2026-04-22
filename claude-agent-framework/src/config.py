"""
設定模組
=========
集中管理 API Key、模型名稱、參數等設定。
"""
import os
from dataclasses import dataclass
from pathlib import Path
from dotenv import load_dotenv

# 載入 .env 檔案
load_dotenv()


@dataclass
class Config:
    """Agent 的全域設定。"""

    # API 相關
    api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    model: str = os.getenv("CLAUDE_MODEL", "claude-sonnet-4-5")

    # Agent 行為
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "true").lower() == "true"
    thinking_budget: int = 2000  # tokens

    # 路徑
    project_root: Path = Path(__file__).parent.parent
    skills_dir: Path = Path(__file__).parent.parent / "skills"

    def validate(self) -> None:
        """確認必要設定是否齊全。"""
        if not self.api_key or self.api_key.startswith("sk-ant-xxxxx"):
            raise ValueError(
                "請在 .env 檔案中設定 ANTHROPIC_API_KEY。\n"
                "可至 https://console.anthropic.com 取得 API Key。"
            )


# 全域 config 實例
config = Config()
