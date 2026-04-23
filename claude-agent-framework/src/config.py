"""
設定模組
=========
集中管理 LLM 接入、Agent 行為等設定。

兩個全域實例：
- ``settings``：對應 llm_service.py 內所需的 LLM / 基礎設施參數
  （模仿原 ``infrastructure.config.settings`` 的欄位命名）。
- ``config``：Agent 層級的行為參數 (迴圈次數、是否開啟 thinking 等)。

所有值皆從專案根目錄的 .env 讀取。
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

# 載入 repo 根目錄的 .env
_PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(_PROJECT_ROOT / ".env")


def _get_optional(name: str) -> Optional[str]:
    """讀取環境變數，若為空字串則視為 None。"""
    v = os.getenv(name)
    return v if v else None


@dataclass
class Settings:
    """
    LLM / 基礎設施設定。

    欄位命名刻意對齊原本 ``infrastructure.config.settings``
    ，讓 ``llm_service.py`` 可以直接沿用。
    """

    # ---- LLM ----
    LLM_BASE_URL: str = os.getenv("LLM_URL", "")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-oss-120b")
    LLM_CLIENT_ID: str = os.getenv("LLM_CLIENT_ID", "")
    LLM_SECRET: str = os.getenv("LLM_SECRET", "")
    LLM_MAX_RESPONSE_TOKENS: int = int(os.getenv("LLM_MAX_RESPONSE_TOKENS", "2048"))

    # ---- TLS ----
    # 若為空 → ``False`` (不驗證)，與原 infrastructure 的行為一致
    PEM_LOCATION: Optional[str] = field(default_factory=lambda: _get_optional("PEM_LOCATION"))

    def validate(self) -> None:
        missing = [
            n for n in ("LLM_BASE_URL", "LLM_CLIENT_ID", "LLM_SECRET", "LLM_MODEL")
            if not getattr(self, n)
        ]
        if missing:
            raise ValueError(
                f"缺少必要的 LLM 設定：{missing}。請在 .env 中補齊。"
            )


@dataclass
class Config:
    """Agent 行為設定。"""

    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    enable_thinking: bool = os.getenv("ENABLE_THINKING", "true").lower() == "true"
    # 對應 gpt-oss-120b 的 reasoning_effort：low / medium / high
    reasoning_effort: str = os.getenv("REASONING_EFFORT", "medium")

    project_root: Path = _PROJECT_ROOT
    skills_dir: Path = _PROJECT_ROOT / "skills"

    def validate(self) -> None:
        settings.validate()


# 全域實例
settings = Settings()
config = Config()
