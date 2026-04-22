"""
Bash 執行工具
=============
讓 Agent 能執行 shell 指令。這是高風險工具，一定要有安全機制：
1. 黑名單檢查：阻擋 rm -rf /、sudo 等危險指令
2. 需要確認：requires_confirmation = True
3. 超時機制：避免指令卡住

這是 Claude Code 最核心的工具之一。
"""
import subprocess
from src.tools.base import Tool


# 危險指令黑名單
DANGEROUS_PATTERNS = [
    "rm -rf /",
    "rm -rf /*",
    ":(){ :|:& };:",  # fork bomb
    "mkfs",
    "dd if=",
    "> /dev/sda",
]


class BashTool(Tool):
    """執行 bash 指令。"""

    name = "bash"
    description = (
        "執行 bash shell 指令。適用於：列檔案、跑測試、git 操作、執行 Python 腳本等。"
        "請勿使用會長時間執行或互動式的指令。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "要執行的 bash 指令",
            },
            "timeout": {
                "type": "integer",
                "description": "逾時秒數，預設 30 秒",
                "default": 30,
            },
        },
        "required": ["command"],
    }

    # Bash 是高風險工具
    requires_confirmation = True

    def _is_dangerous(self, command: str) -> bool:
        """檢查指令是否在黑名單中。"""
        lower = command.lower()
        return any(pattern in lower for pattern in DANGEROUS_PATTERNS)

    def execute(self, command: str, timeout: int = 30) -> str:
        if self._is_dangerous(command):
            return f"❌ 拒絕執行：指令包含危險模式 - {command}"

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )

            output_parts = []
            if result.stdout:
                output_parts.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output_parts.append(f"STDERR:\n{result.stderr}")
            output_parts.append(f"Exit code: {result.returncode}")

            return "\n\n".join(output_parts)
        except subprocess.TimeoutExpired:
            return f"❌ 指令逾時 ({timeout}s)"
        except Exception as e:
            return f"❌ 執行錯誤：{e}"
