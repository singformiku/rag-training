"""
檔案操作工具
============
提供讀、寫、列目錄等基本檔案操作。

設計重點：
- 所有寫入操作都會回傳明確結果，方便 Claude 判斷是否成功
- write_file 被標示為 requires_confirmation，因為會改動檔案系統
"""
from pathlib import Path
from src.tools.base import Tool


class ReadFileTool(Tool):
    """讀取檔案內容。"""

    name = "read_file"
    description = (
        "讀取指定路徑的檔案內容。支援文字檔。"
        "若檔案過大，只會讀取前 10000 個字元。"
    )
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "檔案的絕對路徑或相對路徑",
            },
        },
        "required": ["path"],
    }

    def execute(self, path: str) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"錯誤：檔案不存在 - {path}"
            if not p.is_file():
                return f"錯誤：路徑不是檔案 - {path}"

            content = p.read_text(encoding="utf-8")
            if len(content) > 10000:
                content = content[:10000] + "\n\n... (已截斷，檔案過大)"
            return content
        except Exception as e:
            return f"讀取錯誤：{e}"


class WriteFileTool(Tool):
    """寫入檔案內容。"""

    name = "write_file"
    description = "寫入內容到指定檔案。若檔案不存在會建立，若存在會覆蓋。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "要寫入的檔案路徑",
            },
            "content": {
                "type": "string",
                "description": "要寫入的文字內容",
            },
        },
        "required": ["path", "content"],
    }

    # 寫檔會改變檔案系統，標示為需要確認
    requires_confirmation = True

    def execute(self, path: str, content: str) -> str:
        try:
            p = Path(path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content, encoding="utf-8")
            return f"成功寫入 {len(content)} 字元到 {path}"
        except Exception as e:
            return f"寫入錯誤：{e}"


class ListDirectoryTool(Tool):
    """列出目錄內容。"""

    name = "list_directory"
    description = "列出指定目錄下的檔案和子目錄。"
    input_schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "目錄路徑，預設為當前目錄",
            },
        },
        "required": ["path"],
    }

    def execute(self, path: str) -> str:
        try:
            p = Path(path)
            if not p.exists():
                return f"錯誤：目錄不存在 - {path}"
            if not p.is_dir():
                return f"錯誤：路徑不是目錄 - {path}"

            items = sorted(p.iterdir())
            lines = []
            for item in items:
                marker = "📁" if item.is_dir() else "📄"
                lines.append(f"{marker} {item.name}")
            return "\n".join(lines) if lines else "(空目錄)"
        except Exception as e:
            return f"列出目錄錯誤：{e}"
