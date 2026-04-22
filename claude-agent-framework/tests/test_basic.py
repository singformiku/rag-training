"""
基礎單元測試
============
這些測試不需要 API Key，只測試工具和 skill loader 的邏輯。

執行方式:
    python -m pytest tests/
或
    python tests/test_basic.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tools import (
    CalculatorTool,
    ReadFileTool,
    WriteFileTool,
    ListDirectoryTool,
    ToolRegistry,
)
from src.skills.loader import SkillLoader
from src.memory import Memory


def test_calculator_tool():
    tool = CalculatorTool()
    assert tool.name == "calculator"
    assert "2 + 3" in tool.execute(expression="2 + 3")
    assert "不允許" in tool.execute(expression="__import__('os')")
    print("✅ test_calculator_tool")


def test_file_tools():
    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)

        write = WriteFileTool()
        read = ReadFileTool()
        ls = ListDirectoryTool()

        # Write
        target = tmp_path / "hello.txt"
        result = write.execute(path=str(target), content="Hi there")
        assert "成功" in result

        # Read
        content = read.execute(path=str(target))
        assert content == "Hi there"

        # List
        listing = ls.execute(path=str(tmp_path))
        assert "hello.txt" in listing

    print("✅ test_file_tools")


def test_tool_registry():
    reg = ToolRegistry()
    reg.register(CalculatorTool())
    reg.register(ReadFileTool())

    assert len(reg) == 2
    assert reg.get("calculator") is not None
    assert reg.get("not_exist") is None

    api_format = reg.to_api_format()
    assert len(api_format) == 2
    assert all("name" in t and "description" in t for t in api_format)

    print("✅ test_tool_registry")


def test_skill_loader():
    with tempfile.TemporaryDirectory() as tmp:
        skill_dir = Path(tmp) / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "---\n"
            "name: my_skill\n"
            "description: 測試用 skill\n"
            "---\n\n"
            "# Body\n\nThis is the content.",
            encoding="utf-8",
        )

        loader = SkillLoader(Path(tmp))
        skills = loader.load_all()

        assert "my_skill" in skills
        assert skills["my_skill"].description == "測試用 skill"
        assert "Body" in skills["my_skill"].content

        summary = loader.get_skills_summary()
        assert "my_skill" in summary

    print("✅ test_skill_loader")


def test_memory():
    mem = Memory()
    mem.add_user_message("hello")
    mem.add_assistant_message("hi")
    mem.add_tool_result("tool_123", "result_data")

    messages = mem.get_messages()
    assert len(messages) == 3
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"
    assert messages[2]["content"][0]["type"] == "tool_result"

    mem.clear()
    assert len(mem) == 0

    print("✅ test_memory")


if __name__ == "__main__":
    test_calculator_tool()
    test_file_tools()
    test_tool_registry()
    test_skill_loader()
    test_memory()
    print("\n🎉 所有測試通過")
