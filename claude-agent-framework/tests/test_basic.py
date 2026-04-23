"""
基礎單元測試
============
這些測試不需要 API Key，只測試工具、memory、skill loader 的邏輯。

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

    # OpenAI function-calling 格式
    for t in api_format:
        assert t["type"] == "function"
        assert "name" in t["function"]
        assert "description" in t["function"]
        assert "parameters" in t["function"]

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
    mem.set_system_message("you are a helpful assistant")
    mem.add_user_message("hello")
    mem.add_assistant_message("hi")
    mem.add_tool_result("call_123", "result_data")

    messages = mem.get_messages()
    # system + user + assistant + tool = 4
    assert len(messages) == 4
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "assistant"
    assert messages[2]["content"] == "hi"
    assert messages[3]["role"] == "tool"
    assert messages[3]["tool_call_id"] == "call_123"
    assert messages[3]["content"] == "result_data"

    # clear(keep_system=True)
    mem.clear(keep_system=True)
    remaining = mem.get_messages()
    assert len(remaining) == 1
    assert remaining[0]["role"] == "system"

    # Full clear
    mem.clear()
    assert len(mem) == 0

    print("✅ test_memory")


def test_memory_assistant_with_tool_calls():
    """模擬 OpenAI SDK 的 message-like 物件被存入 memory。"""
    class _Fn:
        def __init__(self, name, arguments):
            self.name = name
            self.arguments = arguments

    class _TC:
        def __init__(self, id_, name, arguments):
            self.id = id_
            self.function = _Fn(name, arguments)

    class _Msg:
        def __init__(self, content, tool_calls):
            self.content = content
            self.tool_calls = tool_calls

    fake = _Msg(None, [_TC("call_abc", "calculator", '{"expression":"1+1"}')])

    mem = Memory()
    mem.add_assistant_message(fake)

    msgs = mem.get_messages()
    assert len(msgs) == 1
    m = msgs[0]
    assert m["role"] == "assistant"
    assert m["content"] is None
    assert m["tool_calls"][0]["id"] == "call_abc"
    assert m["tool_calls"][0]["type"] == "function"
    assert m["tool_calls"][0]["function"]["name"] == "calculator"
    assert m["tool_calls"][0]["function"]["arguments"] == '{"expression":"1+1"}'

    print("✅ test_memory_assistant_with_tool_calls")


if __name__ == "__main__":
    test_calculator_tool()
    test_file_tools()
    test_tool_registry()
    test_skill_loader()
    test_memory()
    test_memory_assistant_with_tool_calls()
    print("\n🎉 所有測試通過")
