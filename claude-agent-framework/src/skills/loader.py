"""
Skill 載入器
============
Skill 是 Claude 獨特的機制：把領域知識用 Markdown 檔 (SKILL.md) 描述，
Agent 在遇到相關任務時動態載入這些指引。

SKILL.md 格式：
---
name: skill_name
description: 什麼情況下該觸發這個 skill (用來幫助 Agent 判斷)
---

# Skill Title

具體的做事方式、最佳實踐、範例...

為什麼這樣設計？
- 把「如何做好某件事」的知識跟程式碼解耦
- 可以由非工程師 (如產品、設計師) 編輯
- Agent 只在需要時才載入，不浪費 context
"""
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Skill:
    """代表一個已載入的 Skill。"""

    name: str
    description: str
    content: str  # 完整的 SKILL.md 內文 (不含 frontmatter)
    path: Path

    def __repr__(self) -> str:
        return f"<Skill {self.name}: {self.description[:40]}...>"


class SkillLoader:
    """從檔案系統載入 Skills。"""

    def __init__(self, skills_dir: Path):
        self.skills_dir = Path(skills_dir)
        self._skills: dict[str, Skill] = {}

    def load_all(self) -> dict[str, Skill]:
        """掃描 skills_dir 下所有的 SKILL.md。"""
        if not self.skills_dir.exists():
            return {}

        for skill_md in self.skills_dir.rglob("SKILL.md"):
            try:
                skill = self._parse_skill_file(skill_md)
                self._skills[skill.name] = skill
            except Exception as e:
                print(f"⚠️ 無法載入 {skill_md}: {e}")

        return self._skills

    def _parse_skill_file(self, path: Path) -> Skill:
        """解析 SKILL.md，拆出 frontmatter 和內文。"""
        text = path.read_text(encoding="utf-8")

        # 嘗試解析 YAML frontmatter
        frontmatter_pattern = r"^---\s*\n(.*?)\n---\s*\n(.*)$"
        match = re.match(frontmatter_pattern, text, re.DOTALL)

        if not match:
            raise ValueError(f"{path} 缺少有效的 frontmatter")

        frontmatter_text, content = match.groups()

        # 簡單的 YAML 解析 (只處理 key: value)
        metadata = {}
        for line in frontmatter_text.strip().split("\n"):
            if ":" in line:
                key, value = line.split(":", 1)
                metadata[key.strip()] = value.strip().strip('"').strip("'")

        if "name" not in metadata or "description" not in metadata:
            raise ValueError(f"{path} 的 frontmatter 缺少 name 或 description")

        return Skill(
            name=metadata["name"],
            description=metadata["description"],
            content=content.strip(),
            path=path,
        )

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def list_skills(self) -> list[Skill]:
        return list(self._skills.values())

    def get_skills_summary(self) -> str:
        """
        產生所有 Skills 的摘要，用於塞進 system prompt。

        這是 Claude 做「skill discovery」的關鍵：把所有 skill 的 name 和
        description 列給 Claude 看，讓它自己判斷何時該參考哪個 skill。
        """
        if not self._skills:
            return ""

        lines = ["## 可用的 Skills", ""]
        lines.append(
            "以下是可參考的知識庫。當使用者的任務符合某個 skill 的描述時，"
            "請主動參考該 skill 的完整內容 (使用 read_file 工具讀取對應的 SKILL.md)。"
        )
        lines.append("")

        for skill in self._skills.values():
            lines.append(f"- **{skill.name}** ({skill.path})")
            lines.append(f"  {skill.description}")

        return "\n".join(lines)
