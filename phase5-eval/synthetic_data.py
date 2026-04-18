"""Synthetic Data 生成（Hamel Field Guide Part 2 方法）

資料來源優先序：
  1. Production logs
  2. User thumbs-down feedback
  3. Adversarial edge cases
  4. Synthetic data（cold start 才用）

必含 adversarial：
- refusal test（context 沒答案是否正確說不知道）
- prompt injection
- multi-hop
- ambiguity
- out-of-domain
"""
import json
from anthropic import Anthropic

client = Anthropic()


PROMPT = """Simulate real Dell enterprise IT admins. Generate {n} diverse questions
covering: User role (sysadmin/DBA/help-desk), Difficulty (simple/multi-hop/edge/adversarial),
Topic (PowerEdge/iDRAC/ProSupport/PowerStore).

Format: JSON array with keys question/role/difficulty/topic.
"""


def generate_synthetic(n=20):
    msg = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=4096,
        messages=[{"role": "user", "content": PROMPT.format(n=n)}],
    )
    text = msg.content[0].text
    # 去掉可能的 markdown fence
    text = text.strip().strip("`").removeprefix("json").strip()
    return json.loads(text)


if __name__ == "__main__":
    questions = generate_synthetic(20)
    with open("synthetic_questions.json", "w", encoding="utf-8") as f:
        json.dump(questions, f, ensure_ascii=False, indent=2)
    print(f"Generated {len(questions)} synthetic questions")
