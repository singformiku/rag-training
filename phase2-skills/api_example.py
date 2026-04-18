"""Claude API 使用 Skills 的範例

一個 request 最多 8 個 skill。
變更 skill 清單會打破 prompt cache，建議固定。
"""
from anthropic import Anthropic

client = Anthropic()

response = client.beta.messages.create(
    model="claude-sonnet-4-5",
    max_tokens=4096,
    container={
        "skills": [
            {"type": "anthropic", "skill_id": "pptx", "version": "latest"}
        ]
    },
    tools=[
        {"type": "code_execution_20250825", "name": "code_execution"}
    ],
    messages=[
        {"role": "user", "content": "Create a 5-slide deck on renewable energy."}
    ],
    betas=[
        "code-execution-2025-08-25",
        "files-api-2025-04-14",
        "skills-2025-10-02",
    ],
)

print(response)
