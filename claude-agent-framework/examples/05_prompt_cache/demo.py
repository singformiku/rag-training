import anthropic
import time
import os
from dotenv import load_dotenv

# Load the .env file into the environment
load_dotenv()

# 🔑 Set your API key first:
ANTHROPIC_API_KEY=os.getenv("ANTHROPIC_API_KEY")

client = anthropic.Anthropic()

# 📦 Load large system prompt
with open("system.md") as f:
    SYSTEM_PROMPT = f.read()

# 🧰 Example tool schema (cached)
TOOLS = [
    {
        "name": "run_kubectl",
        "description": "Run kubectl command",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string"}
            },
            "required": ["command"]
        }
    }
]

# ✅ Enable caching on tools
TOOLS[-1]["cache_control"] = {
    "type": "ephemeral",
    "ttl": "1h"
}


def run_turn(history, user_msg):
    """
    Single turn execution with caching applied
    """

    # 🧠 System prompt caching (long-lived)
    system_blocks = [
        {
            "type": "text",
            "text": "You are a senior SRE assistant."
        },
        {
            "type": "text",
            "text": SYSTEM_PROMPT,
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        },
    ]

    messages = history + [
        {"role": "user", "content": user_msg}
    ]

    # 🔁 Cache previous turn (short-lived ~5 min)
    if len(messages) >= 2:
        messages[-2] = {
            **messages[-2],
            "content": [{
                "type": "text",
                "text": messages[-2]["content"],
                "cache_control": {"type": "ephemeral"}
            }]
        }

    t0 = time.perf_counter()

    resp = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=300,
        tools=TOOLS,
        system=system_blocks,
        messages=messages,
    )

    dt = time.perf_counter() - t0
    u = resp.usage

    print("\n📊 Usage:")
    print(f"  input_tokens: {u.input_tokens}")
    print(f"  cache_read:  {u.cache_read_input_tokens}")
    print(f"  cache_write: {u.cache_creation_input_tokens}")
    print(f"  latency:     {dt:.2f}s")

    return resp


def main():
    history = []

    print("\n🚀 First call (NO cache yet)")
    resp1 = run_turn(history, "How to debug Kubernetes pod crash?")
    history.append({
        "role": "assistant",
        "content": resp1.content[0].text
    })

    print("\n⚡ Second call (cache should HIT)")
    resp2 = run_turn(history, "What logs should I check?")
    history.append({
        "role": "assistant",
        "content": resp2.content[0].text
    })

    print("\n⚡ Third call (strong cache reuse)")
    run_turn(history, "How to check node health?")


if __name__ == "__main__":
    main()