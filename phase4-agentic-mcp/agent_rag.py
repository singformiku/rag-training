"""agent_rag.py — Agentic RAG：讓 Claude 自己決定檢索策略

維度對比：
| 維度 | Traditional | Agentic |
|------|-------------|---------|
| 檢索時機 | 每次都查 | LLM 決定 |
| 檢索次數 | 固定 1 次 | 0~N 次 |
| Query 改寫 | 無 | LLM 可改寫、分解 |
| 自我修正 | 無 | 看到爛結果會重試 |
| Multi-hop | 差 | 原生支援 |
| 成本 | 低 | 較高 |

何時用：複雜問題（比較、推理、分解）、多資料來源 route、retrieval 不穩需要 fallback
何時別用：單純 FAQ、延遲敏感——用傳統 RAG 較省
"""
import json
import chromadb
from chromadb.utils import embedding_functions
from anthropic import Anthropic
from rich import print as rprint

client_chroma = chromadb.Client()
embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
collection = client_chroma.get_or_create_collection("demo_kb", embedding_function=embed_fn)

DOCS = [
    ("doc1", "Claude Sonnet 4.5 於 2025-09 發布，SWE-bench 與 OSWorld SOTA。"),
    ("doc2", "MCP 是 Anthropic 2024-11 發布的開放標準，統一 LLM 應用存取外部工具。"),
    ("doc3", "Claude Skills 於 2025-10 發布,資料夾格式(SKILL.md + scripts/)。"),
    ("doc4", "Agentic RAG 讓 LLM 決定何時檢索、檢索幾次、如何改寫 query。"),
    ("doc5", "MCP 三大 primitives：Tools、Resources、Prompts。"),
]
if collection.count() == 0:
    collection.add(ids=[d[0] for d in DOCS], documents=[d[1] for d in DOCS])

TOOLS = [
    {
        "name": "search_knowledge_base",
        "description": "在內部知識庫搜尋。可以多次呼叫、用不同 query 變形來提高召回率。",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "可以與原始問題不同（改寫、拆解）",
                },
                "top_k": {"type": "integer", "default": 3},
            },
            "required": ["query"],
        },
    },
    {
        "name": "finalize_answer",
        "description": "已蒐集足夠資訊，準備給出最終答案時呼叫。務必引用 doc_id。",
        "input_schema": {
            "type": "object",
            "properties": {
                "answer": {"type": "string"},
                "sources": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["answer", "sources"],
        },
    },
]


def execute_tool(name, args):
    if name == "search_knowledge_base":
        res = collection.query(
            query_texts=[args["query"]],
            n_results=min(args.get("top_k", 3), 10),
        )
        hits = [
            {
                "doc_id": res["ids"][0][i],
                "text": res["documents"][0][i],
                "distance": float(res["distances"][0][i]),
            }
            for i in range(len(res["ids"][0]))
        ]
        return json.dumps(
            {"query": args["query"], "hits": hits}, ensure_ascii=False
        )
    if name == "finalize_answer":
        return json.dumps(args, ensure_ascii=False)


def run_agent(question, max_iters=8):
    client = Anthropic()
    messages = [{"role": "user", "content": question}]
    system = (
        "你是嚴謹研究助理。回答前先用 search_knowledge_base 查。"
        "結果不夠就改寫 query 再查。不確定的事實不編造。"
        "完成後呼叫 finalize_answer 並列引用。用繁中回答。"
    )

    for i in range(max_iters):
        resp = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=4096,
            system=system,
            tools=TOOLS,
            messages=messages,
        )
        messages.append({"role": "assistant", "content": resp.content})

        if resp.stop_reason != "tool_use":
            return {
                "answer": "".join(b.text for b in resp.content if b.type == "text"),
                "iterations": i + 1,
            }

        tool_results = []
        for b in resp.content:
            if b.type == "tool_use":
                result = execute_tool(b.name, b.input)
                rprint(f"🔧 {b.name}({b.input}) → {result[:200]}...")
                tool_results.append(
                    {"type": "tool_result", "tool_use_id": b.id, "content": result}
                )
                if b.name == "finalize_answer":
                    return {**json.loads(result), "iterations": i + 1}
        messages.append({"role": "user", "content": tool_results})

    return {"error": "超過最大迭代", "iterations": max_iters}


if __name__ == "__main__":
    print(run_agent("Claude Skills 和 MCP 有什麼差異？兩者可以一起用嗎？"))
