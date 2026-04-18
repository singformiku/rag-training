"""Langfuse 整合（推薦，open source + self-host）

為什麼選 Langfuse 不選 LangSmith：
- MIT license
- 可 self-host（企業 resec 友好）
- OTel-based 低 lock-in
- RAG cookbook 完整

啟動：
  git clone https://github.com/langfuse/langfuse.git && cd langfuse
  docker compose up -d
  # 2–3 分鐘後 http://localhost:3000 註冊 → 建 project → 拿 keys
"""
import chromadb
from anthropic import Anthropic
from langfuse import observe, get_client

_anthropic = Anthropic()
_client_chroma = chromadb.Client()
_col = _client_chroma.get_or_create_collection("demo")
langfuse = get_client()


@observe(name="retrieve", as_type="retriever")
def retrieve(query, k=3):
    docs = _col.query(query_texts=[query], n_results=k)["documents"][0]
    langfuse.update_current_observation(
        metadata={"top_k": k, "n_results": len(docs)}
    )
    return docs


@observe(name="generate", as_type="generation")
def generate(query, contexts):
    resp = _anthropic.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"Context:\n{chr(10).join(contexts)}\nQ: {query}",
            }
        ],
    )
    langfuse.update_current_observation(
        model="claude-sonnet-4-5",
        usage_details={
            "input": resp.usage.input_tokens,
            "output": resp.usage.output_tokens,
        },
    )
    return resp.content[0].text


@observe(name="rag_qa")
def rag_qa(query, session_id, user_id):
    langfuse.update_current_trace(
        session_id=session_id,
        user_id=user_id,
        tags=["production"],
    )
    return generate(query, retrieve(query))


# 把 Ragas 分數寫回 Langfuse
# langfuse.create_score(name="faithfulness", value=0.85, trace_id=trace_id)
# langfuse.flush()  # 程式結束前一定要 flush
