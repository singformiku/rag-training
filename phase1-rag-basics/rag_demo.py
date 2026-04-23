"""最小可跑 RAG demo（2026 版）
Stack: Claude Sonnet 4.5 + Voyage-3.5 + ChromaDB
執行：python rag_demo.py
"""
import httpx
import chromadb
import voyageai
from anthropic import Anthropic

ANTHROPIC_MODEL = "claude-sonnet-4-5"
EMBED_MODEL = "voyage-3.5"
COLLECTION = "demo_docs"
TOP_K = 4

VOYAGE_API_KEY="pa-qm1t7DmFF-fDLzPOYAEUxkr4eSldcreEKrpsPZBxiWC"
anthropic_client = Anthropic()
voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
chroma_client = chromadb.PersistentClient(path="./chroma_db")


def load_sample_doc() -> str:
    url = "https://raw.githubusercontent.com/anthropics/anthropic-cookbook/main/README.md"
    r = httpx.get(url, timeout=30)
    r.raise_for_status()
    return r.text


def chunk_text(text: str, size=500, overlap=80) -> list[str]:
    paragraphs, chunks, buf = text.split("\n\n"), [], ""
    for p in paragraphs:
        if len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}" if buf else p
        else:
            if buf:
                chunks.append(buf)
            while len(p) > size:
                chunks.append(p[:size])
                p = p[size - overlap:]
            buf = p
    if buf:
        chunks.append(buf)
    return [c.strip() for c in chunks if c.strip()]


def build_index(chunks):
    try:
        chroma_client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = chroma_client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})
    result = voyage_client.embed(texts=chunks, model=EMBED_MODEL, input_type="document")
    col.add(
        ids=[f"c{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=result.embeddings,
    )
    return col


def retrieve(col, query, k=TOP_K):
    q_emb = voyage_client.embed(texts=[query], model=EMBED_MODEL, input_type="query").embeddings[0]
    res = col.query(query_embeddings=[q_emb], n_results=k)
    return res["documents"][0]


def generate(query, contexts):
    context_block = "\n\n---\n\n".join(f"[Doc {i+1}]\n{c}" for i, c in enumerate(contexts))
    msg = anthropic_client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,
        system="你是一個嚴謹的技術助理。只能根據 <context> 內的內容回答。若不足請說『資料不足』。",
        messages=[{"role": "user", "content": f"<context>\n{context_block}\n</context>\n\n問題：{query}"}],
    )
    return msg.content[0].text


def main():
    doc = load_sample_doc()
    chunks = chunk_text(doc)
    print(f"共 {len(chunks)} chunks")
    col = build_index(chunks)
    for q in ["What is the Anthropic Cookbook?", "Does it include tool use examples?"]:
        print(f"\n❓ {q}")
        print(f"✅ {generate(q, retrieve(col, q))}")


if __name__ == "__main__":
    main()
