"""server.py — 用 FastMCP 建 stdio MCP server

安裝：
pip install "mcp[cli]>=1.7.1" "chromadb>=0.5" "sentence-transformers>=3"

⚠️ 重要：Log 只能走 stderr！stdout 給 JSON-RPC
"""
import sys
import json
import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions
from mcp.server.fastmcp import FastMCP

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stderr,
    format="[docs-rag-mcp] %(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

DB_DIR = Path(__file__).parent / ".chroma"
DB_DIR.mkdir(exist_ok=True)
_chroma = chromadb.PersistentClient(path=str(DB_DIR))
_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)
_col = _chroma.get_or_create_collection("docs", embedding_function=_ef)

mcp = FastMCP("docs-rag-mcp")


@mcp.tool()
def search_docs(query: str, top_k: int = 5) -> dict[str, Any]:
    """在本地知識庫做語意搜尋（支援中英文）。"""
    top_k = max(1, min(top_k, 20))
    res = _col.query(query_texts=[query], n_results=top_k)
    hits = [
        {
            "doc_id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "score": round(1 - float(res["distances"][0][i]), 4),
            "source": (res["metadatas"][0][i] or {}).get("source", ""),
        }
        for i in range(len(res["ids"][0]))
    ]
    return {"query": query, "total_indexed": _col.count(), "hits": hits}


@mcp.tool()
def list_sources() -> dict:
    all_ = _col.get()
    return {
        "sources": sorted(
            {(m or {}).get("source", "?") for m in all_["metadatas"]}
        )
    }


@mcp.resource("docs://stats")
def stats() -> str:
    return json.dumps({"total_chunks": _col.count()}, ensure_ascii=False)


@mcp.prompt()
def cite_answer(question: str) -> str:
    return f"請用 search_docs 查詢並在回答中引用 doc_id。\n問題：{question}"


if __name__ == "__main__":
    log.info("Starting MCP server (stdio)...")
    mcp.run()
