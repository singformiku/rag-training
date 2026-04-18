"""Skill 內部 retrieval 腳本"""
import argparse
import json
from pathlib import Path
import chromadb
from chromadb.utils import embedding_functions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    db_path = Path(__file__).parent.parent / ".chroma"
    client = chromadb.PersistentClient(path=str(db_path))
    ef = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )
    col = client.get_collection("anthropic_docs", embedding_function=ef)

    res = col.query(query_texts=[args.query], n_results=args.top_k)
    hits = [
        {
            "url": res["metadatas"][0][i]["url"],
            "title": res["metadatas"][0][i].get("title", ""),
            "snippet": res["documents"][0][i][:500],
            "score": 1 - float(res["distances"][0][i]),
        }
        for i in range(len(res["ids"][0]))
    ]
    print(
        json.dumps(
            {"query": args.query, "hits": hits},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
