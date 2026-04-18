"""Qdrant Vector Store（Production 首選）

啟動：docker run -d -p 6333:6333 qdrant/qdrant:latest

同樣介面也有 ChromaDB、pgvector、sqlite-vec 實作，
RAG pipeline 只要換一行 store = QdrantStore(...) 就從 dev 切到 prod。
"""
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct


class QdrantStore:
    def __init__(self, url="http://localhost:6333", collection="docs", dim=1024):
        self.client = QdrantClient(url=url)
        self.col = collection
        if not self.client.collection_exists(collection):
            self.client.create_collection(
                collection_name=collection,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )

    def add(self, ids, embeddings, docs, metas):
        points = [
            PointStruct(id=i, vector=e, payload={"doc": d, **(m or {})})
            for i, (e, d, m) in enumerate(zip(embeddings, docs, metas))
        ]
        self.client.upsert(collection_name=self.col, points=points, wait=True)

    def search(self, q_emb, k=5):
        res = self.client.query_points(
            collection_name=self.col,
            query=q_emb,
            limit=k,
            with_payload=True,
        ).points
        return [(str(p.id), 1 - p.score, p.payload) for p in res]
