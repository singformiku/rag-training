"""統一 Embedder 抽象層（一行切換 provider）

三層推薦：
- 入門：voyage-3.5
- Production：voyage-3-large
- 離線：BAAI/bge-m3

注意事項：
(1) input_type document/query 要分清（Voyage/BGE/Cohere）
(2) 切換 embedding 必須重建整個索引（維度不相容）
(3) MTEB 分數不要迷信，自己小樣本 eval 最準
"""
from abc import ABC, abstractmethod


class Embedder(ABC):
    dim: int

    @abstractmethod
    def embed(self, texts, kind="document"):
        ...


class VoyageEmbedder(Embedder):
    def __init__(self, model="voyage-3.5"):
        import voyageai
        self.client = voyageai.Client()
        self.model = model
        self.dim = 1024

    def embed(self, texts, kind="document"):
        return self.client.embed(texts=texts, model=self.model, input_type=kind).embeddings


class LocalBGEEmbedder(Embedder):
    def __init__(self, model="BAAI/bge-m3"):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model)
        self.dim = 1024

    def embed(self, texts, kind="document"):
        if kind == "query":
            texts = [f"Represent this sentence for searching relevant passages: {t}" for t in texts]
        return self.model.encode(texts, normalize_embeddings=True).tolist()
