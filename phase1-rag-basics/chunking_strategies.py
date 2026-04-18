"""Chunking 三策略對比
pip install langchain-text-splitters voyageai numpy
"""
import re
import numpy as np
import voyageai
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


def fixed_size_chunks(text, size=500, overlap=50):
    step = size - overlap
    return [text[i:i+size] for i in range(0, len(text), step) if text[i:i+size].strip()]


def recursive_chunks(text, size=500, overlap=80):
    return RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", "。", "！", "？", ". ", " ", ""],
    ).split_text(text)


def recursive_markdown(text, size=800):
    """Markdown 結構感知：把 # ## ### 路徑注入 chunk"""
    md = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")]
    )
    docs = md.split_text(text)
    rec = RecursiveCharacterTextSplitter(chunk_size=size, chunk_overlap=80)
    out = []
    for d in docs:
        header = " > ".join(d.metadata.values())
        for sub in rec.split_text(d.page_content):
            out.append(f"[{header}]\n{sub}")
    return out


def semantic_chunks(text, percentile=85):
    """按 embedding 相似度切：主題切換點才切"""
    sents = [s.strip() for s in re.split(r'(?<=[。！？.!?])\s+', text) if s.strip()]
    if len(sents) < 3:
        return sents
    vo = voyageai.Client()
    embs = np.array(vo.embed(sents, model="voyage-3.5", input_type="document").embeddings)
    cos = lambda a, b: 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    dists = [cos(embs[i], embs[i+1]) for i in range(len(embs)-1)]
    bp = np.percentile(dists, percentile)
    splits = [i+1 for i, d in enumerate(dists) if d > bp]
    chunks, start = [], 0
    for idx in splits:
        chunks.append(" ".join(sents[start:idx]))
        start = idx
    chunks.append(" ".join(sents[start:]))
    return [c for c in chunks if c]


# 決策：90% 場景用 recursive；技術文件/原始碼用 markdown-aware；
# 主題切換頻繁的長文才考慮 semantic；fixed 只用於均質 log。
# 參數經驗值：chunk_size=400–800 tokens、overlap=10–20%。
