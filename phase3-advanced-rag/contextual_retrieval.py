"""Contextual Retrieval（Anthropic 旗艦技術）

核心：為每個 chunk 用 LLM 生成 50–100 tokens 的情境前綴。

原 chunk：公司的營收比上一季成長了 3%。
    ↓
Contextualized：這段來自 ACME Corp 2023 Q2 的 10-Q；上季營收 $314M。
               公司的營收比上一季成長了 3%。

實驗結果：
  retrieval failure 5.7%
    → 3.7%（Contextual Embeddings 單獨 -35%）
    → 2.9%（加 Contextual BM25 -49%）
    → 1.9%（加 reranker -67%）

成本靠 Prompt Caching 壓到極低：$1.02 / M document tokens（一次性）

完整 pipeline：Contextual Embeddings + Contextual BM25 + RRF + Reranker
retrieve 結果拿去 LLM 時用「原始 chunks」（避免 context 前綴污染生成）

何時用：知識庫 >200K、有大量共享 context 的文件（財報、法律、長技術手冊）
短獨立文件（FAQ、推文）邊際效益低
"""
from anthropic import Anthropic

client = Anthropic()


def generate_chunk_context(whole_document, chunk_content):
    response = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=150,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"<document>\n{whole_document}\n</document>",
                        "cache_control": {"type": "ephemeral"},  # ★ 長文件快取
                    },
                    {
                        "type": "text",
                        "text": f"""Here is the chunk: <chunk>{chunk_content}</chunk>

Please give a short succinct context (50-100 tokens) to situate this chunk
within the overall document, for improving search retrieval.
Answer ONLY with the succinct context.""",
                    },
                ],
            }
        ],
    )
    return response.content[0].text.strip()


def build_contextual_chunks(document, chunks):
    """為每個 chunk 加上情境前綴（retrieval 用）。
    注意：傳給 generator LLM 時要用原始 chunks，避免前綴污染。"""
    return [f"{generate_chunk_context(document, c)}\n\n{c}" for c in chunks]
