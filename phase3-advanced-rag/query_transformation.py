"""Query Transformation 四技巧

| 技術 | 適用 | 成本 | 別用 |
|------|------|------|------|
| HyDE | domain 陌生、短 query | +1 LLM | 已 fine-tune embedding |
| Decomposition | multi-hop、比較、複合 | +1 LLM + N retrieval | 單一事實 |
| Multi-query | 召回不足、用詞不一 | +1 LLM + N | 已 hybrid+rerank |
| Step-back | 需要背景原理 | +1 LLM | 簡單事實 |
"""
from typing import List
import anthropic
import instructor
from pydantic import BaseModel

anth_client = instructor.from_anthropic(anthropic.Anthropic())
client = anthropic.Anthropic()


# ===== HyDE（用假答案 embedding 找真答案）=====
# 適合 short query、zero-shot domain
# ⚠️ Anthropic Contextual Retrieval blog 實驗：效果 limited gains
# 已 fine-tune embedding 不一定贏，永遠 A/B
def hyde_generate(q, n=1):
    hyps = []
    for _ in range(n):
        msg = client.messages.create(
            model="claude-3-5-haiku-latest",
            max_tokens=400,
            temperature=0.7,
            messages=[
                {
                    "role": "user",
                    "content": f"你是專業回答者。給定問題，生成一段 3-5 句的「假設答案」，語氣接近真實文件。\n問題：{q}",
                }
            ],
        )
        hyps.append(msg.content[0].text)
    return hyps


# ===== Query Decomposition（複雜問題拆子問題）=====
class SubQuestion(BaseModel):
    question: str
    reason: str


class Decomposition(BaseModel):
    sub_questions: List[SubQuestion]


def decompose(q):
    return anth_client.messages.create(
        model="claude-3-5-sonnet-latest",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": f"把複雜問題拆成 2-5 個原子化、可獨立檢索的子問題。\n原問題：{q}",
            }
        ],
        response_model=Decomposition,
    )


# ===== Multi-query（改寫成多個查詢變體，搭配 RRF 融合）=====
class MultiQueries(BaseModel):
    queries: List[str]


def multi_query(q, n=3):
    return anth_client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=512,
        messages=[
            {
                "role": "user",
                "content": f"把下列問題改寫成 {n} 個不同用詞、但語意相同的搜尋 query。\n原問題：{q}",
            }
        ],
        response_model=MultiQueries,
    )


# ===== Step-back（退一步問更抽象的原理題）=====
def step_back(q):
    msg = client.messages.create(
        model="claude-3-5-haiku-latest",
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": f"把下列具體問題退一步，改寫成更抽象、涵蓋背景原理的問題。\n原問題：{q}",
            }
        ],
    )
    return msg.content[0].text
