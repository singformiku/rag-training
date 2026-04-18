"""Ragas 6 個 metric + AspectCritic（自訂 binary judge）"""
from ragas.metrics import (
    Faithfulness,
    ResponseRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ContextEntityRecall,
    SemanticSimilarity,
    AspectCritic,
)
from ragas_setup import evaluator_llm, evaluator_embeddings


# 1. Faithfulness：答案是否 grounded in context
#    公式：faithful_claims / total_claims
faithfulness = Faithfulness(llm=evaluator_llm)

# 2. Answer Relevancy：從 response 反向生成 question，跟原 question 算 cosine sim
response_relevancy = ResponseRelevancy(
    llm=evaluator_llm, embeddings=evaluator_embeddings
)

# 3. Context Precision：相關 chunk 是否排在前面（對排序敏感）
context_precision = LLMContextPrecisionWithReference(llm=evaluator_llm)

# 4. Context Recall：reference 的每個 claim 是否都在 retrieved_contexts 找到支持
context_recall = LLMContextRecall(llm=evaluator_llm)

# 5. Context Entity Recall：named entities 有多少在 retrieved_contexts
#    entity 密集 domain（地名、藥名、法條）特別有用
entity_recall = ContextEntityRecall(llm=evaluator_llm)

# 6. Semantic Similarity：純 embedding，便宜快速
semantic_similarity = SemanticSimilarity(embeddings=evaluator_embeddings)


# BONUS: AspectCritic — 自訂 binary judge（Hamel 式的起點）
over_promise = AspectCritic(
    name="over_promise",
    definition="Return 1 if response promises SLA the system can't guarantee.",
    llm=evaluator_llm,
)
