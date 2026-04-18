"""完整 20-sample Ragas pipeline"""
from datasets import Dataset
from ragas import evaluate
from ragas_setup import evaluator_llm, evaluator_embeddings
from ragas_metrics import (
    faithfulness,
    response_relevancy,
    context_precision,
    context_recall,
    entity_recall,
    semantic_similarity,
)


def run_ragas(golden_set, rag_answer_fn):
    """
    golden_set: list of {question, ground_truth}
    rag_answer_fn: callable(question) -> {answer, contexts}
    """
    rows = []
    for g in golden_set:
        r = rag_answer_fn(g["question"])
        rows.append(
            {
                "user_input": g["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": g["ground_truth"],
            }
        )

    result = evaluate(
        Dataset.from_list(rows),
        metrics=[
            faithfulness,
            response_relevancy,
            context_precision,
            context_recall,
            entity_recall,
            semantic_similarity,
        ],
    )
    df = result.to_pandas()
    df.to_parquet("ragas_run.parquet")
    print(df.describe())
    return df


# Hamel 警告 🚨：Ragas 分數是 starter metric。
# 看到 faithfulness=0.82 不要高興——點開每個 <0.6 的 sample 看為什麼扣分，
# 這就是 look at your data。
