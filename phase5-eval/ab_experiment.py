"""A/B 實驗 reproducible harness

掃過 3 × 2 × 2 = 12 個 config（chunk_size × embedding × reranker），
結果存 SQLite。
"""
import json
import sqlite3
import hashlib
from dataclasses import dataclass
from itertools import product
from typing import Optional

import pandas as pd


@dataclass(frozen=True)
class RagConfig:
    chunk_size: int
    embedding: str       # "voyage-3" | "openai-3-large"
    reranker: Optional[str]  # None | "cohere-rerank-3"


def config_id(cfg: RagConfig) -> str:
    return hashlib.md5(
        json.dumps(cfg.__dict__, sort_keys=True).encode()
    ).hexdigest()[:8]


def run_experiment(corpus, golden, build_index, retrieve, generate, compute_cost, ragas_score):
    """
    build_index(cfg, corpus) -> index
    retrieve(cfg, index, question) -> contexts
    generate(question, contexts) -> (answer, usage_dict)
    compute_cost(usage_dict) -> float
    ragas_score(rows) -> dict of metric name -> score
    """
    configs = [
        RagConfig(cs, emb, rr)
        for cs, emb, rr in product(
            [512, 1024, 2048],
            ["voyage-3", "openai-3-large"],
            [None, "cohere-rerank-3"],
        )
    ]

    con = sqlite3.connect("experiments.sqlite")
    con.execute(
        """CREATE TABLE IF NOT EXISTS runs (
            run_id TEXT, config_id TEXT, config_json TEXT,
            faithfulness REAL, answer_relevancy REAL,
            context_precision REAL, context_recall REAL,
            latency_ms_p50 REAL, latency_ms_p95 REAL,
            cost_usd_per_query REAL, ts TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )"""
    )

    for cfg in configs:
        cid = config_id(cfg)
        index = build_index(cfg, corpus)
        rows, latencies, costs = [], [], []

        for _, q in golden.iterrows():
            ctx = retrieve(cfg, index, q["question"])
            ans, usage = generate(q["question"], ctx)
            latencies.append(usage["latency_ms"])
            costs.append(compute_cost(usage))
            rows.append(
                {
                    "user_input": q["question"],
                    "response": ans,
                    "retrieved_contexts": ctx,
                    "reference": q["reference"],
                }
            )

        m = ragas_score(rows)
        import numpy as np
        con.execute(
            "INSERT INTO runs (run_id, config_id, config_json, faithfulness, answer_relevancy, "
            "context_precision, context_recall, latency_ms_p50, latency_ms_p95, cost_usd_per_query) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                cid,
                cid,
                json.dumps(cfg.__dict__),
                m.get("faithfulness"),
                m.get("answer_relevancy"),
                m.get("context_precision"),
                m.get("context_recall"),
                float(np.percentile(latencies, 50)),
                float(np.percentile(latencies, 95)),
                float(np.mean(costs)),
            ),
        )
        con.commit()

    df = pd.read_sql(
        "SELECT * FROM runs",
        sqlite3.connect("experiments.sqlite"),
    )
    print(df.sort_values("faithfulness", ascending=False).to_string())
    return df


# ========== 或用 pytest-parametrize ==========
# @pytest.mark.parametrize("cfg", CONFIGS, ids=["512-voyage-noRR", "1024-voyage-RR", ...])
# def test_rag_config_meets_bar(cfg, golden, corpus, record_property):
#     ...
#     assert m["faithfulness"] >= 0.80
#     assert m["context_recall"] >= 0.70
