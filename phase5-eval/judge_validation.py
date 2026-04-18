"""驗證 LLM judge 用 Cohen's kappa + TPR + TNR

Kappa 門檻（Landis & Koch + LLM judge 論文）：
  <0.40  poor（重寫 prompt）
  0.40–0.60  moderate（加 few-shot）
  0.60–0.80  substantial（可 ship CI）
  ≥0.80  human-level（Judge's Verdict 論文：Tier-1 LLM judge 應該 0.78–0.82）
"""
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, classification_report
from llm_judge import judge


def evaluate_judge(golden_csv):
    df = pd.read_csv(golden_csv)
    human = df["label"].map(lambda s: 1 if s.upper() == "PASS" else 0).tolist()
    judge_labels = []

    for _, row in df.iterrows():
        out = judge(
            row["question"],
            row["answer"],
            eval(row["retrieved_contexts"]),
        )
        judge_labels.append(1 if out["label"].upper() == "PASS" else 0)

    kappa = cohen_kappa_score(human, judge_labels)
    cm = confusion_matrix(human, judge_labels, labels=[1, 0])
    tp, fn, fp, tn = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) else 0.0
    tnr = tn / (tn + fp) if (tn + fp) else 0.0

    print(f"Kappa: {kappa:.3f}, TPR: {tpr:.3f}, TNR: {tnr:.3f}")
    print(classification_report(human, judge_labels))

    # 關鍵步驟：看 disagreements
    df["judge_label"] = ["PASS" if x == 1 else "FAIL" for x in judge_labels]
    df[df["label"] != df["judge_label"]].to_csv(
        "judge_disagreements.csv", index=False
    )

    return {"kappa": kappa, "tpr": tpr, "tnr": tnr}
