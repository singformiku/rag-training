# Phase 5 · RAG Evaluation & Eval-driven Development（2 週）⭐

## 檔案清單
- `ragas_setup.py` — Ragas + Claude setup（judge 不同家避免 self-preference bias）
- `ragas_metrics.py` — 6 個 metric + AspectCritic
- `ragas_pipeline.py` — 完整 pipeline
- `llm_judge.py` — Binary pass/fail judge（Hamel 範式）
- `judge_validation.py` — Cohen's kappa + TPR + TNR
- `langfuse_integration.py` — `@observe()` 自動 trace
- `synthetic_data.py` — Cold start synthetic 生成
- `ab_experiment.py` — 12-config A/B reproducible harness
- `.github/workflows/llm-eval.yml` — CI 紅綠燈

## Hamel 方法論七大原則
1. **三層次 eval 架構**：L1 Unit Tests（每 PR）→ L2 LLM-as-judge + human review → L3 A/B testing
2. **Binary pass/fail > Likert scale**：追蹤 1–5 分數是 bad eval 的訊號
3. **Look at your data**（open → axial coding）：看 ≥100 條 trace 寫自由文字 note
4. **Critique Shadowing 七步驟**：找單一 principal expert、binary label + critique、iter judge prompt
5. **驗證 judge 用 TPR/TNR + Cohen's kappa**（不只 alignment）
6. **不要用 generic metric**（BERTScore、ROUGE、cosine 都不 useful）
7. **CI 跑便宜 deterministic；production 裡 sample async 跑 LLM judge**

## Kappa 門檻
- <0.40 poor（重寫 prompt）
- 0.40–0.60 moderate（加 few-shot）
- 0.60–0.80 **substantial（可 ship CI）**
- ≥0.80 human-level

## Phase 5 驗收標準
- [ ] 讀完 Hamel 3 篇（Evals、LLM-as-Judge、Field Guide）+ Eugene 2 篇
- [ ] Ragas 6 metric 每個都有 single-sample demo
- [ ] 建 100 筆 golden set（含 adversarial 與 refusal test）
- [ ] Binary judge 與 human 的 Cohen's kappa ≥ 0.75
- [ ] CI 能在 PR 顯示 eval 報告；能讓故意的 bug PR 紅燈
- [ ] 跑出 12-config A/B leaderboard 並畫 cost-quality Pareto
