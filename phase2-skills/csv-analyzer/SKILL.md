---
name: csv-analyzer
description: Analyzes CSV files and produces data quality reports, column profiles, and summary statistics. Use whenever the user provides a .csv file and wants to "analyze", "summarize", "explore", "profile", or "do EDA on" the data. Also use when the user asks about data quality (nulls, duplicates), column types, or distributions. Do NOT use for .xlsx files (use xlsx skill) or for creating new data.
compatibility: Requires Python 3.10+, pandas, numpy
---

# CSV Analyzer

## Workflow（不要跳步驟）

### Step 1 — Profile
`python scripts/profile.py <csv_path>` → JSON: row_count, column_count, 每欄的
dtype/null_count/unique_count/sample_values。**不要自己重寫這段邏輯**，直接呼叫，省 token。

### Step 2 — Summary stats（數值欄位）
`python scripts/summary_stats.py <csv_path> --columns c1,c2`

### Step 3 — 寫報告（繁中）
包含：資料概覽、欄位類型分佈、⚠️品質警告（null > 5%、全 unique、常數欄）、
💡 Top 3 建議分析方向。

## Edge cases
- 檔案 >1GB → `chunksize=100000` 分塊讀
- 編碼錯誤 → 試 `utf-8` → `big5` → `cp950`（台灣 CSV 常見）
- 分隔非逗號 → `engine='python', sep=None` 自動偵測
