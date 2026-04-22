---
name: data_analysis
description: 當使用者想分析資料、計算統計、或探索資料集時觸發
---

# Data Analysis Skill

## 流程

### 1. 先理解資料
- 使用 `bash` 執行 `head -5 file.csv` 或 `wc -l file.csv` 快速了解資料樣貌
- 確認欄位、資料筆數、是否有 missing value

### 2. 釐清分析目標
若使用者的需求模糊，先問清楚：
- 想看的是 **分布、趨勢、比較、還是關聯**？
- 有沒有特別關心的維度 (時間、地區、類別)？

### 3. 選擇工具
- 小資料 (< 10000 列)：用 Python 搭配 pandas 處理
- 大資料：建議用 SQL + 分塊處理
- 簡單統計：可以直接用 bash + awk

### 4. 輸出
- 用表格呈現關鍵數字
- 必要時產生 matplotlib 圖表 (儲存為 png)
- 最後給一段「白話總結」，點出最重要的發現

## 常用分析模板

### 基礎統計
```python
import pandas as pd
df = pd.read_csv("data.csv")
print(df.describe())
print(df.isnull().sum())
```

### 分組彙總
```python
df.groupby("category")["value"].agg(["count", "mean", "std"])
```

## 原則
- 永遠先檢查資料品質，再做分析
- 數字要有單位和解讀
- 不過度解讀相關性 (correlation ≠ causation)
