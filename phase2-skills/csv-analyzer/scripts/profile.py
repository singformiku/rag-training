#!/usr/bin/env python3
import sys
import json
import pandas as pd


def profile(path):
    df = pd.read_csv(path, low_memory=False)
    cols = [
        {
            "name": c,
            "dtype": str(df[c].dtype),
            "null_count": int(df[c].isna().sum()),
            "unique_count": int(df[c].nunique(dropna=True)),
            "sample_values": df[c].dropna().head(3).astype(str).tolist(),
        }
        for c in df.columns
    ]
    return {"row_count": len(df), "column_count": df.shape[1], "columns": cols}


if __name__ == "__main__":
    print(json.dumps(profile(sys.argv[1]), ensure_ascii=False, indent=2))
