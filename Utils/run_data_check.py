import pandas as pd
import os
import sys
import json

repo_root = os.getcwd()
if repo_root not in sys.path:
    sys.path.append(repo_root)

from Utils.db import get_item_history  # noqa: E402

with open("Backend/bazaar_full_items_ids.json") as f:
    items = json.load(f)

for item in items:
    data = get_item_history(item, order_by="ASC")
    if not data:
        print(f"{item:<25} | No data")
        continue
    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["gap"] = df["timestamp"].diff().dt.total_seconds() / 60

    med = df["gap"].median()
    max_gap = df["gap"].max()
    gaps_10 = (df["gap"] > 10).sum()
    print(
        f"{item:<25} | Rows: {len(df):<6} | Median: {med:.1f}m | Max Gap: {max_gap:.1f}m | Gaps >10m: {gaps_10}"
    )
