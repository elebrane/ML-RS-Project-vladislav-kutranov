"""Session 1, §4 data example: one random transactions.csv row (fixed seed) + minimal fields for a popularity / interaction baseline.
Usage: python3 pick_record.py "<path to Data FAR-Trans>" """
import sys, os
import pandas as pd
D = sys.argv[1]; SEED = 2026
T = pd.Timestamp("2022-05-30")            # assumed cut: last 6 months = test window (pending official protocol)
E = pd.Timestamp("2022-11-30")
raw = pd.read_csv(os.path.join(D, "transactions.csv"), dtype=str)           # verbatim strings
tx = raw.assign(ts=pd.to_datetime(raw.timestamp))
row = raw.sample(1, random_state=SEED).iloc[0]
print("row index (0-based, header excluded):", row.name, "| csv line:", row.name + 2)
print("VERBATIM:", ",".join(raw.loc[row.name].tolist()))
asi = pd.read_csv(os.path.join(D, "asset_information.csv"), dtype=str)
a = asi[(asi.ISIN == row.ISIN)]
print("asset rows:\n", a[["ISIN", "assetName", "assetCategory", "marketID", "timestamp"]].to_string(index=False))
cust = tx[tx.customerID == row.customerID].sort_values("ts")
same = cust[cust.ISIN == row.ISIN]
d = pd.Timestamp(row.timestamp)
side = "train/history (<= T)" if d <= T else ("test window (T, E]" if d <= E else "after E")
print("cut T:", T.date(), "| row side:", side)
print("customer rows total:", len(cust), "| buys:", (cust.transactionType == "Buy").sum(), "| first:", cust.ts.min().date(), "| last:", cust.ts.max().date())
print("customer rows before T:", (cust.ts <= T).sum(), "| buys in test window:", ((cust.ts > T) & (cust.ts <= E) & (cust.transactionType == "Buy")).sum())
print("same asset by this customer before this row date:", (same.ts < d).sum(), "| same date:", (same.ts == d).sum() - 1, "| after:", (same.ts > d).sum())
print("same transactionID shared with other customers:", (raw.transactionID == row.transactionID).sum() - 1)
print("asset popularity rank by buys before T:", int((tx[(tx.ts <= T) & (tx.transactionType == 'Buy')].ISIN.value_counts().index.get_loc(row.ISIN) + 1)) if row.ISIN in set(tx[(tx.ts <= T) & (tx.transactionType == 'Buy')].ISIN) else "not bought before T")
