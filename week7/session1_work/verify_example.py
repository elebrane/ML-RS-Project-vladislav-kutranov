"""Worked example for hand-checking nDCG@10 of the popularity baseline on one customer (Session 1 §4 verification).
Same protocol as baseline_popularity.py; cut 2021-11-25 (one of the 74 cuts); picks the first customer, in ID order,
with exactly 2 relevant new assets and exactly one hit below rank 1 in the top-10."""
import sys, os, math
import pandas as pd
D = sys.argv[1]
tx = pd.read_csv(os.path.join(D, "transactions.csv"), usecols=["customerID", "ISIN", "transactionType", "timestamp"], parse_dates=["timestamp"])
lp = pd.read_csv(os.path.join(D, "limit_prices.csv"), parse_dates=["minDate", "maxDate"])
names = pd.read_csv(os.path.join(D, "asset_information.csv")).drop_duplicates("ISIN").set_index("ISIN").assetName
T = pd.Timestamp("2021-11-25"); E = T + pd.DateOffset(months=6)
buy = tx.transactionType.eq("Buy")
tr = tx[tx.timestamp < T]; te = tx[buy & (tx.timestamp >= T) & (tx.timestamp < E)]
cand = set(tr.ISIN) & set(lp.ISIN[(lp.minDate <= T) & (lp.maxDate >= E)])
te = te[te.ISIN.isin(cand) & te.customerID.isin(set(tr.customerID))]
seen = tr[tr.customerID.isin(set(te.customerID))].groupby("customerID").ISIN.agg(set).to_dict()
trb = tr[buy.loc[tr.index] & tr.ISIN.isin(cand)]
pop = trb.groupby("ISIN").customerID.nunique().reindex(sorted(cand), fill_value=0).sort_values(ascending=False, kind="mergesort")
order = list(pop.index)
for u, s in sorted(te.groupby("customerID").ISIN.agg(set).items()):
    R = s - seen[u]
    if len(R) != 2: continue
    recs = [a for a in order if a not in seen[u]][:10]
    hits = [i + 1 for i, a in enumerate(recs) if a in R]
    if len(hits) == 1 and hits[0] > 1:
        print("cut", T.date(), "| window", T.date(), "->", (E - pd.Timedelta(days=1)).date(), "| customer", u[:6] + "…")
        print("history before cut (distinct assets):", len(seen[u]), [names.get(a, a) for a in sorted(seen[u])][:6])
        print("relevant NEW assets bought in window:", [(a, names.get(a, a)) for a in sorted(R)])
        for i, a in enumerate(recs):
            print(f"  rank {i+1:2d}  {a}  {names.get(a, a)[:40]:40s}  buyers_before_cut={pop[a]:5d}  {'HIT' if a in R else ''}")
        dcg = sum(1 / math.log2(r + 1) for r in hits); idcg = 1 / math.log2(2) + 1 / math.log2(3)
        print(f"DCG = 1/log2({hits[0]}+1) = {dcg:.4f}; IDCG (2 relevant) = 1/log2(2) + 1/log2(3) = {idcg:.4f}; nDCG@10 = {dcg/idcg:.4f}")
        break
