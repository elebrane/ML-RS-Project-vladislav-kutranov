"""Popularity baseline on a FAR-Trans protocol that mirrors the paper (Sanz-Cruzado et al., 2024).
Assumptions (official course protocol pending):
  - cuts every 14 days from 2019-08-01 to 2022-05-23; test window [T, T + 6 months)
  - evaluated customers: have any transaction before T and buy >= 1 asset that is NEW to them in the window
  - candidates: assets with any transaction before T and a price series covering the whole window
  - relevant: candidate assets bought in the window that the customer never bought or sold before T
  - recommendation lists exclude assets the customer already interacted with before T
Popularity = number of distinct customers who bought the asset before T (as in the paper's code);
'recent' variant counts distinct buyers in the 90 days before T. Ties broken by ISIN.
Usage: python3 baseline_popularity.py "<path to Data FAR-Trans>"  -> baseline_popularity_by_cut.csv"""
import sys, os, math, time
import numpy as np, pandas as pd
D = sys.argv[1]
t0 = time.time()
tx = pd.read_csv(os.path.join(D, "transactions.csv"), usecols=["customerID", "ISIN", "transactionType", "timestamp"], parse_dates=["timestamp"])
lp = pd.read_csv(os.path.join(D, "limit_prices.csv"), parse_dates=["minDate", "maxDate"])
buy = tx.transactionType.eq("Buy")
LOG = [1.0 / math.log2(i + 2) for i in range(50)]
IDCG = np.cumsum(LOG)

def evaluate(order, rel, seen, K=10):
    nd, r10, hit, r20, r30, r50 = [], [], [], [], [], []
    for u, R in rel.items():
        S = seen[u]; recs = []
        for a in order:
            if a not in S:
                recs.append(a)
                if len(recs) == 50: break
        dcg = sum(LOG[i] for i, a in enumerate(recs[:K]) if a in R)
        nd.append(dcg / IDCG[min(len(R), K) - 1])
        h10 = len(R.intersection(recs[:10])); r10.append(h10 / len(R)); hit.append(h10 > 0)
        r20.append(len(R.intersection(recs[:20])) / len(R)); r30.append(len(R.intersection(recs[:30])) / len(R)); r50.append(len(R.intersection(recs)) / len(R))
    return np.mean(nd), np.mean(r10), np.mean(hit), np.mean(r20), np.mean(r30), np.mean(r50)

rows = []
for T in pd.date_range("2019-08-01", "2022-05-23", freq="14D"):
    E = T + pd.DateOffset(months=6)
    tr = tx[tx.timestamp < T]
    te = tx[buy & (tx.timestamp >= T) & (tx.timestamp < E)]
    priced = set(lp.ISIN[(lp.minDate <= T) & (lp.maxDate >= E)])
    cand = set(tr.ISIN) & priced
    te = te[te.ISIN.isin(cand) & te.customerID.isin(set(tr.customerID))]
    users = set(te.customerID)
    seen = tr[tr.customerID.isin(users)].groupby("customerID").ISIN.agg(set).to_dict()
    rel = {}
    for u, s in te.groupby("customerID").ISIN.agg(set).items():
        r = s - seen[u]
        if r: rel[u] = r
    trb = tr[buy.loc[tr.index] & tr.ISIN.isin(cand)]
    def order_of(df):
        c = df.groupby("ISIN").customerID.nunique().reindex(sorted(cand), fill_value=0)
        return list(c.sort_values(ascending=False, kind="mergesort").index)
    o_all = order_of(trb)
    o_rec = order_of(trb[trb.timestamp >= T - pd.Timedelta(days=90)])
    ra = evaluate(o_all, rel, seen); rr = evaluate(o_rec, rel, seen)
    rows.append([T.date(), len(rel), len(cand), np.mean([len(r) for r in rel.values()]), *ra, *rr])
cols = ["cut", "users", "candidates", "rel_per_user",
        "pop_ndcg10", "pop_recall10", "pop_hit10", "pop_recall20", "pop_recall30", "pop_recall50",
        "recent_ndcg10", "recent_recall10", "recent_hit10", "recent_recall20", "recent_recall30", "recent_recall50"]
df = pd.DataFrame(rows, columns=cols)
df.to_csv("baseline_popularity_by_cut.csv", index=False, float_format="%.4f")
print(f"cuts {len(df)} | runtime {time.time() - t0:.1f}s")
print(df.drop(columns=["cut"]).mean().round(4).to_string())
print("\nby year (mean nDCG@10):")
print(df.assign(y=pd.to_datetime(df.cut).dt.year).groupby("y")[["users", "pop_ndcg10", "recent_ndcg10"]].mean().round(4).to_string())
