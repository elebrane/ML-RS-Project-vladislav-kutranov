"""Packs FAR-Trans transactions into a compact binary for in-browser filtering (fartrans_overview.html).
Usage: python3 build_dashboard_cube.py "<path to Data FAR-Trans>"  -> dashboard_cube.json
Layout of tx blob (little-endian, gzip, base64): 4 Uint16 arrays of length n, concatenated:
  cust index | asset index | day since 2018-01-01 | flags (bit0 = Sell, bits1-2 = channel, bits3-8 = market index).
Rows are sorted by customer, then date (needed for distinct counts in the browser)."""
import sys, os, json, gzip, base64
import numpy as np, pandas as pd
D = sys.argv[1]
rd = lambda f, **k: pd.read_csv(os.path.join(D, f), **k)
tx = rd("transactions.csv", parse_dates=["timestamp"])
cu = rd("customer_information.csv", parse_dates=["timestamp"])
asi = rd("asset_information.csv", parse_dates=["timestamp"])
mk = rd("markets.csv"); cp = rd("close_prices.csv", parse_dates=["timestamp"])
b64 = lambda a: base64.b64encode(gzip.compress(a.tobytes(), 9)).decode()

# customers (latest snapshot)
cl = cu.sort_values("timestamp").groupby("customerID").tail(1).set_index("customerID")
custs = sorted(tx.customerID.unique()); cidx = {c: i for i, c in enumerate(custs)}
SEG = ["Mass", "Premium", "Professional", "Inactive", "Legal Entity"]
RISK = ["Conservative", "Income", "Balanced", "Aggressive", "Not_Available"]
CAP = ["CAP_LT30K", "CAP_30K_80K", "CAP_80K_300K", "CAP_GT300K", "Not_Available"]
def code(v, vocab):
    v = str(v); pred = v.startswith("Predicted_"); base = v.replace("Predicted_", "")
    if base == "GT300K": base = "CAP_GT300K"
    return vocab.index(base) + (8 if pred else 0)
c = cl.reindex(custs)
cust = np.zeros((len(custs), 3), dtype=np.uint8)
cust[:, 0] = [SEG.index(v) for v in c.customerType]
cust[:, 1] = [code(v, RISK) for v in c.riskLevel]
cust[:, 2] = [code(v, CAP) for v in c.investmentCapacity]

# assets traded (latest row), ordered by number of buys
al = asi.sort_values("timestamp").groupby("ISIN").tail(1).set_index("ISIN")
order = tx[tx.transactionType == "Buy"].ISIN.value_counts()
isins = list(order.index) + sorted(set(tx.ISIN) - set(order.index)); aidx = {a: i for i, a in enumerate(isins)}
CATS = ["Stock", "MTF", "Bond"]
sectors = sorted(al.loc[isins].sector.dropna().unique().tolist())
assets = [[a, str(al.at[a, "assetName"]), CATS.index(al.at[a, "assetCategory"]),
           (sectors.index(al.at[a, "sector"]) if pd.notna(al.at[a, "sector"]) else -1), str(al.at[a, "assetSubCategory"]) if pd.notna(al.at[a, "assetSubCategory"]) else ""] for a in isins]

# markets
mkt = mk.drop_duplicates("marketID").reset_index(drop=True); midx = {m: i for i, m in enumerate(mkt.marketID)}
markets = [[m, str(n), str(cn)] for m, n, cn in zip(mkt.marketID, mkt.name, mkt.country)]

# transactions
CH = ["Internet Banking", "Branch", "Phone Banking"]
t = tx.assign(ci=tx.customerID.map(cidx), ai=tx.ISIN.map(aidx), day=(tx.timestamp - pd.Timestamp("2018-01-01")).dt.days)
t = t.sort_values(["ci", "day", "ai"], kind="mergesort")
flags = (t.transactionType.eq("Sell").astype(int) + t.channel.map({c: i for i, c in enumerate(CH)}).astype(int) * 2 + t.marketID.map(midx).astype(int) * 8).astype(np.uint16)
blob = np.concatenate([t.ci.astype(np.uint16).values, t.ai.astype(np.uint16).values, t.day.astype(np.uint16).values, flags.values])

# weekly price sums/counts per category (for the filtered average-price chart)
cpc = cp.assign(cat=cp.ISIN.map(al.assetCategory)).dropna(subset=["cat"])
cpc["week"] = cpc.timestamp.dt.to_period("W-SUN").dt.start_time
g = cpc.groupby(["week", "cat"]).closePrice.agg(["sum", "count"]).unstack(fill_value=0)
weeks = [w.strftime("%Y-%m-%d") for w in g.index]
price = {cat: {"sum": [round(float(x), 2) for x in g["sum"].get(cat, pd.Series(0, index=g.index))], "count": [int(x) for x in g["count"].get(cat, pd.Series(0, index=g.index))]} for cat in CATS}

out = {"n": int(len(t)), "tx": b64(blob), "cust": b64(cust.reshape(-1)), "nCust": len(custs),
       "assets": assets, "sectors": sectors, "markets": markets, "channels": CH, "seg": SEG, "risk": RISK, "cap": CAP, "cats": CATS,
       "dayMin": int(t.day.min()), "dayMax": int(t.day.max()), "weeks": weeks, "price": price}
json.dump(out, open("dashboard_cube.json", "w", encoding="utf-8"), ensure_ascii=False)
print("rows", out["n"], "| tx b64 MB", round(len(out["tx"]) / 1e6, 2), "| cust b64 KB", round(len(out["cust"]) / 1e3, 1), "| assets", len(assets), "| sectors", len(sectors))

# sanity: how concentrated are buyers? (reported in chat)
b = tx[tx.transactionType == "Buy"]
top10 = set(order.index[:10]); top20 = set(order.index[:20])
buyers = b.groupby("customerID").ISIN.apply(set)
print("buyers", len(buyers), "| bought >=1 of top-10:", round(buyers.apply(lambda s: bool(s & top10)).mean() * 100, 1), "% | of top-20:", round(buyers.apply(lambda s: bool(s & top20)).mean() * 100, 1), "%")
print("top-1 buyers share", round(b[b.ISIN == order.index[0]].customerID.nunique() / len(buyers) * 100, 1), "%")
print("top-5 assets:", [(a, al.at[a, 'assetName'], int(order[a])) for a in order.index[:5]])
