"""Builds dashboard_data.json for fartrans_overview.html from the raw FAR-Trans CSVs.
Usage: python3 build_dashboard_data.py "<path to Data FAR-Trans>"
Assumption for the protocol block (Session 1, today): already-bought assets are neither recommended nor counted
(the FAR-Trans paper rule: test pairs the customer interacted with before the cut are removed)."""
import sys, os, json
import numpy as np, pandas as pd

D = sys.argv[1]
rd = lambda f, **k: pd.read_csv(os.path.join(D, f), **k)
tx = rd("transactions.csv", parse_dates=["timestamp"])
cu = rd("customer_information.csv", parse_dates=["timestamp", "lastQuestionnaireDate"])
asi = rd("asset_information.csv", parse_dates=["timestamp"])
mk = rd("markets.csv")
cp = rd("close_prices.csv", parse_dates=["timestamp"])
lp = rd("limit_prices.csv", parse_dates=["minDate", "maxDate"])
out = {}

# ---------- schema & integrity ----------
def uniq(df, cols): return bool(not df.duplicated(cols).any())
def orphans(child, ckey, parent, pkey):
    s = set(child[ckey].dropna().astype(str)) - set(parent[pkey].dropna().astype(str))
    return {"orphan_keys": len(s), "orphan_rows": int(child[ckey].astype(str).isin(s).sum()), "examples": sorted(s)[:3]}
out["tables"] = {
    "transactions": {"rows": len(tx), "cols": list(tx.columns), "key_unique": {"transactionID": uniq(tx, ["transactionID"]), "transactionID+customerID": uniq(tx, ["transactionID", "customerID"])}},
    "customer_information": {"rows": len(cu), "cols": list(cu.columns), "entities": int(cu.customerID.nunique()), "key_unique": {"customerID": uniq(cu, ["customerID"]), "customerID+timestamp": uniq(cu, ["customerID", "timestamp"])}},
    "asset_information": {"rows": len(asi), "cols": list(asi.columns), "entities": int(asi.ISIN.nunique()), "key_unique": {"ISIN": uniq(asi, ["ISIN"]), "ISIN+timestamp": uniq(asi, ["ISIN", "timestamp"])}},
    "markets": {"rows": len(mk), "cols": list(mk.columns), "key_unique": {"marketID": uniq(mk, ["marketID"]), "exchangeID": uniq(mk, ["exchangeID"])}},
    "close_prices": {"rows": len(cp), "cols": list(cp.columns), "entities": int(cp.ISIN.nunique()), "key_unique": {"ISIN+timestamp": uniq(cp, ["ISIN", "timestamp"])}},
    "limit_prices": {"rows": len(lp), "cols": list(lp.columns), "key_unique": {"ISIN": uniq(lp, ["ISIN"])}},
}
out["fk"] = {
    "transactions.customerID -> customer_information.customerID": orphans(tx, "customerID", cu, "customerID"),
    "transactions.ISIN -> asset_information.ISIN": orphans(tx, "ISIN", asi, "ISIN"),
    "transactions.marketID -> markets.marketID": orphans(tx, "marketID", mk, "marketID"),
    "asset_information.marketID -> markets.marketID": orphans(asi, "marketID", mk, "marketID"),
    "close_prices.ISIN -> asset_information.ISIN": orphans(cp, "ISIN", asi, "ISIN"),
    "limit_prices.ISIN -> asset_information.ISIN": orphans(lp, "ISIN", asi, "ISIN"),
    "asset_information.ISIN -> close_prices.ISIN": orphans(asi, "ISIN", cp, "ISIN"),
}
al = asi.sort_values("timestamp").groupby("ISIN").tail(1).set_index("ISIN")
m = tx[["ISIN", "marketID"]].drop_duplicates().merge(al[["marketID"]].rename(columns={"marketID": "assetMarket"}), left_on="ISIN", right_index=True, how="left")
out["tx_market_matches_asset_market_share"] = round(float((m.marketID == m.assetMarket).mean()), 4)
shared = tx[tx.transactionID.duplicated(keep=False)]
out["shared_tx_ids"] = {"ids": int(shared.transactionID.nunique()), "rows": len(shared)}

# ---------- Table 1 (paper) recomputed ----------
b = tx[tx.transactionType == "Buy"]; s = tx[tx.transactionType == "Sell"]
up = lambda d: int(d.groupby(["customerID", "ISIN"]).ngroups)
out["table1"] = [
    ["Unique assets", int(asi.ISIN.nunique()), 806],
    ["Assets with investments", int(tx.ISIN.nunique()), 321],
    ["Unique markets", int(mk.marketID.nunique()), 38],
    ["Price data points", len(cp), 703303],
    ["Average return (by assets, whole period), %", round(float(lp.profitability.mean() * 100), 2), 37.16],
    ["% profitable assets", round(float((lp.profitability > 0).mean() * 100), 2), 54.28],
    ["Unique customers", int(tx.customerID.nunique()), 29090],
    ["Transactions", len(tx), 388049],
    ["Transactions, unique customer-asset-type", up(b) + up(s), 154103],
    ["Acquisitions", len(b), 228913],
    ["Acquisitions, unique customer-asset", up(b), 89884],
    ["Sales", len(s), 159136],
    ["Sales, unique customer-asset", up(s), 64219],
    ["Average return (by customers, whole period), %", None, 22.89],
    ["% customers with profits", None, 54.56],
]

# ---------- Figure 1 ----------
avg = cp.groupby("timestamp").closePrice.mean()
w = avg.resample("W").mean().dropna()
out["fig1a_avg_price_weekly"] = [[d.strftime("%Y-%m-%d"), round(float(v), 2)] for d, v in w.items()]
out["fig1a_assets_per_day_median"] = int(cp.groupby("timestamp").ISIN.nunique().median())
bought = set(b.ISIN)
cat = al.assetCategory.value_counts()
out["fig1b_category"] = [[c, int(cat[c]), int(sum(1 for i in al.index[al.assetCategory == c] if i in bought)),
                          round(float((b.ISIN.map(al.assetCategory) == c).mean() * 100), 1)] for c in ["MTF", "Stock", "Bond"]]
ctry = al.marketID.map(mk.drop_duplicates("marketID").set_index("marketID").country).fillna("Unknown").value_counts()
out["fig1c_country"] = [[k, int(v)] for k, v in ctry.items()]

# ---------- Figure 2 ----------
mon = tx.groupby([tx.timestamp.dt.to_period("M"), "transactionType"]).size().unstack(fill_value=0)
out["fig2a_monthly"] = [[str(p), int(r.get("Buy", 0)), int(r.get("Sell", 0))] for p, r in mon.iterrows()]
pc = tx.groupby("customerID").size().value_counts().sort_index()
out["fig2b_customers_by_ntx"] = [[int(k), int(v)] for k, v in pc.items()]
pa = tx.groupby("ISIN").size().value_counts().sort_index()
out["fig2c_assets_by_ntx"] = [[int(k), int(v)] for k, v in pa.items()]
ntx = tx.groupby("customerID").size()
out["cust_tx_facts"] = {"share_le3": round(float((ntx <= 3).mean() * 100), 1), "gt100": int((ntx > 100).sum()), "median": float(ntx.median())}

# ---------- Figure 3 (latest snapshot per customer) ----------
cl = cu.sort_values("timestamp").groupby("customerID").tail(1)
out["fig3a_segment"] = [[k, int(v)] for k, v in cl.customerType.value_counts().items()]
def split(col, order):
    base = cl[col].str.replace("Predicted_", "", regex=False)
    pred = cl[col].str.startswith("Predicted_")
    return [[o, int(((base == o) & ~pred).sum()), int(((base == o) & pred).sum())] for o in order]
out["fig3b_risk"] = split("riskLevel", ["Conservative", "Income", "Balanced", "Aggressive", "Not_Available"])
cap_norm = cl.investmentCapacity.str.replace("CAP_GT300K", "CAP_GT_300K").str.replace("Predicted_GT300K", "Predicted_CAP_GT_300K")
cl = cl.assign(investmentCapacity=cap_norm)
out["cap_values_seen"] = sorted(cl.investmentCapacity.unique().tolist())
out["fig3c_capacity"] = split("investmentCapacity", ["CAP_LT30K", "CAP_30K_80K", "CAP_80K_300K", "CAP_GT_300K", "Not_Available"])

# ---------- Concentration & histories ----------
pop = b.ISIN.value_counts(); cum = (pop.cumsum() / pop.sum() * 100).round(2)
out["concentration"] = [[i + 1, float(v)] for i, v in enumerate(cum.values)]
na = b.groupby("customerID").ISIN.nunique()
bins = [(1, 1, "1"), (2, 2, "2"), (3, 3, "3"), (4, 5, "4–5"), (6, 10, "6–10"), (11, 10**9, "11+")]
out["distinct_assets_hist"] = [[lab, int(((na >= lo) & (na <= hi)).sum())] for lo, hi, lab in bins]
bs = b.sort_values("timestamp"); first = bs.groupby(["customerID", "ISIN"]).timestamp.transform("min")
out["kpi"] = {
    "buy_share_stocks": round(float((b.ISIN.map(al.assetCategory) == "Stock").mean() * 100), 1),
    "buy_share_xath": round(float((b.marketID == "XATH").mean() * 100), 1),
    "top12_share": float(cum.iloc[11]), "top10_share": float(cum.iloc[9]),
    "median_distinct_assets": float(na.median()), "share_one_asset": round(float((na == 1).mean() * 100), 1),
    "rebuy_row_share": round(float((bs.timestamp > first).mean() * 100), 1),
    "risk_predicted_share": round(float(cl.riskLevel.str.startswith("Predicted_").mean() * 100), 1),
    "quest_after_snapshot_share": round(float((cu.lastQuestionnaireDate > cu.timestamp).mean() * 100), 1),
    "first_profile_after_first_buy_share": round(float((lambda j: (j.snap > j.buy).mean())(pd.concat([cu.groupby("customerID").timestamp.min(), b.groupby("customerID").timestamp.min()], axis=1, keys=["snap", "buy"]).dropna()) * 100), 1),
    "opening_day_buys": int((b.timestamp == tx.timestamp.min()).sum()),
    "date_min": tx.timestamp.min().strftime("%Y-%m-%d"), "date_max": tx.timestamp.max().strftime("%Y-%m-%d"),
}

# ---------- Protocol under the 'new assets only' assumption (paper rule), 6-month windows, biweekly cuts ----------
rows = []
cuts = pd.date_range("2019-08-01", tx.timestamp.max() - pd.DateOffset(months=6), freq="14D")
lpi = lp.set_index("ISIN")
for T in cuts:
    E = T + pd.DateOffset(months=6)
    tr = tx[tx.timestamp < T]; te = b[(b.timestamp >= T) & (b.timestamp < E)]
    priced = set(lpi.index[(lpi.minDate <= T) & (lpi.maxDate >= E)])           # asset with prices over the whole test window
    train_items = set(tr.ISIN)
    te = te[te.customerID.isin(set(tr.customerID)) & te.ISIN.isin(priced & train_items)]
    tep = te[["customerID", "ISIN"]].drop_duplicates()
    seen = tr[["customerID", "ISIN"]].drop_duplicates().assign(seen=1)
    tep = tep.merge(seen, on=["customerID", "ISIN"], how="left")
    new = tep[tep.seen.isna()]
    nu = new.customerID.nunique()
    rows.append([T.strftime("%Y-%m-%d"), int(nu), round(float(len(new) / nu), 2) if nu else 0.0,
                 round(float(tep.seen.notna().mean() * 100), 1) if len(tep) else 0.0, len(priced & train_items)])
out["protocol"] = rows

with open("dashboard_data.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False)
print(json.dumps({k: out[k] for k in ["fk", "table1", "tx_market_matches_asset_market_share", "shared_tx_ids", "kpi", "cap_values_seen", "fig1b_category", "fig3b_risk", "fig3c_capacity", "cust_tx_facts"]}, ensure_ascii=False, indent=0)[:6000])
print("tables", json.dumps(out["tables"], ensure_ascii=False)[:2500])
print("protocol first/last", rows[:2], rows[-2:], "n", len(rows))
print("country", out["fig1c_country"][:12])
