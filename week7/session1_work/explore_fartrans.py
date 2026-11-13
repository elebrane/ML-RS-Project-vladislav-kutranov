"""Step 1 of Session 1: descriptive statistics of the raw FAR-Trans release.
Usage: python3 explore_fartrans.py "<path to Data FAR-Trans>"
Only pandas/numpy; reads the raw CSVs, prints the numbers used in the report."""
import sys, os
import numpy as np, pandas as pd

D = sys.argv[1]
tx = pd.read_csv(os.path.join(D, "transactions.csv"), parse_dates=["timestamp"])
cu = pd.read_csv(os.path.join(D, "customer_information.csv"), parse_dates=["timestamp", "lastQuestionnaireDate"])
asi = pd.read_csv(os.path.join(D, "asset_information.csv"), parse_dates=["timestamp"])
lp = pd.read_csv(os.path.join(D, "limit_prices.csv"), parse_dates=["minDate", "maxDate"])

def h(t): print("\n## " + t)

h("Transactions")
print("rows", len(tx), "| Buy", (tx.transactionType == "Buy").sum(), "| Sell", (tx.transactionType == "Sell").sum())
print("customers", tx.customerID.nunique(), "| assets", tx.ISIN.nunique(), "| dates", tx.timestamp.min().date(), "->", tx.timestamp.max().date())
print("duplicate transactionID", tx.transactionID.duplicated().sum())
print("channel share:", tx.channel.value_counts(normalize=True).round(3).to_dict())
b = tx[tx.transactionType == "Buy"].copy()
print("buyers", b.customerID.nunique(), "| assets bought", b.ISIN.nunique(), "| unique (customer,asset) buy pairs", b.groupby(["customerID", "ISIN"]).ngroups)

h("Activity per buyer")
nb = b.groupby("customerID").size(); na = b.groupby("customerID").ISIN.nunique()
print("buys per buyer: median", nb.median(), "mean", round(nb.mean(), 2), "p90", nb.quantile(.9), "max", nb.max())
print("distinct assets per buyer: median", na.median(), "mean", round(na.mean(), 2), "| share with 1 asset", round((na == 1).mean(), 3), "| <=3", round((na <= 3).mean(), 3))

h("Popularity concentration (Buy transactions)")
pop = b.ISIN.value_counts()
cum = pop.cumsum() / pop.sum()
print("top-1 share", round(cum.iloc[0], 3), "| top-10", round(cum.iloc[9], 3), "| top-12", round(cum.iloc[11], 3), "| top-50", round(cum.iloc[49], 3))
print("assets needed for 50% of buys", int((cum < .5).sum() + 1), "| for 80%", int((cum < .8).sum() + 1))
popu = b.groupby("ISIN").customerID.nunique().sort_values(ascending=False)
print("top-10 by distinct buyers cover", round(popu.iloc[:10].sum() / b.groupby(['customerID','ISIN']).ngroups, 3), "of unique buy pairs")

h("Repeat purchases")
bs = b.sort_values(["customerID", "ISIN", "timestamp"])
first = bs.groupby(["customerID", "ISIN"]).timestamp.transform("min")
print("share of Buy rows on a (customer,asset) pair already bought on an EARLIER date", round((bs.timestamp > first).mean(), 3))
pc = bs.groupby(["customerID", "ISIN"]).timestamp.nunique()
print("share of (customer,asset) pairs bought on >1 distinct date", round((pc > 1).mean(), 3))

h("Buys by asset category (asset_information latest row per ISIN)")
al = asi.sort_values("timestamp").groupby("ISIN").tail(1).set_index("ISIN")
print("asset_information rows", len(asi), "| unique ISIN", asi.ISIN.nunique())
print("catalogue by category:", al.assetCategory.value_counts().to_dict())
bc = b.ISIN.map(al.assetCategory)
print("buys by category share:", bc.value_counts(normalize=True).round(3).to_dict(), "| unmapped", bc.isna().sum())
print("sector missing share (catalogue)", round(al.sector.isna().mean(), 3), "| industry missing", round(al.industry.isna().mean(), 3))
print("sector missing by category:", al.groupby("assetCategory").sector.apply(lambda s: round(s.isna().mean(), 2)).to_dict())

h("Customer information")
print("rows", len(cu), "| unique customers", cu.customerID.nunique(), "| customers with >1 snapshot", (cu.groupby("customerID").size() > 1).sum())
print("snapshot timestamp range", cu.timestamp.min().date(), "->", cu.timestamp.max().date())
def kind(v):
    v = str(v)
    return "missing" if v.startswith("Not_Available") or v == "nan" else ("predicted" if v.startswith("Predicted") else "observed")
for col in ["riskLevel", "investmentCapacity"]:
    print(col, "rows:", cu[col].map(kind).value_counts(normalize=True).round(3).to_dict())
cl = cu.sort_values("timestamp").groupby("customerID").tail(1)
for col in ["riskLevel", "investmentCapacity"]:
    print(col, "latest snapshot per customer:", cl[col].map(kind).value_counts(normalize=True).round(3).to_dict())
print("customerType (latest):", cl.customerType.value_counts().to_dict())
qq = cu.lastQuestionnaireDate.notna()
print("rows with lastQuestionnaireDate", int(qq.sum()), "| of them lastQuestionnaireDate > snapshot timestamp:", round((cu.lastQuestionnaireDate[qq] > cu.timestamp[qq]).mean(), 3))
chg = cu.groupby("customerID").riskLevel.nunique()
print("customers whose riskLevel changes across snapshots", int((chg > 1).sum()))
print("buyers without any profile row", len(set(b.customerID) - set(cu.customerID)))
fsnap = cu.groupby("customerID").timestamp.min(); fbuy = b.groupby("customerID").timestamp.min()
j = pd.concat([fsnap, fbuy], axis=1, keys=["snap", "buy"]).dropna()
print("buyers whose first profile snapshot is AFTER their first buy:", round((j.snap > j.buy).mean(), 3))

h("Prices")
print("assets with price series", lp.ISIN.nunique(), "| bought assets without price series", len(set(b.ISIN) - set(lp.ISIN)))

h("Time cuts: 6-month window after cut T (Buy only)")
for T in ["2020-05-29", "2020-11-29", "2021-05-29", "2021-11-29", "2022-05-29"]:
    T = pd.Timestamp(T); E = T + pd.DateOffset(months=6)
    tr = b[b.timestamp <= T]; te = b[(b.timestamp > T) & (b.timestamp <= E)]
    trp = set(zip(tr.customerID, tr.ISIN)); tep = set(zip(te.customerID, te.ISIN))
    warm = set(te.customerID) & set(tr.customerID)
    tepw = {p for p in tep if p[0] in warm}
    rep = len([p for p in tepw if p in trp]) / max(len(tepw), 1)
    usr_new = sum(1 for u in warm if any((u, i) not in trp for i in te[te.customerID == u].ISIN.unique()))
    print(f"T={T.date()} test buyers {te.customerID.nunique()} | warm (also in train) {len(warm)} ({len(warm)/max(te.customerID.nunique(),1):.1%}) | "
          f"warm test pairs already bought in train {rep:.1%} | warm users with >=1 NEW asset {usr_new}")
