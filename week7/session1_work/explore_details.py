"""Step 1 (details): duplicate transaction ids, top assets, yearly volume, profile timing, candidate record."""
import sys, os
import pandas as pd
D = sys.argv[1]
tx = pd.read_csv(os.path.join(D, "transactions.csv"), parse_dates=["timestamp"])
cu = pd.read_csv(os.path.join(D, "customer_information.csv"), parse_dates=["timestamp", "lastQuestionnaireDate"])
asi = pd.read_csv(os.path.join(D, "asset_information.csv"), parse_dates=["timestamp"])
al = asi.sort_values("timestamp").groupby("ISIN").tail(1).set_index("ISIN")

print("## Duplicate transactionID")
d = tx[tx.transactionID.duplicated(keep=False)]
print("rows involved", len(d), "| ids", d.transactionID.nunique())
print("fully identical rows", tx.duplicated().sum())
g = d.groupby("transactionID").agg(nc=("customerID", "nunique"), ni=("ISIN", "nunique"), nt=("transactionType", "nunique"), nd=("timestamp", "nunique"))
print("ids differing in customer", (g.nc > 1).sum(), "| asset", (g.ni > 1).sum(), "| type", (g.nt > 1).sum(), "| date", (g.nd > 1).sum())
print(d.sort_values("transactionID").head(6).to_string(index=False))

print("\n## Buys per year")
b = tx[tx.transactionType == "Buy"]
print(b.timestamp.dt.year.value_counts().sort_index().to_dict())

print("\n## Top-10 assets by Buy transactions")
t = b.ISIN.value_counts().head(10).rename("buys").to_frame()
t["buyers"] = b.groupby("ISIN").customerID.nunique().reindex(t.index)
t = t.join(al[["assetCategory", "assetSubCategory", "marketID", "sector"]])
print(t.to_string())
print("\nBuys by marketID (top 5 share):", b.marketID.value_counts(normalize=True).head(5).round(3).to_dict())

print("\n## Profile snapshot timing")
print("snapshot timestamp == 2000-01-01:", int((cu.timestamp == "2000-01-01").sum()))
print("snapshots per year:", cu.timestamp.dt.year.value_counts().sort_index().to_dict())
print("lastQuestionnaireDate range", cu.lastQuestionnaireDate.min().date(), "->", cu.lastQuestionnaireDate.max().date())
nq = cu.groupby("customerID").lastQuestionnaireDate.nunique()
print("customers whose lastQuestionnaireDate differs across snapshots", int((nq > 1).sum()))
ni = cu.groupby("customerID").investmentCapacity.nunique()
print("customers whose investmentCapacity differs across snapshots", int((ni > 1).sum()))
pr = cu[cu.riskLevel.str.startswith("Predicted")]
print("predicted riskLevel rows: lastQuestionnaireDate range", pr.lastQuestionnaireDate.min().date(), "->", pr.lastQuestionnaireDate.max().date())

print("\n## Candidate records: Buy of a stock with sector, customer with 2 snapshots, observed risk, questionnaire after the buy")
multi = cu.groupby("customerID").filter(lambda x: len(x) == 2)
multi = multi[~multi.riskLevel.str.startswith(("Predicted", "Not_Available"))]
cand = b[b.customerID.isin(multi.customerID) & b.ISIN.map(al.assetCategory).eq("Stock") & b.ISIN.map(al.sector).notna()]
cand = cand.merge(cu.groupby("customerID").lastQuestionnaireDate.max().rename("lq"), left_on="customerID", right_index=True)
cand = cand[(cand.lq > cand.timestamp) & (cand.timestamp.dt.year == 2021)]
print("candidates", len(cand))
row = cand.sort_values("transactionID").iloc[len(cand) // 2]
print("\nTRANSACTION ROW:"); print(tx[tx.transactionID == row.transactionID].to_string(index=False))
print("\nPROFILE ROWS:"); print(cu[cu.customerID == row.customerID].to_string(index=False))
print("\nASSET ROW(S):"); print(asi[asi.ISIN == row.ISIN].to_string(index=False))
hist = b[(b.customerID == row.customerID)].sort_values("timestamp")
print("\ncustomer buys total", len(hist), "| distinct assets", hist.ISIN.nunique(), "| first", hist.timestamp.min().date(), "| last", hist.timestamp.max().date())
print("same asset bought before this date:", int(((hist.ISIN == row.ISIN) & (hist.timestamp < row.timestamp)).sum()))
