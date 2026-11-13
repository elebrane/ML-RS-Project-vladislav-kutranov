# Session 1 — Step 1: FAR-Trans raw data notes

Source: public FAR-Trans release, University of Glasgow research data repository (https://researchdata.gla.ac.uk/1658/), files in `../Data FAR-Trans/`.
Reproduce: `python3 explore_fartrans.py "../Data FAR-Trans"` and `python3 explore_details.py "../Data FAR-Trans"` (outputs: `data_stats.txt`, `data_details.txt`).
All numbers below are descriptive statistics of the raw files, not experiment results.

## 1. Key statistics

| Item | Value |
|---|---|
| Transactions | 388,048 rows (228,913 Buy, 159,135 Sell), 2018-01-02 → 2022-11-30 |
| Customers / assets traded | 29,090 customers (every customer has ≥1 Buy) / 320 assets bought, out of 806 in the catalogue |
| Catalogue by category | 313 mutual funds (MTF), 284 stocks, 209 bonds |
| **Buys by category** | **Stocks 88.9%**, funds 6.1%, bonds 5.0%; 87.7% of buys on the Athens Stock Exchange (XATH) |
| Popularity concentration | top-1 asset = 11.2% of buys; top-10 = 48.2%; **12 assets = 51.7%**; 44 assets = 80% |
| Top-10 assets | all Greek large-cap stocks (XATH) |
| History length | distinct assets per customer: median 1, mean 3.09; 51.3% bought exactly one asset, 78.6% ≤ 3 |
| Repeat buying | 60.5% of Buy rows re-buy an asset the customer bought on an earlier date; 30.7% of customer–asset pairs are bought on >1 date |
| Profile coverage (latest snapshot) | riskLevel: 74.4% observed, 24.5% algorithm-predicted, 1.1% missing (investmentCapacity similar) |
| Asset text fields | sector missing for 100% of funds, 20% of stocks, 5% of bonds |

### 6-month windows after a cut T (Buy only; assumed protocol, not the official one)

| Cut T | Test buyers | With prior buys ("warm") | Warm test pairs already bought before T |
|---|---|---|---|
| 2020-05-29 | 4,223 | 82.9% | 58.7% |
| 2020-11-29 | 6,378 | 68.1% | 49.0% |
| 2021-05-29 | 7,296 | 65.8% | 42.5% |
| 2021-11-29 | 5,339 | 82.2% | 59.0% |
| 2022-05-29 | 3,717 | 85.9% | 64.5% |

→ Whether repeat purchases count changes the task: 42–65% of warm users' future customer–asset pairs are re-buys. 14–34% of future buyers have no prior buys (cold start).

## 2. Data quirks to handle

1. **Shared transaction IDs.** 23,388 transactionIDs appear for 2–5 different customers with identical asset, type, date, value and units (likely joint holders of one account — hypothesis). One row per transactionID leaves 213,639 Buy rows instead of 228,913.
2. **Start-of-data spike.** 35,686 Buy rows (18,964 customers) are dated 2018-01-02, the first day of the data; possibly an opening-portfolio snapshot rather than real purchases (hypothesis).
3. **Profile rows are back-filled, not as-of.** riskLevel and investmentCapacity never change across a customer's snapshots; in 38.5% of profile rows lastQuestionnaireDate is later than the row timestamp; 11.6% of customers' first profile row is dated after their first buy; 130 rows carry a placeholder date 2000-01-01; all Predicted_* rows have lastQuestionnaireDate = 2000-01-01.

## 3. One actual record (anonymized)

```
source: FAR-Trans public release (researchdata.gla.ac.uk/1658), transactions.csv + customer_information.csv + asset_information.csv
record id: transactionID 8912167
fields: customerID, ISIN, transactionID, transactionType, timestamp, totalValue, units, channel, marketID
values: 4AEA…(masked), GRS393503008, 8912167, Buy, 2021-05-12, 1490.0, 100.0, Internet Banking, XATH
customer profile rows (customer_information.csv):
  customerType, riskLevel, investmentCapacity, lastQuestionnaireDate, timestamp
  Premium, Income, CAP_LT30K, 2021-05-17, 2018-01-02
  Premium, Income, CAP_LT30K, 2021-05-17, 2021-06-14
asset row (asset_information.csv):
  assetName, assetShortName, assetCategory, assetSubCategory, marketID, sector, industry, timestamp
  MYTILINAIOS SA, ΜΥΤΙΛ, Stock, Large Cap, XATH, Industrials, Conglomerates, 2018-01-02
window: training side of an assumed cut T = 2021-05-29 (test window 2021-05-29 → 2021-11-29); the official split is pending
```
customerID is already a pseudonymous hash in the public release; it is masked here as well. The customer made 113 buys of 48 distinct assets (2018-01-02 → 2022-11-11) and had bought this asset once before this date.

### Field status and leakage check (prediction at the transaction date, 2021-05-12)

| Field | Status | Timestamp / when known | Leakage check |
|---|---|---|---|
| customerID, ISIN, transactionID, marketID | observed | 2021-05-12 | identifiers only |
| transactionType, totalValue, units, channel | observed | 2021-05-12 | label-side fields: usable only for events before the cut |
| customerType = Premium | observed | profile rows 2018-01-02 and 2021-06-14 (same value) | static in the release; true as-of date unknown |
| riskLevel = Income | observed (not Predicted_*) | from questionnaire dated **2021-05-17** | **future leakage at 2021-05-12**: the questionnaire is 5 days after the buy, yet the value is also printed on the 2018-01-02 profile row |
| investmentCapacity = CAP_LT30K | observed | same questionnaire, 2021-05-17 | same leakage as riskLevel |
| lastQuestionnaireDate | observed | 2021-05-17 | is the check itself: use risk/capacity only if lastQuestionnaireDate ≤ cut |
| profile `timestamp` | observed | 2018-01-02 / 2021-06-14 | not an as-of date for the profile values |
| assetName, assetCategory, assetSubCategory, sector, industry | observed (static metadata) | asset row 2018-01-02 | low risk; sector/industry may be a later classification |
| close prices | observed | daily | only prices ≤ cut may be inputs |

Proposed rules: inputs use only rows with timestamp ≤ cut; riskLevel/investmentCapacity are used only when lastQuestionnaireDate ≤ cut, otherwise treated as missing; Predicted_* values are treated as estimated, not observed; shared-ID duplicates are counted once for popularity statistics.

## 4. What this means for the research question

- The repeat-purchase rule (pending) decides which baseline is strongest; the question should state it as an assumption.
- Purchases are Greek large-cap stocks, not funds: an LLM may "know" these companies from pre-training — both a possible advantage and a contamination risk.
- Very short histories (median one asset) make "sparse-history users" a natural, well-defined population.
- Profile fields are a leakage trap; any method that uses them needs the questionnaire-date rule, and the baseline must get the same fields.
