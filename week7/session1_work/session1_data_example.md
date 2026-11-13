# Session 1 §4 — One actual data example (minimal fields)

Selection: one uniformly random row of `transactions.csv` (pandas `sample(1, random_state=2026)`, script `pick_record.py`).
Assumed split (official one pending): cut T = 2022-05-30; test window (2022-05-30, 2022-11-30]; rule of the day — already-bought assets are neither recommended nor counted.

```
source: FAR-Trans public release (University of Glasgow, researchdata.gla.ac.uk/1658), transactions.csv, line 75,804; assetName joined from asset_information.csv by ISIN
record id: transactionID 10559805
fields: customerID, ISIN, transactionType, timestamp | assetName
values: 133BB7…(masked), GRS003003035, Buy, 2022-03-21 | NATIONAL BANK OF GREECE
window: history (input) side of the assumed cut T = 2022-05-30
```

customerID is a pseudonymous hash in the public release and is masked here as well.

| Field | Role in the baseline | Status | Timestamp | Leakage check |
|---|---|---|---|---|
| customerID | user key | observed | static identifier | identifier only, carries no future information |
| ISIN | item key | observed | static identifier | identifier only |
| transactionType = Buy | positive interaction (Sell rows are not positives) | observed | 2022-03-21 | known at event time; usable as input only because 2022-03-21 ≤ T |
| timestamp | event date | observed | 2022-03-21 | decides the side of the cut: a row dated after T is a label and never an input |
| assetName | display text; later the LLM's item text | observed (static metadata) | asset row dated 2018-01-02 | no future value in the field itself; an LLM may know the company's later history (contamination risk, not data leakage) |

Not used: transactionID (record id only); totalValue, units, channel (not needed for popularity); marketID (needed only to tell apart the 25 ISINs listed on two markets); all customer-profile fields (back-filled after the fact, see step1_data_notes.md).

Context of this record: the customer has 7 transactions (3 buys) between 2018-01-02 and 2022-11-11 and had bought this asset 3 times before 2022-03-21, so the row is a repeat purchase. Under the "new assets only" rule it serves only as history, and this asset is excluded from the customer's Top-10. GRS003003035 is the most-bought asset before T, so a popularity list would rank it first for every customer who does not already hold it.
