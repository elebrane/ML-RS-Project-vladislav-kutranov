# Week 9 — HW3: does market news help an LLM ranker fill the new-asset slots? (FAR-Trans)

Team **Sparse Signals**, individual part (Vladislav Kutranov). Continues the HW2 "Duet" question: the LLM is
used only as a **Ranker**, and only where transaction data add nothing over popularity — the order of
never-bought assets in the slots below the client's own assets. The evidence is the information field of
the Greek market: a daily per-asset summary of GDELT news (articles, sources, tone, attention vs the
asset's usual level) plus the latest headlines, all first seen before the cutoff, company names masked.

**RQ.** For warm FAR-Trans clients whose main asset category is Stock and who have free new-asset slots
(669 of 1,498 development clients at the 1 May 2021 cutoff), does retrieving time-safe Greek-press news evidence
for each candidate stock — article counts over 30 and 7 days, attention relative to the stock's usual level,
average tone and the two latest masked headlines, all first seen by GDELT before the cutoff — change nDCG@10 of a
zero-shot Qwen3-4B-Instruct ranker relative to the same ranker without news, when the candidates, their initial
order and the client's own assets are held fixed?

Base: the HW2 "Duet" rule cascade (cell 1, own assets first) — the Duet v2 encoder gave the same 10-item set for
every client, so it is not used; the reranker is narrowed to the new-asset block and given news.
Run scope for the deadline: LLM only for Stock-main clients, news for the 30 most frequent candidate stocks,
one ablation (A1, news removed); A2, A3 and the subset checks are implemented but not run.

## Design

Fixed in every arm: users, cutoff, own-asset block (HW2 rule order), number of new slots
(10 − own assets), the 20-asset new block (never-traded assets of the client's main category by bank
buyers in the last 30 days), its initial order, the shown order and IDs of candidates, model, decoding.

| Arm | What the candidate rows contain | Role |
|---|---|---|
| `c0_popularity` | — (no LLM) | matched control = HW2 rule cascade (cell 1) |
| `cnews_attention` | — (code sorts by news attention vs usual level) | code control, pre-declared |
| `cnews_volume`, `cnews_tone` | — (code sorts by articles / tone) | exploratory code controls |
| `llm_nonews` | base row: category, sector, bank buyers, rank, 30-day price change | **A1** (news removed) |
| `llm_numbers` | + news numbers, no headlines | **A2** (text removed) |
| `llm_full` | + news numbers + latest 2 headlines (names masked) | **M** (main method) |
| `llm_shuffled` | `llm_full` news lines moved to other stocks of the same block | **A3** placebo (same tokens) |
| `llm_full_order2` (subset 200) | `llm_full` with a second shown order | position sensitivity |
| `llm_full_names` (subset 200) | `llm_full` with names shown, headlines unmasked | name-recall (look-ahead) check |

Contrasts fixed before any LLM run (paired bootstrap, 10,000 resamples, seed 0; sign test): M−A1, M−A2,
M−A3, M−C0, A1−C0, A2−C-news, M−C-news, C-news−C0, M−Latest buy.

## Cohorts (frozen in `configs/cohorts.json`, `outputs/<cohort>/protocol.json`)

| Cohort | Cutoff | Rule (past data only) | Users | Role |
|---|---|---|---:|---|
| D0_june | 2021-06-01 | starter's 100 warm users | 100 | HW2 teaching set; `protocol_hash` 50efd4aeb415 = starter |
| D1_may | 2021-05-01 | first 3,000 warm, ≥1 Buy in 90 days before cutoff | 1,498 | development evaluation |
| TUNE_apr | 2021-04-01 | same rule | 1,491 | prompt smoke test, code go/no-go only |

No-LLM results already computed (`outputs/<cohort>/nollm_nonews`): on D0 the perfect block order equals
C0 (0.0262), so **no arm can change D0**; D1 has a ceiling of +0.0059 (all users) / +0.0218 (408 buyers) /
+0.0555 (160 buyers of a never-traded asset) over C0.

## Files

| File | What |
|---|---|
| `common.py` | cohorts, frozen protocol (starter hash + `hw3_protocol_hash`), block, list assembly |
| `freeze.py` | writes `outputs/<cohort>/protocol.json`, `blocks.json`, `news/candidate_stocks.json` |
| `make_aliases.py` → `news/aliases.json` | 55 candidate stocks: GDELT query phrases, Greek/Latin title patterns |
| `collect_gdelt.py` | GDELT DOC 2.0 collector (stdlib; resumable; raw responses kept in `news/raw/`) |
| `build_infofield.py` | `news/daily_infofield.csv`, `news/headlines.jsonl`, `news/coverage.json`, linking sample |
| `news.py` | matching, masking, time boundary, news cards |
| `block_ranker.py` | all arms, parsing/repair/fallback, metrics, contrasts, subset diagnostics |
| `prompts/block_ranker_v1.txt` | the one prompt shared by all LLM arms |
| `test_news_rag.py` | 6 tests: boundary, masking, parsing, arm isolation, fake-model run |
| `summarize.py` → `outputs/summary.md` | all tables for the report |

Code from `../../week8/far-trans-llm` (starter loader, `hybrid.py` facts and Market, model wrapper, cache)
is imported unchanged.

## Run (on the Mac)

1. `run_news_collect.command` — GDELT is reachable only from the Mac; 1–3 h; `STATUS: NEWS_DONE`.
2. `run_hw3_llm.command` — tests, code go/no-go, smoke (40 users), D1 M+A1, D0 all arms, D1 all arms,
   `outputs/summary.md`; 8–11 h, safe to stop and restart (model answers are cached by prompt).

Model `Qwen/Qwen3-4B-Instruct-2507` @ `cdbee75f17c01a7cc42f958dc650907174af0554`, greedy, thinking off,
float16 on MPS, context 8192, 90 new tokens. Preparation and generation times are reported separately in
each `hw3_summary.json`. Provided LLM comparison on D0: course LLMRank 0.0201 / 0.1827 (HW2_RESULTS.md, same
protocol hash).

## Time safety

Transactions, prices and profiles strictly before the cutoff (HW2 `Market.assert_no_future`). News counts
use UTC days up to cutoff − 2; a headline is used only if GDELT first saw it before 21:00 UTC on cutoff − 1
(Athens midnight); both are asserted and unit-tested. GDELT is asked for nothing after that moment.

## Manual checks still to do (by me)

- Label `news/linking_check_sample.csv` (is the headline about this company? 1/0) → linking precision.
- Hand-compute nDCG@10 for one D1 user from `predictions.csv` and `transactions.csv`.
- Open one headline URL from a prompt and confirm its date is before the cutoff.

Development results only, not official benchmark scores. Teaching experiment; not investment advice.
