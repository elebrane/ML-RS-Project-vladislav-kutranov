# Week 8 — HW2 "Duet": zero-shot LLM encoder + reranker on FAR-Trans (2×2 ablation)

Team **Sparse Signals**. Experiments in this folder, in order: **Chronicler** (E1, the LLM writes the client
profile itself; `self_structured_ranker.py`), **Scout** (E2, the LLM fills the new-asset slots;
`explore_slots.py`), **Compass** (no-LLM checks on Mar–May 2021; `analysis/`, `hybrid_headroom.py`) and
**Duet** (the HW2 method below; `hybrid.py`).

**Research question.** Does a zero-shot two-stage LLM pipeline (Qwen3-4B-Instruct-2507) improve
nDCG@10 over the provided Latest buy baseline for the 100 warm FAR-Trans users at the 1 June 2021
cutoff, where (1) the encoder reads a code-computed table of behavioural facts for each asset the
client traded — repurchase rhythm, current position and the result of the last sale, and bank-wide
attention (buyer rank, 7-day buying spike and 30-day price change, at the client's first buy and at
the cutoff) — and returns cited patterns plus one repurchase status per asset (core, exited at a
gain, exited at a loss, new, lapsed, one-off); and (2) the reranker orders a fixed pool of the
client's own assets and the most-bought new assets of the client's main category using that
evidence, keeping own assets first unless the evidence points elsewhere?

**Answer on the course test: no.** The full hybrid scored 0.0195 vs 0.0277 for Latest buy; the
encoder alone with a fixed cascade scored 0.0295, the best of all eight methods. The PDF report
is submitted through the course channel; `report/build_hw2_report.py` rebuilds it from the run outputs.

| Method (100 users, 11 with purchases) | nDCG@10 all | nDCG@10 nonempty | Failed cases |
|---|---:|---:|---:|
| Popularity | 0.0286 | 0.2599 | 0 |
| CBF | 0.0201 | 0.1827 | 0 |
| Latest buy | 0.0277 | 0.2520 | 0 |
| Latest buy or sell | 0.0271 | 0.2464 | 0 |
| 1. Rule, no LLM (`hyb_c1_rule`) | 0.0262 | 0.2379 | 0 |
| 2. LLM encoder only (`hyb_c2_encoder`) | **0.0295** | **0.2683** | 2 |
| 3. LLM reranker only (`hyb_c3_reranker`) | 0.0113 | 0.1030 | 0 |
| 4. Encoder + reranker (`hyb_c4_both`) | 0.0195 | 0.1776 | 4 |

`protocol_hash` 50efd4aeb415… equals the starter baseline run.

## Files

| Path | What it is | Origin |
|---|---|---|
| `far-trans/` | course starter, provided baselines, HW2 text | course repo dryjins/RecSys-LLMs @ f685086, unchanged |
| `far-trans-llm/data_adapter.py`, `runtime.py`, `ranker.py`, `encoder.py`, tests | course LLM baseline modules | same, unchanged |
| `far-trans-llm/hybrid.py` | facts, pool, the four ablation cells, validation, fallbacks, metrics | own |
| `far-trans-llm/test_hybrid.py` | parsers, status rule, fake-model run, leakage checks | own |
| `far-trans-llm/prompts/` | `encoder_v1–v3.txt` (v3 used on the test), `reranker_v1.txt` | own |
| `far-trans-llm/configs/hybrid_*.json` | smoke (3 May buyers) and test (June) configs | own |
| `far-trans-llm/analysis/status_order.py` | no-LLM check of the status precedence on Mar–May 2021 | own |
| `far-trans-llm/analysis/prelim_validation.py`, `fusion_validation.py` | no-LLM checks on Mar–May 2021 (report Table 5) | own |
| `far-trans-llm/analysis/method_overlap.py` | overlap of the methods' top-10 lists and hindsight best-of ceiling | own |
| `far-trans-llm/self_structured_ranker.py`, `configs/hw2_main.json`, `test_self_structured.py` | Experiment 1 (LLM ranks 50 candidates; self-built vs code-built profile), June test cutoff | own |
| `far-trans-llm/explore_slots.py`, `configs/explore_q4b_val.json`, `test_explore_slots.py`, `analysis/exp2_partial_scores.py` | Experiment 2 (LLM fills new-asset slots), May validation; scored on the 77 completed users | own |
| `far-trans-llm/hybrid_v1.py`, `hybrid_headroom.py` | first hybrid design (four-source pool) and its no-LLM pool analysis | own |
| `report/data/*.json` | outputs of the `analysis/` scripts used in the report | own |
| `run_hybrid_smoke3_then_test.command` | macOS runner used for the test (smoke v3, then June) | own |
| `report/` | report builder (`build_hw2_report.py` → HTML → `render.js` → PDF) | own |
| `Data FAR-Trans` | symlink to `../week7/Data FAR-Trans` | dataset |

Run outputs (`far-trans-llm/outputs/`), the virtual environment and logs stay local and are not
committed; the AI-session log is submitted separately and never uploaded here.

## Run

```bash
cd week8/far-trans-llm
python3.13 -m venv .venv && source .venv/bin/activate
pip install torch==2.11.0 transformers==4.57.6      # model inference only
python -m unittest test_hybrid                       # 8 tests, no model needed
python hybrid.py --config configs/hybrid_test_june.json --no-llm   # baselines + cell 1 in seconds
python hybrid.py --config configs/hybrid_test_june.json            # all four cells, ~1 h 45 min on the Mac used (MPS)
python ../report/build_hw2_report.py outputs ../report/kutranov_vladislav_hw2_report.html
```

Model: `Qwen/Qwen3-4B-Instruct-2507`, revision `cdbee75f17c01a7cc42f958dc650907174af0554`,
greedy decoding, thinking off, context 8192, max new tokens 1000 (encoder) / 100 (reranker).
Every prompt and answer is cached under `outputs/<run>/generation_cache`, so a rerun with the same
config reproduces the predictions without new model calls.

Teaching experiment on historical data; a predicted purchase is not investment advice.
