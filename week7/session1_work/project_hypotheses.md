# FAR-Trans project — hypothesis backlog

Shared task: Top-10 assets per customer, nDCG@10 against future purchases. Working assumption: only assets new to the customer count (official protocol pending).
Status values: planned · pre-check · running · done (with result, including null/negative).

---

## H1. News context for the LLM reranker (RAG, Session 3)

**Status:** planned · **Track:** Ranker + RAG · **Owner:** TBD · **Recorded:** 2026-11-13

**Idea.** Customers tend to buy companies that are in the news; positive coverage may add to that. Retrieved news about the candidate companies, given to the LLM reranker as context, may help it pick the assets customers buy next.

**Grounding.**
- Barber & Odean (2008), *Review of Financial Studies* 21(2): individual investors are net buyers of attention-grabbing stocks — stocks in the news, with abnormal trading volume or extreme one-day returns. The effect is about attention, not necessarily positive tone.
- Tetlock (2007), *Journal of Finance* 62(3): media pessimism predicts downward price pressure — tone carries information.

**Testable form.**
> We test whether adding news headlines about the candidate companies, published in the 30 days before the cut, to the LLM reranker's prompt improves nDCG@10 over the same reranker without news, on the same candidates, for FAR-Trans customers with purchase history before the cut.

- H1a (attention): headline **count** helps.
- H1b (tone): **positive tone** adds information beyond the count.

| Element | Definition |
|---|---|
| Intervention | LLM reranker prompt + retrieved pre-cut headlines per candidate company |
| Control | the same reranker, same candidates, same history, no news |
| Same-information classical control | popularity candidates re-sorted by news mention count (H1a) and by a lexicon tone score, e.g. Loughran–McDonald (H1b) — separates the value of the LLM from the value of the news |
| Outcome | nDCG@10 (primary); runtime and API cost per customer |
| Population | evaluated customers with history before the cut; stocks get news, funds and bonds are "no news" (missing) |
| Ablations | news window 7 / 30 / 90 days; count-only vs headlines; real names vs anonymised companies |

**Data needed.**
- A news source with publication timestamps covering Greek listed companies, 2018–2022 (candidate: GDELT — mentions and tone; coverage of Greek companies unverified; much coverage is in Greek).
- A mapping company name ↔ ISIN, built from `asset_information.assetName` / `assetShortName`.

**Leakage and contamination rules.**
- Only news published strictly before the cut T.
- The LLM judges only the headlines we supply, never news "from memory"; the anonymised-names ablation measures how much pre-training knowledge leaks in.

**Go / no-go pre-check (before investing in H1).**
1. Coverage: the news source mentions the companies behind most stock purchases (e.g. the top-50 assets, ~80% of buys) in most months.
2. Cheap in-data signal first — H0 below.

**Risks and ethics.** Positive news is not a good investment; nDCG@10 measures purchase prediction, not suitability or returns. Recommending "hyped" stocks raises a suitability question for the advisor (MiFID) — state it in the threats section. No profit claims.

---

## H0. In-data attention proxies (no external data; pre-check for H1)

**Status:** planned · **Track:** classical feature / control · **Recorded:** 2026-11-13

**Idea.** The attention proxies of Barber & Odean that already exist in FAR-Trans: recent abnormal buying (transactions in the last weeks vs the asset's usual level) and extreme recent price moves (`close_prices`).

> We test whether re-sorting popularity candidates by recent abnormal buying or extreme recent returns improves nDCG@10 over all-time popularity for the common evaluation users.

**Hint so far (assumed protocol, 74 cuts, not a Session 1 requirement):** popularity over the last 90 days gave nDCG@10 0.281 vs 0.261 for all-time popularity (`baseline_popularity_by_cut.csv`) — consistent with an attention effect, not a test of it.

---

## References
- B. M. Barber and T. Odean, "All that glitters: The effect of attention and news on the buying behavior of individual and institutional investors," *Rev. Financial Studies*, vol. 21, no. 2, pp. 785–818, 2008.
- P. C. Tetlock, "Giving content to investor sentiment: The role of media in the stock market," *J. Finance*, vol. 62, no. 3, pp. 1139–1168, 2007.
