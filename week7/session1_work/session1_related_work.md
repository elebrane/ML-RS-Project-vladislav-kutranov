# Session 1 §2 Related Work — source analysis (draft)

Status: candidate pool of 10 studies, checked by Claude against the paper text (via arXiv/ar5iv/publisher pages) on 2026-11-13.
The assignment allows citing only studies **you opened**: open each one before it goes into the report.
The report table needs 3–5 rows; section 4 suggests which ones, depending on the question.
FAR-Trans numbers below come from our own analysis of the raw files (step1_data_notes.md, fartrans_overview.html).

## 0. Report-ready table (assignment §3 template, Ranker-track default)

| Study (authors, year, venue) | Method | Data and evaluation | Limitations and connection to our question |
|---|---|---|---|
| Sanz-Cruzado, Droukas, McCreadie, 2024, Fin-RecSys @ IJCAI [1] | Releases FAR-Trans; benchmarks price-based regressors, Popularity, ARM, UB-kNN, MF, LightGCN and LightGBM hybrids | FAR-Trans (29,090 customers, 806 assets, 2018–2022); 61 time cuts, 6-month windows, only newly acquired assets relevant; nDCG@10 and ROI@10. LightGCN 0.340, Popularity 0.271 | No LLM methods; tuned for ROI, not nDCG. Gives our data, protocol template and classical baselines; Popularity is the bar to clear |
| Spadea, Seneviratne, 2025, ICAIF '25 [3] | Qwen3 (0.6–8B) prompted with personal and market knowledge graphs generates 3 assets; LoRA + KTO fine-tuning | FAR-Trans; train to 2021-11, test 2021-12 → 2022-11; relevant = bought within 180 days (and profitable for Comb@3); Hits@3 against the baselines of [1] | Nearest prior work. Top-3 hits, not nDCG@10; generation after fine-tuning, not reranking a fixed list; baseline scores only in a figure. Leaves open zero-shot reranking with a same-information control |
| Hou et al., 2024, ECIR [7] | Zero-shot LLM ranks a candidate list given the user's history as text; candidates random or from classical retrievers | MovieLens-1M, Amazon Games; leave-one-out; NDCG@10 of the best prompt 45.8 vs trained SASRec 78.9 on ML-1M | Shows position and popularity bias and weak use of history order; mitigations: recency prompt, candidate shuffling. Method template for our reranker; popularity bias is critical on FAR-Trans |
| Li, Jullien, Ariannezhad, de Rijke, 2023, ACM TOIS [6] | Re-evaluates next-basket methods, splitting every metric into repeat and explore (new-to-user) items | TaFeng, Dunnhumby, Instacart; Recall/NDCG/PHR@K; repeat items give >97% of the best methods' performance on two datasets | Grocery, not finance. Explains why our "new assets only" rule (60.5% of FAR-Trans buys are re-buys) turns the task into pure exploration, the hard part |
| Ferrari Dacrema, Cremonesi, Jannach, 2019, ACM RecSys [10] | Reproduces 18 neural top-N papers against tuned simple baselines (TopPop, kNN, graph random walks) | Original datasets and protocols; 7 of 18 reproducible, 6 of these 7 often beaten by simple heuristics | Not finance or LLMs. Justifies a tuned Popularity + graph (PPR-like) baseline that receives the same information |

**Nearest prior work.** The closest study is [3]: it is the only LLM work we found on FAR-Trans. It answered whether fine-tuned small LLMs with knowledge-graph context recommend assets that customers later buy (Pref@3) and that are also profitable (Comb@3), compared with the baselines of [1]. It left open, under our condition, a top-10 ranking scored with nDCG@10, an LLM that only reorders a fixed candidate list, a classical control given the same information, and an evaluation where only assets new to the customer count. **Gap:** we did not find a study reporting nDCG@10 for LLM reranking of a fixed candidate set against popularity and graph baselines on FAR-Trans when only assets new to the customer count as relevant. *(Verify how [3] treats already-held assets before finalising this sentence.)*

## 1. Candidate pool

### A. Domain and data (financial asset recommendation)

| Study (authors, year, venue) | Method | Data and evaluation | Limitations and connection to our question |
|---|---|---|---|
| [1] Sanz-Cruzado, Droukas, McCreadie, 2024, Fin-RecSys @ IJCAI | Releases FAR-Trans; benchmarks 11 recommenders: price-based regressors (RF, LR, LightGBM), transaction-based (Popularity, ARM, UB-kNN, MF, LightGCN) and LightGBM hybrids | 29,090 customers of a European bank, 806 assets, 2018–2022; 61 bi-weekly time cuts, 6-month test window; relevant = asset newly acquired in the window (already-held assets removed); nDCG@10 and monthly ROI@10. LightGCN nDCG@10 0.340, Popularity 0.271, MF 0.178 | Hyper-parameters tuned for ROI, not nDCG; no LLM methods. Defines our data, protocol template and the classical baselines; Popularity is the bar any LLM method must clear |
| [2] McCreadie et al., 2022, ch. 10 in *Big Data and AI in Digital Finance* (Springer) | Robo-advisor pipeline; MF, LightGCN, customer-profile similarity, profitability regressors, CF+feature hybrids | Private National Bank of Greece data 2018–2021 (Greek stocks); temporal splits, 9-month test; nDCG@1/5/10 and annualised return | Earlier version of the same data and group; LightGCN > MF for predicting future investments. Confirms the data source and that graph CF is the strongest classical family here |
| [3] Spadea, Seneviratne, 2025, ICAIF '25 (FLARKO) | Qwen3 0.6B–8B prompted with a personal knowledge graph (transaction history) and a market graph (10-week price summaries); generates 3 ISINs; LoRA + KTO fine-tuning; federated variant | FAR-Trans; train 2018-01 → 2021-11, test 2021-12 → 2022-11; label = bought within 180 days (+ profitable for the joint metric); Hits@3 (Pref@3, Prof@3, Comb@3); baselines from [1] | **Nearest prior work.** LLM on the same data, but: top-3 hits instead of top-10 nDCG; fine-tuned generation, not reranking of a fixed candidate list; baseline numbers only in a figure; label rule for already-held assets not stated in the summary we read — check in the paper |

### B. Predicting the next purchase / next item

| Study | Method | Data and evaluation | Limitations and connection |
|---|---|---|---|
| [4] Kang, McAuley, 2018, IEEE ICDM (SASRec) | Causal self-attention over the user's item sequence predicts the next item | Amazon Beauty/Games, Steam, MovieLens-1M; leave-one-out, 100 sampled negatives; Hit@10, NDCG@10; beats GRU4Rec, Caser, TransRec (Games NDCG@10 0.536 vs 0.476) | Sampled-negative evaluation is optimistic; needs longer sequences — on sparse data it attends mainly to the last items. In FAR-Trans 51% of customers bought one distinct asset, so sequence signal is thin; useful as an optional trained baseline |
| [5] Hu, He, Gao, Zhang, 2020, ACM SIGIR (TIFU-KNN) | User vector = personalised item frequency with time decay; blended with k nearest neighbours' vectors | Dunnhumby, ValuedShopper, Instacart, TaFeng; Recall, NDCG; often beats RNN next-basket models | Shows that frequency + neighbours capture most of next-basket signal and RNNs struggle to learn it. Its neighbour part is a cheap, explainable classical baseline for "new" purchases |
| [6] Li, Jullien, Ariannezhad, de Rijke, 2023, ACM TOIS (NBR reality check) | Re-evaluates 10 next-basket methods; splits every metric into **repeat** items (bought before) and **explore** items (new to the user) | TaFeng, Dunnhumby, Instacart; Recall/NDCG/PHR@K plus repeat/explore ratios; repeat items give >97% of the strongest methods' performance on two datasets; personal top-frequency baselines are competitive | Directly explains our protocol choice: FAR-Trans has 60.5% re-buys, and the "new assets only" rule turns the task into pure **exploration** — the hard part. Justifies reporting repeat and new separately if the course counts repeats |

### C. LLMs as rankers or tuned recommenders

| Study | Method | Data and evaluation | Limitations and connection |
|---|---|---|---|
| [7] Hou et al., 2024, ECIR (LLMRank) | Zero-shot LLM (ChatGPT, GPT-4, open models) ranks candidates given the user's history as text; candidates from random sampling or retrievers (Pop, BPRMF, SASRec…) | MovieLens-1M, Amazon Games; leave-one-out; NDCG@K; best zero-shot prompt NDCG@10 45.8 (ML-1M) vs trained SASRec 78.9; with retrieved candidates LLM beats trained Pop/BPRMF but not SASRec | Documents order insensitivity, **position bias and popularity bias**; mitigations: recency-focused prompt, in-context examples, bootstrapping (shuffle candidates). The method template for a Ranker-track question; popularity bias matters because FAR-Trans is popularity-dominated |
| [8] Yue et al., 2023, arXiv:2311.02089 (LlamaRec) | Two-stage: small sequential retriever (LRURec) → Llama-2-7B QLoRA ranker scoring 20 candidates via a verbalizer (one forward pass) | ML-100k, Amazon Beauty, Games; NDCG/Recall/MRR@5,10; gains of ~4–12% over the retriever; inference < 1 s vs 56 s for a generation baseline | Ranker gains are capped by retriever recall — so measure recall@K of the candidate stage first. The verbalizer trick addresses runtime, which breaks ties in our benchmark |
| [9] Bao et al., 2023, ACM RecSys (TALLRec) | LLaMA-7B, LoRA; Alpaca tuning then recommendation tuning on Yes/No "will the user like the item" prompts | MovieLens100K, BookCrossing; AUC; with 64 samples AUC 67.5 (movies) where in-context LLMs ≈ 0.5 | Binary point-wise task, not top-K ranking. Relevant only if the team takes the optional fine-tuning step (Session 4) |

### D. Evaluation rigour

| Study | Method | Data and evaluation | Limitations and connection |
|---|---|---|---|
| [10] Ferrari Dacrema, Cremonesi, Jannach, 2019, ACM RecSys | Reproduces 18 neural top-N papers (7 reproducible) against tuned simple baselines: TopPop, UserKNN, ItemKNN, P3α, RP3β (random-walk graph methods) | Original datasets and protocols of each paper | 6 of 7 neural methods were often beaten by tuned simple heuristics. Supports a tuned Popularity + graph random-walk (PPR-like) baseline with the same information instead of a weak MF |

## 2. What the next-purchase literature means for FAR-Trans

1. **Our target is "what new asset will the customer buy in the next six months"** — closer to next-basket exploration [6] than to next-click prediction [4]. With repeats excluded, frequency-of-own-history methods [5], [6] lose their main signal; popularity and collaborative/graph neighbours remain.
2. **Sequence models are a weak fit here** [4]: median one distinct asset per customer, and many buys are dated at the dataset start (2018-01-02 opening snapshot).
3. **An LLM ranker inherits the retriever's ceiling** [8] and tends to favour popular items [7]. Expected outcome to test: the LLM may reorder candidates well but has little room above Popularity unless it uses information the baseline lacks — which then requires a same-information control.
4. **Baselines must be tuned and strong** [10]; [1] and [2] show graph CF is the strongest classical family on this data.

## 3. Nearest prior work and gap (draft — finalise after the question is fixed)

- **Nearest prior work: [3] FLARKO.** It answered whether fine-tuned small LLMs with knowledge-graph context can recommend FAR-Trans assets that customers later buy (Pref@3) and that are also profitable (Comb@3), against the baselines of [1].
- **Left open under our condition:** a top-10 ranking evaluated with nDCG@10; a fixed candidate list where only the order changes (reranking, not generation); a same-information classical control; separate treatment of assets new to the customer; a check of pre-training knowledge about the companies (names shown vs hidden).
- **Gap statement (to verify while reading [3]):** "We did not find a study reporting nDCG@10 for LLM reranking of a fixed candidate set against popularity and graph baselines on FAR-Trans when only assets new to the customer count as relevant."

## 4. Which 3–5 to put in the report table

For a Ranker-track question (LLM reranking vs Popularity/PPR, new assets only, nDCG@10):
[1] FAR-Trans (data, protocol, baselines) · [3] FLARKO (nearest prior work) · [7] LLMRank (method and known biases) · [6] NBR reality check (repeat vs explore) · [10] or [8] (baseline rigour or two-stage design).
For an Encoder-track question, swap [7] for [8] or [9] and keep [1], [3], [6].

## 5. Research importance and business value (draft, depends on the final question)

- **Research importance:** evidence on whether LLM reranking adds anything over popularity and graph baselines in a domain where purchases are concentrated, histories are very short and only exploratory (new-to-customer) purchases count; a negative result is informative because [6] and [10] suggest simple methods are hard to beat.
- **Business value:** the stakeholder is the bank's investment advisor or the product owner of an advisory tool ([2] describes recommendations shown to advisors). The decision that changes: whether to pay the latency and API cost of an LLM stage in the advisor's shortlist, or keep a popularity/graph list. No claim about returns or customer acceptance.

## 6. References (IEEE)

[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An investment dataset for financial asset recommendation," in *Proc. IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys)*, Jeju, South Korea, 2024. [Online]. Available: https://arxiv.org/abs/2407.08692

[2] R. McCreadie *et al.*, "Next-generation personalized investment recommendations," in *Big Data and Artificial Intelligence in Digital Finance*. Cham, Switzerland: Springer, 2022, pp. 171–198, doi: 10.1007/978-3-030-94590-9_10.

[3] F. Spadea and O. Seneviratne, "Aligning language models with investor and market behavior for financial recommendations," in *Proc. ACM Int. Conf. AI in Finance (ICAIF '25)*, Singapore, 2025. [Online]. Available: https://arxiv.org/abs/2510.15993

[4] W.-C. Kang and J. McAuley, "Self-attentive sequential recommendation," in *Proc. IEEE Int. Conf. Data Mining (ICDM)*, 2018. [Online]. Available: https://arxiv.org/abs/1808.09781

[5] H. Hu, X. He, J. Gao, and Z.-L. Zhang, "Modeling personalized item frequency information for next-basket recommendation," in *Proc. 43rd Int. ACM SIGIR Conf. Research and Development in Information Retrieval*, 2020. [Online]. Available: https://arxiv.org/abs/2006.00556

[6] M. Li, S. Jullien, M. Ariannezhad, and M. de Rijke, "A next basket recommendation reality check," *ACM Trans. Inf. Syst.*, 2023. [Online]. Available: https://arxiv.org/abs/2109.14233

[7] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao, "Large language models are zero-shot rankers for recommender systems," in *Proc. 46th European Conf. Information Retrieval (ECIR)*, 2024. [Online]. Available: https://arxiv.org/abs/2305.08845

[8] Z. Yue *et al.*, "LlamaRec: Two-stage recommendation using large language models for ranking," arXiv:2311.02089, 2023.

[9] K. Bao, J. Zhang, Y. Zhang, W. Wang, F. Feng, and X. He, "TALLRec: An effective and efficient tuning framework to align large language model with recommendation," in *Proc. 17th ACM Conf. Recommender Systems (RecSys '23)*, 2023, doi: 10.1145/3604915.3608857.

[10] M. Ferrari Dacrema, P. Cremonesi, and D. Jannach, "Are we really making much progress? A worrying analysis of recent neural recommendation approaches," in *Proc. 13th ACM Conf. Recommender Systems (RecSys '19)*, 2019. [Online]. Available: https://arxiv.org/abs/1907.06902
