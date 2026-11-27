"""Build the HW3 paper-style report (HTML) from the run outputs; re-run after new runs finish."""
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NR = HERE.parent
OUT = NR / 'outputs'


def load(p):
    p = OUT / p
    return json.loads(p.read_text()) if p.exists() else None


a1 = load('D1_may/a1_now/hw3_summary.json')
m1 = load('D1_may/m_subset/hw3_summary.json')
d0 = load('D0_june/night1_main_and_A1/hw3_summary.json') or load('D0_june/nollm_nonews/hw3_summary.json')
gng = load('TUNE_apr/gonogo_code_news/hw3_summary.json')
cov = json.loads((NR / 'news' / 'coverage.json').read_text())
ch_a1 = load('D1_may/a1_now/choice_analysis_llm_nonews.json')
ch_m = load('D1_may/m_subset/choice_analysis_llm_full.json')
ch_a1s = load('D1_may/m_subset/choice_analysis_llm_nonews.json')
main = a1
T = main['table']
NAMES = [('popularity', 'Popularity'), ('cbf', 'CBF'), ('latest_buy', 'Latest buy'),
         ('latest_buy_or_sell', 'Latest buy or sell'), ('c0_popularity', 'C0: initial order (matched control)'),
         ('cnews_attention', 'C-news: attention, code'), ('cnews_volume', 'C-news: volume, code'),
         ('cnews_tone', 'C-news: tone, code'), ('llm_nonews', 'A1: LLM ranker, no news'),
         ('llm_full', '<b>M: LLM ranker + news</b>'), ('oracle_block', 'Ceiling: perfect block order')]
f4 = lambda x: '—' if x is None else f'{x:.4f}'


def table(tab):
    rows = ['<tr><th>Method</th><th>nDCG@10 all (n)</th><th>buyers (n)</th><th>bought a new asset (n)</th></tr>']
    for k, nm in NAMES:
        if k in tab:
            r = tab[k]
            rows.append(f"<tr><td>{nm}</td><td>{f4(r['all'])} ({r['n_all']})</td><td>{f4(r['buyers'])} ({r['n_buyers']})</td>"
                        f"<td>{f4(r['bought_new'])} ({r['n_bought_new']})</td></tr>")
    return '<table class="small">' + ''.join(rows) + '</table>'


def contrasts(s):
    rows = ['<tr><th>Contrast</th><th>Δ all users [95% CI]</th><th>better / worse</th><th>sign p</th><th>Δ buyers [95% CI]</th></tr>']
    for c in s['contrasts']:
        x, y = c['all_users'], c['buyers']
        rows.append(f"<tr><td>{c['a']} − {c['b']}<br><span class='caption'>{c['question']}</span></td>"
                    f"<td>{x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]</td><td>{x['better']} / {x['worse']}</td>"
                    f"<td>{x['sign_p']:.4f}</td><td>{y['diff']:+.4f} [{y['ci95'][0]:+.4f}, {y['ci95'][1]:+.4f}]</td></tr>")
    return '<table class="small">' + ''.join(rows) + '</table>'


def choice(ch_list):
    rows = ['<tr><th>Picks of the LLM vs C0 (same clients, same k)</th>' + ''.join(f"<th>{n}</th>" for n, _ in ch_list) + '</tr>']
    keys = [('mean_overlap_with_C0_topk', 'share of picks that C0 also shows', None),
            ('mean_share_equal_to_first_k_shown', 'share equal to the first k shown (random order)', None),
            ('median_popularity_rank_of_picks', 'median bank-popularity rank, LLM / C0', 'pair'),
            ('share_in_client_past_sectors', 'picks in a sector the client bought before, LLM / C0', 'pair'),
            ('mean_price_change_30d_of_picks', 'mean 30-day price change of picks, LLM / C0', 'pct'),
            ('bought_assets_dropped_from_C0_slots', 'bought assets the LLM removed from C0 slots', None),
            ('bought_assets_added_by_llm', 'bought assets the LLM brought into the slots', None)]
    for k, label, kind in keys:
        cells = []
        for _, ch in ch_list:
            v = ch[k]
            if kind == 'pair':
                cells.append(f"{v['llm']} / {v['c0']}")
            elif kind == 'pct':
                cells.append(f"{100 * v['llm']:+.1f}% / {100 * v['c0']:+.1f}%")
            else:
                cells.append(str(v))
        rows.append(f"<tr><td>{label}</td>" + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
    return '<table class="small">' + ''.join(rows) + '</table>'


ca1 = next(c for c in a1['contrasts'] if c['a'] == 'llm_nonews')['all_users']
st_a1 = a1['meta']['llm_status']['llm_nonews']
gen_a1 = a1['meta']['seconds']['generation_by_arm_incl_cached']['llm_nonews']
covd1 = cov['coverage']['D1_may']
if m1:
    cm = {(c['a'], c['b']): c['all_users'] for c in m1['contrasts']}
    ma = cm.get(('llm_full', 'llm_nonews'))
    mc = cm.get(('llm_full', 'c0_popularity'))
    st_m = m1['meta']['llm_status']['llm_full']
    st_m_calls = st_m.get('ok', 0) + st_m.get('repaired', 0) + st_m.get('fallback', 0)
    mgen = m1['meta']['seconds']['generation_by_arm_incl_cached'].get('llm_full', 0)
    m_text = (f"On the first {m1['meta']['users']} clients of the cohort order ({st_m_calls} LLM calls; the M run was stopped at the deadline), adding news changed nDCG@10 by <b>{ma['diff']:+.4f}</b> [{ma['ci95'][0]:+.4f}, {ma['ci95'][1]:+.4f}] relative to the "
              f"same ranker without news ({ma['better']} clients better, {ma['worse']} worse) and by {mc['diff']:+.4f} "
              f"[{mc['ci95'][0]:+.4f}, {mc['ci95'][1]:+.4f}] relative to the popularity order.")
    m_status = (f"M: {st_m.get('ok', 0)} valid, {st_m.get('repaired', 0)} repaired, {st_m.get('fallback', 0)} fallbacks for {st_m_calls} calls; "
                f"generation {mgen / 60:.0f} min ({mgen / max(st_m_calls, 1):.0f} s per client: news made prompts 2.4–2.9k tokens instead of 1.2k, "
                f"and the Mac slowed to 70–145 s per call, so M covers only the first {m1['meta']['users']} clients of D1.")
else:
    m_text = ('<span class="todo">[PENDING: the M run is in progress; this sentence is filled in automatically when '
              'outputs/D1_may/night1_main_and_A1 exists.]</span>')
    m_status = '<span class="todo">[M status pending]</span>'

subset_html = ''
if m1:
    subset_html = (f"<h3>Main method M on a fixed prefix of D1</h3><p>M was run on the clients in the cohort's own order (a rule on past data only) "
                   f"until the deadline: the first {m1['meta']['users']} clients, of whom {m1['meta']['llm_status']['llm_full'].get('ok', 0)} "
                   f"Stock-main clients got an LLM call. All arms are re-scored on exactly these clients.</p>"
                   + table(m1['table']) + '<div class="caption">Table 4. First ' + str(m1['meta']['users']) + ' clients of D1 (same protocol).</div>'
                   + contrasts(m1) + '<div class="caption">Table 5. Paired differences on the same prefix.</div>')
gng_text = ''
if gng:
    g = next(c for c in gng['contrasts'] if c['a'] == 'cnews_attention')['all_users']
    gt = gng['table']
    gng_text = (f"<p><b>News signal without an LLM (April 2021, tuning month).</b> Ordering the block by news attention in code "
                f"({f4(gt['cnews_attention']['buyers'])} over buyers), by volume ({f4(gt['cnews_volume']['buyers'])}) or by tone "
                f"({f4(gt['cnews_tone']['buyers'])}) does not beat the popularity order ({f4(gt['c0_popularity']['buyers'])}); "
                f"attention vs C0 on all clients: {g['diff']:+.4f} [{g['ci95'][0]:+.4f}, {g['ci95'][1]:+.4f}], {g['better']}/{g['worse']}. "
                f"The collected news numbers alone carry no usable signal for first purchases in this month.</p>")
m_disc = '<span class="todo">[M discussion pending.]</span>'
if m1 and ch_m:
    m_disc = (f"<b>With news (M, first 78 clients):</b> the picks barely move — overlap with C0 {ch_m['mean_overlap_with_C0_topk']} vs "
              f"{ch_a1s['mean_overlap_with_C0_topk']} without news, median popularity rank {ch_m['median_popularity_rank_of_picks']['llm']} vs "
              f"{ch_a1s['median_popularity_rank_of_picks']['llm']}, the same preference for recent price gains; it drops "
              f"{ch_m['bought_assets_dropped_from_C0_slots']} bought assets instead of {ch_a1s['bought_assets_dropped_from_C0_slots']}. "
              "News nudges the model slightly back towards popular stocks but does not change what drives its choice.")
claim = (f"For 669 Stock-main FAR-Trans clients (May 2021), a zero-shot Qwen3-4B ranker <i>without</i> news changed mean "
         f"nDCG@10 over all 1,498 clients by {ca1['diff']:+.4f} [{ca1['ci95'][0]:+.4f}, {ca1['ci95'][1]:+.4f}] relative to the "
         f"popularity order of the same 20 candidates. ") + m_text

css = (HERE.parent.parent.parent / 'week8' / 'report' / 'style.css').read_text()
html = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>HW3 Kutranov</title><style>{css}
.todo{{background:#fef3c7;color:#92400e}} .prelim{{border:1px solid #b45309;background:#fffbeb;padding:4px 8px;margin:4px 0;font-size:8pt}}</style></head><body>
<h1>Does Market News Help a Zero-Shot LLM Fill the New-Asset Slots? A Controlled Test on FAR-Trans</h1>
<div class="meta"><div>Vladislav Kutranov · Team Sparse Signals (Zlata Kranina, Maria Kodzaeva) · LLM4Rec, HSE, Homework 3</div>
<div>Repository: github.com/elebrane/ML-RS-Project-vladislav-kutranov, folder week9/news_rag · experiment commit: <span class="todo">[add after push]</span></div></div>
{'' if m1 else '<div class="prelim"><b>Preliminary version.</b> Results for the main method M are still being computed.</div>'}
<div class="abstract"><b>Abstract.</b> In FAR-Trans most purchases are repeats, and the course baselines show that recency is hard to beat.
Our HW2 hybrid ("Duet") found that an LLM which only restates code-computed facts adds nothing. In HW3 we narrow the LLM to the one place
where transaction data have no signal beyond popularity, the slots for never-bought assets below the client's own assets, and give it
evidence from outside the dataset: a time-safe daily summary of the Greek press coverage of each candidate stock (GDELT), with masked headlines.
Development results (1 May 2021 cutoff, 1,498 warm clients): {claim} All results are development results, not official benchmark scores.</div>

<h2>1. Introduction</h2>
<div class="rq"><b>RQ.</b> For warm FAR-Trans clients whose main asset category is Stock and who have free new-asset slots (669 of 1,498
development clients at the 1 May 2021 cutoff), does retrieving time-safe Greek-press news evidence for each candidate stock — article
counts over 30 and 7 days, attention relative to the stock's usual level, average tone and the two latest masked headlines, all first seen
by GDELT before the cutoff — change nDCG@10 of a zero-shot Qwen3-4B-Instruct ranker relative to the same ranker without news, when the
candidates, their initial order and the client's own assets are held fixed?</div>
<p><b>Why this place.</b> Before any LLM run we checked, without a model, every transaction signal we could build for ordering
never-bought assets: co-buyers of the client's assets, "bought X then Y" transitions, 7- and 90-day popularity, buying spikes, name and
fund-house similarity, subcategory. On March–April 2021 none beat 30-day popularity (0.548–0.573 vs 0.571 nDCG@10 over buyers; perfect order
0.601). Of 160 new purchases in the April block, 156 were stocks of large Greek companies (Piraeus, Alpha, NBG, OTE, Mytilineos) — firms
that the press covers daily. Retail investors are known to buy attention-grabbing stocks [5], so outside news is the one candidate signal
left. <b>LLM role:</b> Ranker only. It changes only the order (and choice) of never-bought stocks in the free slots; the own-asset block and
the candidate set come from the HW2 rule cascade and are identical in every arm.</p>

<h2>2. Related Work</h2>
<table class="small"><tr><th>Study</th><th>What it shows</th><th>What we take</th></tr>
<tr><td>[1] Sanz-Cruzado, Droukas, McCreadie 2024</td><td>FAR-Trans: customers of a Greek bank, 2018–2022, with prices and metadata; no news.</td><td>Data; we add an external, time-stamped source.</td></tr>
<tr><td>[2] Hou et al., ECIR 2024 (LLMRank)</td><td>Zero-shot LLMs can rerank a fixed candidate list; sensitive to order and to the history shown.</td><td>Fixed candidates, random shown order, matched no-LLM control.</td></tr>
<tr><td>[3] Lyu et al., NAACL Findings 2024 (LLM-Rec)</td><td>LLM-generated text about items can improve content-based recommendation.</td><td>Course encoder baseline; we keep the LLM out of encoding (HW2 showed no gain).</td></tr>
<tr><td>[4] Lewis et al., NeurIPS 2020 (RAG)</td><td>Retrieved evidence conditions generation on a fixed model.</td><td>Evidence retrieval with the candidates held fixed (lecture 9).</td></tr>
<tr><td>[5] Barber, Odean, RFS 2008</td><td>Individual investors are net buyers of attention-grabbing stocks (news, volume, extreme returns).</td><td>Why news attention could predict first purchases.</td></tr>
<tr><td>[6] Leetaru, Schrodt 2013 (GDELT)</td><td>Global news monitored and machine-translated, with dates and tone.</td><td>Our news source, filtered to Greek outlets.</td></tr></table>

<h2>3. Method</h2>
<p><b>List assembly (all arms).</b> (1) Up to 10 own assets in the HW2 rule order (decayed purchase frequency; this is "Duet" cell 1 — the
Duet v2 encoder gave the same 10-item set as the rule for every client, so it is not used). (2) Free slots k = 10 − own assets. (3) A block of
the 20 most-bought never-traded assets of the client's main category (bank buyers in the 30 days before the cutoff). C0 fills the k slots in
this popularity order; A1 and M let the LLM choose k of the 20. The LLM is called only for Stock-main clients (the news layer covers listed
stocks; the rule reads past data only); other clients keep C0.</p>
<p><b>Prompt.</b> Client: main category, number of assets, buys in the last and previous 90 days, sectors of past buys, profile. 20 candidate
rows in random order with random IDs (A10–A99): category, sector, bank buyers in 30 days and rank, 30-day price change; no names or ISINs.
<b>M</b> adds, under each stock with collected news, one line of numbers (articles in 30 and 7 days, ratio to the stock's own level over the
previous 60 days, share of coverage, average GDELT tone) and the two latest headlines that name the company, with the name replaced by
"[company]". <b>A1</b> is the identical prompt without these lines (a unit test checks this). Output: a JSON array of k IDs. Invalid answers
keep their valid IDs and are completed in the initial order (repair); with no valid ID the list falls back to C0; every case is counted.</p>
<p><b>Time safety.</b> Transactions, prices and profiles strictly before the cutoff (HW2 assertion). News counts use UTC days up to cutoff − 2;
a headline is used only if GDELT first saw it before 21:00 UTC on cutoff − 1 (Athens midnight); both are asserted and unit-tested.
<b>Model.</b> Qwen/Qwen3-4B-Instruct-2507 @ cdbee75, greedy, thinking off, float16 on Apple M3 Pro (MPS, 18 GB), torch 2.11.0,
transformers 4.57.6, 90 new tokens. Code: <code>block_ranker.py</code>, <code>news.py</code>, <code>collect_gdelt.py</code>, 6 unit tests.</p>

<h2>4. Experiments</h2>
<p><b>Protocol (fixed before comparing methods).</b> Development cohort D1: the first 3,000 warm users of the starter order with ≥1 Buy in
the 90 days before 1 May 2021 (1,498 clients; 408 bought in May; 160 bought a never-traded asset); 30-day window; repeats allowed; selection
from past data only (protocol_hash e8a41099ddb7, hw3_protocol_hash 381cbd19c21f). Prompt checks used April 2021 only. Teaching set D0: the
starter's 100 users at 1 June 2021 (protocol_hash 50efd4aeb415 = the starter run). Primary metric: mean nDCG@10 over all clients (empty
truth = 0); paired bootstrap (10,000 resamples, seed 0) and sign test.</p>
{table(T)}
<div class="caption">Table 1. D1, May 2021. The ceiling reorders the block perfectly and bounds what any arm can reach: +{T['oracle_block']['all'] - T['c0_popularity']['all']:.4f} over C0 on all clients.</div>
{contrasts(main)}
<div class="caption">Table 2. Paired differences on the same 1,498 clients (clients with identical lists contribute 0).</div>
{gng_text}
{subset_html}
<p><b>Validity and cost.</b> A1: {st_a1.get('ok', 0)} valid answers, {st_a1.get('repaired', 0)} repaired, {st_a1.get('fallback', 0)} fallbacks
for 669 calls; generation {gen_a1 / 60:.0f} min ({gen_a1 / 669:.1f} s per client, ~3,760-character prompts); preparation of the cohort 13 s.
{m_status} <b>News coverage:</b> GDELT rate limits allowed a full download for only part of the 30 planned stocks within the deadline; at the
May cutoff {covd1['tracked_stocks_in_blocks']} stocks in clients' blocks have news counts, {covd1['with_articles_30d']} of them with articles
in the last 30 days and {covd1['with_headline']} with a headline that names the company; the three largest banks are not covered
(collection stopped by the rate limit). News collection took about 12 h of wall time, mostly waiting.</p>
<p><b>Teaching set D0.</b> Popularity {f4(d0['table']['popularity']['all'])}, CBF {f4(d0['table']['cbf']['all'])}, Latest buy
{f4(d0['table']['latest_buy']['all'])}, Latest buy or sell {f4(d0['table']['latest_buy_or_sell']['all'])}, C0 {f4(d0['table']['c0_popularity']['all'])};
course LLMRank 0.0201 (HW2 results, same protocol hash). The perfect block order equals C0 here, so no arm can improve D0; conclusions rest on D1.</p>

<h2>5. Discussion and failure analysis</h2>
{choice([('A1, 669 clients', ch_a1)] + ([('A1, first 78', ch_a1s), ('M, first 78', ch_m)] if ch_m else []))}
<div class="caption">Table 3. What the LLM picked instead of the popularity order (669 Stock-main clients, May).</div>
<p><b>Observation.</b> Without news, the LLM keeps only about half of C0's picks, chooses less popular stocks and stocks that rose more in
the last 30 days, and is not biased towards the first rows shown. It removed {ch_a1['bought_assets_dropped_from_C0_slots']} assets that clients
then bought and brought in {ch_a1['bought_assets_added_by_llm']}. <b>Possible explanation (not tested):</b> the model reads "price change" as a reason
to buy, while in this bank the crowd signal (how many clients bought it) is what predicts the next first purchase. {m_disc}</p>
<p><b>Limitations.</b> One development month and one cutoff; the effect can only appear for the ~80 clients who bought something from their block
(April estimate). Partial news coverage, no large banks. GDELT tone is lexicon-based on machine translation; title matching is precise but
misses articles that do not name the company in the title. Not run for lack of time: the placebo with news moved to other stocks (A3), numbers
without headlines (A2), a second shown order and visible names; these are implemented (<code>run_hw3_night2.command</code>). Purchase
prediction is not suitability or profit; not investment advice.</p>
<p><b>Team decisions and my contribution.</b> Agreed: shared data and course protocol, the four baselines and nDCG@10; each member tests one component. Zlata (asset-field filling, all 10,404 warm clients, 1 June 2021): LLM-inferred sector/industry/region (0.0427) ≈ paraphrase (0.0424), a keyword-rule ablation adds nothing, Latest buy is strongest (0.0526). Maria: history weights for the encoder. Our three parts agree: on top of what code already does, the LLM has not yet given a significant gain, and repeat purchases are the main signal. Proposed merge (not yet confirmed by all): the final system is the cascade "own assets first, then new assets by popularity" (it contains the Latest-buy idea and is not worse than it on May: 0.1486 vs 0.1478); each component plugs into its own slot as a separate ablation — Maria: order of own assets, Zlata: filled fields for the content-based part and candidate descriptions, me: order of the new-asset block — and only components not worse than the cascade on the official set enter the final submission. Open: who assembles the shared cascade code and when the final version is frozen. My work: the HW3 question and its narrowing from HW2, the no-LLM signal checks, the news design, all runs on my Mac. <b>My own check:</b> client 0E72D1 (D1, May): its only May purchase is INTRACOM S.A. HOLDINGS (GRS087003000, bought 2021-05-10, from transactions.csv). In predictions.csv it is at rank 4 for M (llm_full) and rank 8 for C0. With one relevant item IDCG = 1, so nDCG@10 = 1/log2(4+1) = 0.4307 for M and 1/log2(8+1) = 0.3155 for C0, both equal to per_user.json. I also opened one headline used in the May prompts (902.gr, "Τέσσερις απολύσεις σε μια εβδομάδα από την ΤΕΡΝΑ ΕΝΕΡΓΕΙΑΚΗ"): GDELT first saw it at 17:30 UTC on 30 April 2021, before the 1 May cutoff boundary.</p>

<h2>6. AI Usage Disclosure</h2>
<ul><li><b>Tools:</b> Claude Opus 5.5 (Anthropic) in the Claude app with file, shell and Mac tools; Qwen3-4B-Instruct-2507 is the model under test.</li>
<li><b>Used for:</b> reading the assignment and lecture, the no-LLM signal checks, the code (collector, news layer, ranker, tests, run scripts),
the choice analysis, and drafting this report from the run outputs; numbers in tables are produced by <code>report/build_hw3_report.py</code>.</li>
<li><b>Checks the AI performed:</b> baselines and C0 reproduce HW2 on D0 to 4 decimals; C0 equals the HW2 rule list for all 100 D0 users;
6 unit tests (time boundary, masking, parsing, arm isolation, fake-model run); collector tested on synthetic GDELT responses.</li></ul>

<h2>References</h2>
<p class="small">[1] J. Sanz-Cruzado, N. Droukas, R. McCreadie. FAR-Trans: An Investment Dataset for Financial Asset Recommendation. IJCAI Workshop on Recommender Systems in Finance, 2024.<br>
[2] Y. Hou et al. Large Language Models are Zero-Shot Rankers for Recommender Systems. ECIR 2024.<br>
[3] H. Lyu et al. LLM-Rec: Personalized Recommendation via Prompting Large Language Models. Findings of NAACL 2024.<br>
[4] P. Lewis et al. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.<br>
[5] B. Barber, T. Odean. All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors. Review of Financial Studies 21(2), 2008.<br>
[6] K. Leetaru, P. Schrodt. GDELT: Global Data on Events, Location and Tone, 1979–2012. ISA Annual Convention, 2013.</p>
</body></html>"""
out = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'kutranov_vladislav_hw3_report.html'
out.write_text(html, encoding='utf-8')
print('written', out, '| M included:', bool(m1))
