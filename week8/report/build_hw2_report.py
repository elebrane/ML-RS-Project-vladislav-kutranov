"""Build the HW2 report (HTML -> PDF via render.js) from the saved run outputs.

Every number in the tables is read from run or analysis files, not typed by hand:
  outputs/hybrid_test_june/{metrics.json, per_user.json, run_manifest.json, details.json}   main test
  outputs/hybrid_smoke_may{,_v2,_v3}/run_manifest.json                                    smoke runs
  outputs/hw2_main/{metrics.json, run_manifest.json, stage1_accuracy.json}                Experiment 1
  outputs/_scratch/exp2_partial_scores.json, outputs/explore_q4b_val/propensity.json       Experiment 2
  outputs/_scratch/headroom3k/headroom_2021-05-01.json                                     pool analysis
  report/data/{prelim_validation, fusion_validation, overlap_june}.json                    no-LLM analyses
Usage: python build_hw2_report.py [OUTPUTS_DIR] [OUT_HTML]
Then:  node render.js "$PWD/<OUT_HTML>" "$PWD/<OUT_PDF>"
"""
from __future__ import annotations

import json
import random
import statistics
import sys
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parent / 'far-trans-llm' / 'outputs'
OUT_HTML = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'kutranov_vladislav_hw2_report.html'
DATA = HERE / 'data'


def load(p):
    return json.loads(Path(p).read_text())


TEST = OUT_DIR / 'hybrid_test_june'
metrics = load(TEST / 'metrics.json')
per_user = load(TEST / 'per_user.json')
manifest = load(TEST / 'run_manifest.json')
details = load(TEST / 'details.json')
rt = manifest['runtime']
smokes = {v: load(OUT_DIR / f'hybrid_smoke_may{v}' / 'run_manifest.json')['runtime'] for v in ('', '_v2', '_v3')}
e1 = load(OUT_DIR / 'hw2_main' / 'metrics.json')
e1rt = load(OUT_DIR / 'hw2_main' / 'run_manifest.json')['runtime']
e1acc = load(OUT_DIR / 'hw2_main' / 'stage1_accuracy.json')
e2 = load(OUT_DIR / '_scratch' / 'exp2_partial_scores.json')
prop = load(OUT_DIR / 'explore_q4b_val' / 'propensity.json')
head_may = load(OUT_DIR / '_scratch' / 'headroom3k' / 'headroom_2021-05-01.json')
prelim = load(DATA / 'prelim_validation.json')
fusion = {r['cutoff']: r for r in load(DATA / 'fusion_validation.json')}
overlap = load(DATA / 'overlap_june.json')

BASE = [('popularity', 'Popularity'), ('cbf', 'CBF'), ('latest_buy', 'Latest buy'),
        ('latest_buy_or_sell', 'Latest buy or sell')]
CELLS = [('hyb_c1_rule', '1. Rule, no LLM (control)'), ('hyb_c2_encoder', '2. LLM encoder only'),
         ('hyb_c3_reranker', '3. LLM reranker only'), ('hyb_c4_both', '4. Encoder + reranker (method)')]
fails = rt['failures_by_stage']
fail_n = {'hyb_c1_rule': 0, 'hyb_c2_encoder': fails.get('encoder', 0),
          'hyb_c3_reranker': fails.get('reranker_c3_p1', 0),
          'hyb_c4_both': fails.get('encoder', 0) + fails.get('reranker_c4_p1', 0)}
secs = rt['runtime_seconds_by_method']

score = {}
for r in per_user:
    score.setdefault(r['user_id'], {})[r['method']] = (r['ndcg_at_10'], r['relevant_count'])
buyers = [u for u in manifest['users'] if score[u]['latest_buy'][1] > 0]


def f4(x):
    return f'{x:.4f}'


def boot(a, b, n=10000):
    diffs = [score[u][a][0] - score[u][b][0] for u in buyers]
    rng = random.Random(0)
    bs = sorted(statistics.mean(rng.choices(diffs, k=len(diffs))) for _ in range(n))
    return (statistics.mean(diffs), bs[int(0.025 * n)], bs[int(0.975 * n)],
            sum(d > 0 for d in diffs), sum(d < 0 for d in diffs))


def results_table():
    rows = []
    for key, name in BASE + CELLS:
        m = metrics[key]
        run = ('&lt;1 s' if secs[key] < 1 else f'{secs[key]:,.0f} s'.replace(',', '&#8239;')) if key in secs else 'in run'
        v1, v2 = f4(m['ndcg_at_10']), f4(m['ndcg_at_10_nonempty'])
        if key == 'hyb_c2_encoder':
            v1, v2 = f'<b>{v1}</b>', f'<b>{v2}</b>'
        rows.append(f"<tr><td>{name}</td><td>{m['users']}</td><td>{m['nonempty_truth_users']}</td>"
                    f"<td>{v1}</td><td>{v2}</td><td>{run}</td><td>{fail_n.get(key, 0)}</td></tr>")
    return ('<table class="small"><tr><th>Method</th><th>Users</th><th>Users with future purchases</th>'
            '<th>nDCG@10: all users</th><th>nDCG@10: nonempty truth</th><th>Runtime</th>'
            '<th>Failed cases</th></tr>' + ''.join(rows) + '</table>')


def contrast_table():
    pairs = [('hyb_c4_both', 'latest_buy', '4 vs Latest buy (RQ)'),
             ('hyb_c1_rule', 'latest_buy', '1 vs Latest buy'),
             ('hyb_c2_encoder', 'hyb_c1_rule', '2 vs 1 (encoder)'),
             ('hyb_c3_reranker', 'hyb_c1_rule', '3 vs 1 (reranker)'),
             ('hyb_c4_both', 'hyb_c3_reranker', '4 vs 3 (evidence)'),
             ('hyb_c4_both', 'hyb_c2_encoder', '4 vs 2')]
    rows = []
    for a, b, name in pairs:
        mean, lo, hi, w, l = boot(a, b)
        rows.append(f'<tr><td>{name}</td><td>{mean:+.4f}</td><td class="nw">[{lo:+.4f}, {hi:+.4f}]</td><td>{w} / {l}</td></tr>')
    return ('<table class="small"><tr><th>Contrast, 11 buyers</th><th>Mean &Delta;</th>'
            '<th>95% paired bootstrap CI</th><th>Better / worse</th></tr>' + ''.join(rows) + '</table>')


def buyer_table():
    rows = []
    for u in buyers:
        info = details[u]
        own = {x['id'] for x in info['pool']['own']}
        tp = info['truth_in_pool']
        new_in = sum(1 for x in tp if x not in own)
        cells = ''.join(f'<td>{score[u][k][0]:.3f}</td>' for k in
                        ('latest_buy', 'hyb_c1_rule', 'hyb_c2_encoder', 'hyb_c3_reranker', 'hyb_c4_both'))
        enc = '' if info.get('encoder', {}).get('valid') else '&dagger;'
        rows.append(f'<tr><td>{u[:6]}{enc}</td><td>{info["truth_size"]}</td><td>{len(tp)} ({new_in})</td>{cells}</tr>')
    return ('<table class="small"><tr><th>User</th><th>Bought</th><th>In pool (new)</th><th>Latest buy</th>'
            '<th>Cell 1</th><th>Cell 2</th><th>Cell 3</th><th>Cell 4</th></tr>' + ''.join(rows) + '</table>')


def earlier_table():
    w, inv = e1rt['wall_seconds_by_arm'], e1rt['invalid_by_arm']
    r1 = [('freq_recency', 'Own by frequency, then recency; then CBF (control)', '—', '0'),
          ('llm_R_raw', 'R: LLM ranks from the raw dated history', f"{w['R']:,.0f} s", inv['R']),
          ('llm_Rp_permuted_dates', 'Rp: as R, purchase dates shuffled', f"{w['Rp']:,.0f} s", inv['Rp']),
          ('llm_S_self_structured', 'S: LLM writes a profile, then ranks by it', f"{w['S']:,.0f} s", inv['S']),
          ('llm_C_code_structured', 'C: as S, profile computed by code', f"{w['C']:,.0f} s", inv['C'])]
    rows = ['<tr><th colspan="5">E1 — course test cutoff 2021-06-01, 100 users (11 with purchases), Qwen3-1.7B, 50 fixed candidates; Latest buy as in Table 1</th></tr>']
    for k, name, t, f in r1:
        m = e1[k]
        rows.append(f"<tr><td>{name}</td><td>{f4(m['ndcg_at_10'])}</td><td>{f4(m['ndcg_at_10_nonempty'])}</td><td>{t.replace(',', '&#8239;')}</td><td>{f}</td></tr>")
    em, inv2 = e2['methods'], e2['invalid_by_arm']
    r2 = [('latest_buy', 'Latest buy (provided)'), ('latest_buy_or_sell', 'Latest buy or sell (provided)'),
          ('explore_control_popularity', 'New slots filled by popularity in client category (control)'),
          ('explore_llm', 'LLM picks the new assets'), ('explore_llm_masked_names', 'As LLM, asset names hidden')]
    rows.append(f"<tr><th colspan=\"5\">E2 — validation cutoff 2021-05-01, run stopped after {e2['users_done']} of {e2['users_planned']} users ({e2['buyers']} with purchases), Qwen3-4B-Instruct-2507</th></tr>")
    for k, name in r2:
        m = em[k]
        caught = '' if m['new_caught'] is None else f" (new caught {m['new_caught']}/{e2['new_assets_bought']})"
        rows.append(f"<tr><td>{name}{caught}</td><td>{f4(m['ndcg_all'])}</td><td>{f4(m['ndcg_buyers'])}</td><td>—</td><td>{inv2.get(k.replace('explore_', ''), 0) if k.startswith('explore_llm') else 0}</td></tr>")
    return ('<table class="small"><tr><th>Earlier attempt / method</th><th>nDCG@10: all users</th><th>nDCG@10: nonempty truth</th>'
            '<th>LLM time</th><th>Failed</th></tr>' + ''.join(rows) + '</table>')


def worked_example(prefix='6BAA2D'):
    """Real encoder and reranker calls of one test user, read from generations.jsonl and details.json."""
    import re
    uid = next(u for u in details if u.startswith(prefix))
    info = details[uid]
    ids = info['table_ids']
    recs = [json.loads(line) for line in (TEST / 'generations.jsonl').read_text().splitlines()]
    enc = next(r for r in recs if r['max_new_tokens'] != 100 and ('these ids, no more and no fewer: ' + ', '.join(ids)) in r['prompt'])
    c3 = next(r for r in recs if r['max_new_tokens'] == 100 and r['response']['text'] == info['c3_p1']['raw'] and ids[0] + ' | Stock' in r['prompt'])
    c4 = next(r for r in recs if r['max_new_tokens'] == 100 and r['response']['text'] == info['c4_p1']['raw'] and '"asset_status"' in r['prompt'])
    st = enc['settings']

    def short(line):
        line = re.sub(r'category rank (\d+) \((?:not )?top-10\)', r'rank \1', line)
        line = re.sub(r'(rank \d+), spike no, price 30d ', r'\1, ', line)
        line = re.sub(r'(rank \d+), spike yes, price 30d ', r'\1, spike, ', line)
        return line.replace(' | Stock, ', ' | ')

    def tok(r):
        return f"{r['response']['input_tokens']:,}/{r['response']['output_tokens']}, {r['response']['seconds']:.0f} s"

    lines = [l for l in enc['prompt'].splitlines()]
    prof = next(l for l in lines if l.startswith('Client profile'))
    act = next(l for l in lines if l.startswith('Client activity'))
    rows = [short(l) for l in lines if l.split(' | ')[0] in ('A76', 'A19', 'A94')]
    out = json.loads(enc['response']['text'])
    E = []
    E.append(f"<b>ENCODER</b> prompts/encoder_v3.txt | {st['model'].split('/')[1]}@{st['resolved_revision'][:7]}, greedy | max_new_tokens {enc['max_new_tokens']} | tokens in/out {tok(enc)}")
    prof = prof.replace('Client profile at 2021-06-01: ', '').replace('customer type ', '').replace('risk profile ', 'risk ').replace('investment capacity ', 'capacity ')
    E.append('<b>Input</b> (3 of 6 rows): profile ' + escape(prof))
    E.append('  activity: ' + escape(act.replace('Client activity: ', '').replace('. Main asset category:', '; main category')))
    E.append('  id | sector | buys all/90d/30d | first, last buy | gap | sells, last | position | vs avg buy | decayed | at first buy | at cutoff')
    E += ['  ' + escape(r) for r in rows]
    E.append('<b>Output</b> (complete, one line in the original):')
    pats = [escape(f'"{ptn["pattern"]}"') + ' &rarr; ' + ', '.join(ptn['supporting_assets']) for ptn in out['observed_patterns']]
    plain = lambda x: len(re.sub('&[a-z]+;', 'x', x))
    line = '  observed_patterns: '
    for ptn in pats:
        if plain(line) + plain(ptn) > 124 and line.strip():
            E.append(line.rstrip('; '))
            line = ' ' * 21
        line += ptn + '; '
    E.append(line.rstrip('; '))
    sts = [f'{a} {v["status"]} ("{v["note"]}")' for a, v in out['asset_status'].items()]
    for i in range(0, len(sts), 2):
        E.append(('  asset_status: ' if i == 0 else '                ') + escape('; '.join(sts[i:i + 2])))
    E.append('  unknowns: ' + escape('; '.join(f'"{u}"' for u in out['unknowns'])))
    E.append(f"<b>RERANKER</b> prompts/reranker_v1.txt | same model | max_new_tokens {c4['max_new_tokens']} | tokens in/out: cell 3 {tok(c3)}; cell 4 {tok(c4)}")
    cl = c4['prompt'].splitlines()
    cand = [l for l in cl if ' | own | ' in l or ' | new | ' in l]
    pick = [l for l in cand if l.split(' | ')[0] in ('A94', 'A76', 'A26')]
    E.append(f'<b>Input</b>: client block (cell 3: fact rows; cell 4: encoder JSON) + {len(cand)} shuffled candidates; 3 of them as in cell 4:')
    E += ['  ' + escape(short(l)) for l in pick]
    sc = {k: score[uid][k][0] for k in ('hyb_c1_rule', 'hyb_c2_encoder', 'hyb_c3_reranker', 'hyb_c4_both')}
    rk = info['rankings']
    E.append('<b>Output</b> cell 3: ' + escape(info['c3_p1']['raw'].replace('\n', '').replace('  ', '').replace('"', '').replace(',', ', ')) + f' &rarr; only new assets, A94 dropped, nDCG@10 {sc["hyb_c3_reranker"]:.3f}')
    E.append('<b>Output</b> cell 4: ' + escape(info['c4_p1']['raw'].replace('"', '')) + f' &rarr; A94 at rank {rk["hyb_c4_both"].index("A94") + 1}, nDCG@10 {sc["hyb_c4_both"]:.3f}')
    E.append(f'<b>Cascades</b> (no reranker): cell 1 {", ".join(rk["hyb_c1_rule"][:4])}&hellip; nDCG@10 {sc["hyb_c1_rule"]:.3f}; cell 2 {", ".join(rk["hyb_c2_encoder"][:4])}&hellip; nDCG@10 {sc["hyb_c2_encoder"]:.3f}. Bought in June: A94.')
    rule = info['rule_status']
    differ = [a for a in ids if out['asset_status'][a]['status'] != rule[a]]
    return '<pre class="wx">' + '\n'.join(E) + '</pre>', len(differ), len(ids)


def validation_table():
    rows = []
    for p in prelim:
        fu = fusion[p['cutoff']]
        rows.append(f"<tr><td class='nw'>{p['cutoff'][:7]}</td><td>{p['buyers']}</td><td>{f4(p['latest_buy'])}</td><td>{f4(p['cell1'])}</td>"
                    f"<td>{f4(p['core_first'])}</td><td>{f4(p['exited_first'])}</td><td>{f4(fu['rrf(all four)'])}</td>"
                    f"<td>{f4(fu['oracle_best_of(all four)'])}</td><td>{100 * p['pool_recall']:.0f}%</td><td>{f4(p['oracle_pool'])}</td></tr>")
    return ('<table class="small"><tr><th>Cutoff</th><th>Buyers (of 3,000)</th><th>Latest buy</th><th>Cell 1 rule</th>'
            '<th>Status cascade, rule statuses, core first</th><th>&hellip; exited first</th><th>Rank fusion of 4 methods</th>'
            '<th>Best of 4 per user (hindsight)</th><th>Purchases in pool</th><th>Perfect order of pool</th></tr>'
            + ''.join(rows) + '</table>')


def sign_p(w, l):
    from math import comb
    k = w + l
    return min(1.0, 2 * sum(comb(k, i) for i in range(min(w, l) + 1)) / 2 ** k) if k else 1.0


in_pool_buyers = [u for u in buyers if details[u]['truth_in_pool']]
own_top3, new_first, dropped_own = [], 0, 0
for u in manifest['users']:
    d = details[u]
    own = {x['id'] for x in d['pool']['own']}
    if not own:
        continue
    rk = d['rankings']['hyb_c3_reranker']
    k = min(3, len(own))
    own_top3.append(sum(1 for i in rk[:k] if i in own) / k)
    new_first += rk[0] not in own
for u in in_pool_buyers:
    d = details[u]
    own = {x['id'] for x in d['pool']['own']}
    c1, c3 = d['rankings']['hyb_c1_rule'], d['rankings']['hyb_c3_reranker']
    dropped_own += any(t in own and t in c1 and t not in c3 for t in d['truth_in_pool'])
diag = rt['diagnostics']
agree = diag['status_agreement_with_rule']
agree_pct = 100 * agree['agree'] / (agree['agree'] + agree['differ'])
s1, s2, s3 = smokes[''], smokes['_v2'], smokes['_v3']
c4lb = boot('hyb_c4_both', 'latest_buy')
c21 = boot('hyb_c2_encoder', 'hyb_c1_rule')
c31 = boot('hyb_c3_reranker', 'hyb_c1_rule')
c43 = boot('hyb_c4_both', 'hyb_c3_reranker')
m = {k: metrics[k] for k, _ in BASE + CELLS}
wall_min = rt['total_wall_seconds'] / 60
enc_calls = len(manifest['users'])
hm = head_may['mean_ndcg_buyers']
hr, hn = head_may['mean_ndcg_only_repeats'], head_may['mean_ndcg_bought_new']
src = head_may['new_found_by_source']
ov = overlap['mean_common_items_in_top10']
gain_lo = min(fusion[c]['oracle_best_of(all four)'] - max(fusion[c]['latest_buy'], fusion[c]['cell1']) for c in fusion)
gain_hi = max(fusion[c]['oracle_best_of(all four)'] - max(fusion[c]['latest_buy'], fusion[c]['cell1']) for c in fusion)
wx_html, wx_diff, wx_n = worked_example()
e1best = max(('llm_R_raw', 'llm_Rp_permuted_dates', 'llm_S_self_structured', 'llm_C_code_structured'),
             key=lambda k: e1[k]['ndcg_at_10_nonempty'])

CSS = (HERE / 'style.css').read_text() if (HERE / 'style.css').exists() else ''

html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>HW2 report — Vladislav Kutranov</title>
<style>{CSS}</style></head><body>

<h1>Does a Zero-Shot LLM Encoder + Reranker Beat Latest Buy on FAR-Trans? A 2&times;2 Ablation</h1>
<div class="meta">
  <div><b>Student:</b> Vladislav Kutranov &nbsp; <b>Team:</b> <span class="fill">[team name]</span> &nbsp; <b>Email:</b> vvkutranov@edu.hse.ru &nbsp; <b>Date:</b> <span class="fill">[submission date]</span></div>
  <div><b>Assignment:</b> Homework 2 — Implement and Test Your Idea (LLM4Rec, Session 2, Week 8) &nbsp; <b>Track:</b> Hybrid (Encoder + Ranker) &nbsp; <b>State:</b> implemented, run once on the course test</div>
  <div><b>Repository:</b> <a href="https://github.com/elebrane/ML-RS-Project-vladislav-kutranov">github.com/elebrane/ML-RS-Project-vladislav-kutranov</a>, folder <code>week8/</code> &nbsp; <b>Experiment commit:</b> <span class="fill">[commit hash after push]</span></div>
</div>

<div class="abstract"><b>Abstract</b> — Under the course protocol most FAR-Trans purchases in the next 30 days are re-buys of assets the client already traded, so the provided recency baseline Latest buy is hard to beat. After two zero-shot LLM attempts failed to beat simple controls, I built a hybrid with Qwen3-4B-Instruct-2507: code computes literature-grounded facts per client asset, an LLM encoder turns them into cited evidence and statuses, and an LLM reranker orders a fixed pool of own and popular assets; a 2&times;2 ablation switches each part on and off. On the 100 course users the full hybrid scored nDCG@10 {f4(m['hyb_c4_both']['ndcg_at_10'])} against {f4(m['latest_buy']['ndcg_at_10'])} for Latest buy, while the encoder alone with a fixed cascade scored {f4(m['hyb_c2_encoder']['ndcg_at_10'])}, the best of all eight methods (11 users with purchases). The LLM helped as a re-describer of the client's own history and hurt as the final ranker, because it promoted popular new assets over the client's repeats.</div>
<div class="terms">Index Terms — financial asset recommendation, LLM reranking, zero-shot prompting, repeat purchases, behavioural finance, ablation.</div>

<h2>1. Introduction</h2>
<p><b>Problem.</b> Given a bank client's transactions before 1 June 2021, rank 10 assets so that the ones bought in the next 30 days come first (nDCG@10, repeats allowed) [1], [2]. Only 11 of the 100 course users buy anything in the window; they buy 18 assets, 12 of which are in our candidate pool and 10 of those are re-purchases.</p>
<p><b>From Session 1 to this homework.</b> In Session 1 I planned a Ranker study: an LLM reorders the 30 most popular assets to find assets <i>new</i> to the client, against popularity, as in the FAR-Trans benchmark [1]. The Session 2 protocol counts repeats over 30 days, and my Session 1 audit found that 60.5% of buy rows repeat an earlier purchase; on the May validation cutoff Latest buy scores {f4(hr['latest_buy'])} for the {hr['n']} buyers who only re-bought but {f4(hn['latest_buy'])} for the {hn['n']} who bought something new. So I kept the reranker and added an encoder for the client's own history (Hybrid track); two intermediate attempts (E1, E2, &sect;4.4) shaped the design.</p>
<p><b>Motivation.</b> Studies of investors' <i>trades</i> — preferred to surveys, whose most frequent answers are not the strongest drivers of actual trading [3] — give concrete, computable signals: individuals buy what catches attention, such as stocks with abnormal volume or extreme returns [4] and the day's most-bought stocks [5]; investors trading on smartphones buy more recent winners [6]; investors avoid re-buying a stock they sold at a loss [7]; and in next-basket recommendation repeat items carry most of the accuracy [8]. These signals are relational ("re-bought every three weeks", "sold everything at a loss"), so an LLM might read them better than one frequency score; the FAR-Trans benchmark models use the purchase matrix or prices instead [1].</p>
<p><b>Concrete example (user 6BAA2D; full inputs and outputs in &sect;3).</b> The rule (cell 1) ranks the client's own assets A76, A61, A19, <u>A94</u>&hellip; by decayed frequency; the encoder labels A76 <i>exited_loss</i> and A19, A94 <i>core</i>, so the same cascade (cell 2) gives A19, <u>A94</u>, A61&hellip;. A94 is the June purchase: nDCG@10 rises from 0.431 to 0.631.</p>
<div class="rq"><b>RQ.</b> We test whether a zero-shot two-stage LLM pipeline (Qwen3-4B-Instruct-2507) improves nDCG@10 over the provided Latest buy baseline for the 100 warm FAR-Trans users at the 1 June 2021 cutoff, where (1) the <b>encoder</b> reads a code-computed table of behavioural facts for each asset the client traded — repurchase rhythm, current position and the result of the last sale, and bank-wide attention (buyer rank, 7-day buying spike and 30-day price change, each at the client's first buy and at the cutoff) — and returns cited patterns plus one repurchase status per asset (core, exited at a gain, exited at a loss, new, lapsed, one-off); and (2) the <b>reranker</b> orders a fixed pool of the client's own assets and the most-bought new assets of the client's main category using that evidence, keeping own assets first unless the evidence points elsewhere.<br>
<span class="pico"><b>What is new</b> compared with the course LLM baselines [2], whose encoder paraphrases asset descriptions for TF-IDF matching and whose ranker reorders 50 CBF candidates from Buy counts and the 15 latest dated buys: (a) LLM input built in code from findings on investor behaviour (Table A) instead of raw history or descriptions; (b) encoder output as cited evidence with repurchase statuses, which also drives a cascade without a reranker (cell 2); (c) a pool of own plus attention-ranked new assets, and a reranker whose departures from "own first" must rest on that evidence; (d) a same-pool 2&times;2 ablation isolating each LLM part. <b>Control:</b> Latest buy, the best provided baseline on validation ({f4(hm['latest_buy'])} vs {f4(hm['latest_buy_or_sell'])} for Latest buy or sell on {head_may['buyers']} May buyers), plus the same pool ordered by a decayed-frequency rule without an LLM (cell 1). <b>Outcome:</b> mean nDCG@10 over all 100 users (empty truth = 0), plus over users with purchases. <b>Population:</b> the starter's 100 deterministic warm users, 30-day window.</span></div>
<ul><li>A runnable hybrid with additions (a)–(d), and all baselines on one protocol hash.</li>
<li>A negative answer to the RQ with one positive ablation finding (encoder statuses improve the rule cascade for all 5 users whose order changed), on 11 labelled users only, plus two earlier negative attempts.</li></ul>

<h2>2. Related Work</h2>
<table class="small rw"><tr><th>Study</th><th>Finding</th><th>What we take from it</th></tr>
<tr><td>[1] Sanz-Cruzado, Droukas, McCreadie, 2024</td><td>FAR-Trans: a European bank's Greek market, 2018–2022; 11 algorithms, 61 cuts, 6-month horizon, only assets new to the client relevant; LightGCN nDCG@10 0.3404, Popularity 0.2710.</td><td>Data and provided baselines; our protocol differs (30 days, repeats count).</td></tr>
<tr><td>[9] Sanz-Cruzado et al., ACM TOIS 2025</td><td>On the same bank, transaction-based metrics (nDCG) and profitability metrics of recommenders are weakly negatively correlated.</td><td>nDCG@10 measures purchase prediction, not suitability or profit (&sect;5).</td></tr>
<tr><td>[4] Barber, Odean, RFS 2008</td><td>Individual investors are net buyers of attention-grabbing stocks: in the news, with abnormal volume, with extreme one-day returns.</td><td>Popularity rank, 7-day spike and 30-day price change as facts.</td></tr>
<tr><td>[5] Barber et al., J. Finance 2022</td><td>Robinhood users herd into the day's most-bought stocks, which then underperform.</td><td>Popular-new half of the pool; ethics caveat.</td></tr>
<tr><td>[7] Strahilevitz, Odean, Barber, JMR 2011</td><td>Investors avoid repurchasing stocks sold at a loss and stocks that rose after the sale.</td><td>Position and exit-result facts; <i>exited_gain / exited_loss</i>.</td></tr>
<tr><td>[8] Li et al., ACM TOIS 2023</td><td>Much of next-basket accuracy comes from repeat items; repeat and explore should be evaluated separately.</td><td>Own-first default; repeat/new breakdown.</td></tr>
<tr><td>[10] Hou et al., ECIR 2024</td><td>Zero-shot LLM rankers show position and popularity bias and struggle to use the order of the history.</td><td>Shuffled candidates; history given as computed facts.</td></tr></table>
<p>The course LLM baselines on the same users (Qwen3-1.7B) reached 0.0254 (encoder) and 0.0201 (ranker) [2]. <b>Alternatives considered.</b> (a) An encoder that outputs weights over four candidate sources — dropped: its formula control lost to Latest buy on May ({f4(hm['A2_formula'])} vs {f4(hm['latest_buy'])}). (b) Similar-asset and recent-gainer sources in the pool — dropped: of {sum(src.values())} new May purchases found by the pool, {src['POPULAR']} came from popular assets, {src['SIMILAR']} from similar, {src['RECENT_GAINERS']} from gainers. (c) Few-shot prompts — not used; zero-shot is enough for HW2 and cannot leak evaluation users' purchases.</p>

<h2>3. Method</h2>
<p><b>Approach.</b> Code computes a fact table per client from rows strictly before the cutoff; the LLM encoder re-represents it as cited evidence — "the information stays the same, the representation changes", the encoder role in the course design [11]; a fixed candidate pool is ordered either by a rule cascade or by an LLM reranker. Each ablation cell changes exactly one LLM component.</p>
<div class="pipeline"><div class="box"><b>Rows &lt; cutoff</b><br>trades, prices, profile</div><div class="arr">&rarr;</div><div class="box"><b>Fact table (code)</b><br>Table A, &le;12 own assets</div><div class="arr">&rarr;</div><div class="box"><b>Representation</b><br>cells 1, 3: facts<br>cells 2, 4: encoder JSON</div><div class="arr">&rarr;</div><div class="box"><b>Fixed pool (code)</b><br>&le;10 own + &le;10 popular new</div><div class="arr">&rarr;</div><div class="box"><b>Ordering</b><br>cells 1, 2: cascade<br>cells 3, 4: LLM reranker</div><div class="arr">&rarr;</div><div class="box"><b>Top-10</b><br>validated, scored</div></div>
<table class="small"><tr><th>What the encoder receives (per own asset unless noted)</th><th>How it is computed</th><th>Why, and the evidence behind it</th></tr>
<tr><td>Last buy, last sell</td><td>days before the cutoff</td><td>Latest buy / Latest buy or sell are the strongest provided baselines on validation [2]; recency drives repeat purchases [8].</td></tr>
<tr><td>Buys all / 90 d / 30 d; decayed frequency &Sigma; exp(&minus;age/60)</td><td>whole history</td><td>Repeat frequency carries most next-basket accuracy [8]; the cascade built on it (cell 1) beat Latest buy on all three validation months (Table 5).</td></tr>
<tr><td>First buy; usual gap (median of the last 5 gaps)</td><td>whole history</td><td>Repurchase timing: is the next re-buy due? [8]</td></tr>
<tr><td>Position: holds / sold part / sold all</td><td>at cutoff, initial positions included</td><td>Separates a holding from an exit; needed to read the next row.</td></tr>
<tr><td>Last sale vs average buy price</td><td>each sale</td><td>Investors avoid re-buying what they sold at a loss [7].</td></tr>
<tr><td>Rank in category by distinct bank buyers</td><td>30 days before the client's first buy <i>and</i> before the cutoff</td><td>Attention: individuals buy what others buy [4], [5]. The first-buy value shows whether this client followed the crowd; the cutoff value shows the state now.</td></tr>
<tr><td>7-day spike (&ge;3&times; the weekly mean of the 4 weeks before, &ge;5 buyers)</td><td>same two dates</td><td>Abnormal trading volume attracts individual buyers [4].</td></tr>
<tr><td>Price change over 30 days</td><td>same two dates</td><td>Extreme returns attract attention [4]; smartphone traders buy recent winners [6].</td></tr>
<tr><td>Category, sector; client profile (type, risk level with questionnaire date, capacity); buys in the last 90 d vs the 90 d before; main category</td><td>last record before the cutoff</td><td>Item attributes and profile are encoder inputs in the course design [11]; risk is "unknown at cutoff" if the questionnaire is dated later (back-filled fields).</td></tr></table>
<div class="caption">Table A. Encoder input (one row per asset, at most 12 by decayed frequency). Asset names and ISINs are replaced by random per-client IDs (A10–A99), so the model cannot recall 2021 prices from pre-training.</div>
<p><b>Encoder output.</b> One-line JSON with the schema of the course design [11]: <code>observed_patterns</code>, each citing its <code>supporting_assets</code>, and <code>unknowns</code>; I added <code>asset_status</code>, one label per row, "first that fits": <i>core</i> (&ge;2 buys, last within 90 days, even if sold since — the repeat habit [8]); <i>exited_gain</i> / <i>exited_loss</i> (fully sold, last sale above / below the average buy price [7]); <i>new</i> (first buy within 60 days); <i>lapsed</i> (&ge;2 buys, none in 90 days); <i>one_off</i>. Core comes before the exits because active traders often sell out and buy again within weeks; this order was chosen on Mar–May without any LLM (Table 5). The prompt (<code>prompts/encoder_v3.txt</code>) forbids motives, risk and price forecasts and requires numbers copied from the rows; code checks the JSON, that every row has exactly one valid status and that every cited ID belongs to the client. In cell 2 own assets are sorted by status rank core &gt; new &gt; lapsed &gt; one_off &gt; exited_gain &gt; exited_loss (ties: decayed frequency) — the starter's encoder seam: the LLM reweights past assets.</p>
<p><b>Reranker.</b> It receives the client block (cell 3: Table A; cell 4: the encoder JSON) and the pool in shuffled order, each candidate with its facts; new candidates (bank buyers, rank, spike, price change, client's buys in the sector) look the same in cells 3 and 4. It decides two things the cascade fixes by rule: where the boundary between own and new assets falls, and which new assets go first. The prompt (<code>prompts/reranker_v1.txt</code>) sets the default "own assets first, then new" because repeats dominate [8], and allows a departure only on evidence: a pattern of buying popular new assets [4], [5], or an own asset the client stopped buying or sold [7]. Candidates are shuffled with a fixed seed, because LLM rankers favour what is listed first [10] (a second permutation ran only in the smoke tests). Output: a JSON array of exactly 10 distinct candidate IDs.</p>
<table class="small"><tr><th>Cell</th><th>Client representation</th><th>Ordering of the same pool</th><th>Comparison it enables</th></tr>
<tr><td>1. Rule (control)</td><td>Table A &rarr; decayed frequency</td><td>cascade: own by score, then popular new by buyers</td><td>1 vs Latest buy: pool and rule without LLM</td></tr>
<tr><td>2. Encoder only</td><td>encoder JSON &rarr; status rank</td><td>same cascade</td><td>2 vs 1: encoder effect under classical ordering</td></tr>
<tr><td>3. Reranker only</td><td>Table A</td><td>LLM reranker</td><td>3 vs 1: reranker on raw facts</td></tr>
<tr><td>4. Both (method)</td><td>encoder JSON</td><td>LLM reranker</td><td>4 vs Latest buy: the RQ; 4 vs 3: does evidence help the reranker?</td></tr></table>
<p><b>Pool and invalid outputs.</b> Own assets: top 10 by decayed frequency; new: up to 10 never-traded assets of the client's main asset category by distinct bank buyers in the last 30 days; filled from starter popularity if fewer than 10. The pool is identical in all four cells (fixed candidates, as HW2 requires for a reranker). No retries: an invalid encoder answer makes cell 2 use the cell 1 order and cell 4 send the cell 3 input; an invalid reranker answer is replaced by the cascade of the same representation (cell 3 &rarr; 1, cell 4 &rarr; 2). Failures are counted and no user is dropped.</p>
<p><b>Worked example: parameters, inputs and outputs of both LLM components</b> (user 6BAA2D on the June test, copied from <code>generations.jsonl</code>; inputs abridged: "rank" = rank in category by bank buyers over 30 days, then "spike" if present, then the 30-day price change).</p>
{wx_html}
<div class="caption">The encoder output also shows its typical errors: it disagrees with the written status definitions on {wx_diff} of {wx_n} rows (e.g. A61 "new" although first bought 105 days ago; A75 "exited_loss" although only partly sold) and lists as "unknown" a price change that is in the A94 row. Its reordering still lifted A94; the reranker without evidence dropped all own assets.</div>
<p><b>Key design decisions.</b> (1) Counting is done in code, not by the 4B model: in E1 a model-written profile got recency right for only {100 * e1acc['recency_exact']:.1f}% of assets. (2) Names and ISINs never reach the model; a unit test checks every prompt for ISIN patterns. (3) Status precedence and the pool were fixed on Mar–May before any LLM run; only the encoder's output format changed afterwards (&sect;5).</p>
<p><b>Tools &amp; configuration.</b> Qwen/Qwen3-4B-Instruct-2507, revision cdbee75f17c01a7cc42f958dc650907174af0554 [12], [13]; Python 3.13.1, torch 2.11.0 (MPS, float16), transformers 4.57.6, tokenizers 0.22.2, safetensors 0.8.0; greedy decoding (no sampling seed needed), thinking off, context 8192, max_new_tokens encoder 1000 / reranker 100; shuffle seed = sha256(user|cutoff|permutation). Starter, <code>data_adapter.py</code> and <code>runtime.py</code> are the course files unchanged (commit f685086 of [2]); own code: <code>hybrid.py</code>, <code>test_hybrid.py</code>, prompts, configs, <code>analysis/</code>. Run: <code>python hybrid.py --config configs/hybrid_test_june.json</code>.</p>

<h2>4. Experiments</h2>
<p><b>Setup.</b> Course defaults: 100 deterministic warm users, cutoff 2021-06-01, 30-day window, repeats allowed, binary nDCG@10 with empty truth = 0. <code>protocol_hash</code> {escape(rt['protocol_hash'][:12])} equals the starter baseline run (full hash in <code>protocol.json</code>). Hardware: Apple-silicon Mac, MPS GPU <span class="fill">[chip, RAM]</span>. One test run on 2026-11-19, {wall_min:.0f} min, {enc_calls} encoder + {2 * enc_calls} reranker calls; encoder output {diag['output_tokens']['encoder']['mean']:.0f} tokens on average (max {diag['output_tokens']['encoder']['max']}).</p>
{results_table()}
<div class="caption">Table 1. Course test, 100 users. Runtime = summed LLM-call time (cell 4 includes the encoder); baselines and cell 1 run in seconds inside the whole run ({rt['total_wall_seconds']:.0f} s), not timed separately. Failed cases: {fails.get('encoder', 0)} invalid encoder answers (an ID outside the rows; a status outside the list) and {fails.get('reranker_c4_p1', 0)} reranker answers with duplicate IDs in cell 4; all replaced by the fallback order and kept.</div>
<p><b>Comparison vs baseline.</b> The full hybrid (cell 4) is below Latest buy by {abs(c4lb[0]):.4f} on the 11 users with purchases (better for {c4lb[3]}, worse for {c4lb[4]}, sign test p = {sign_p(c4lb[3], c4lb[4]):.2f}); the interval is wide but centred below zero, so the answer to the RQ is <b>no</b>: no improvement was shown, though a loss is not proven either. {len(buyers) - len(in_pool_buyers)} of the 11 bought nothing in the pool and score 0 in every cell, so the four cells differ on at most {len(in_pool_buyers)} users. In this sample Popularity, not Latest buy, is the strongest provided baseline, as also reported in [2]. Cell 2 is above all four baselines and both course LLM baselines.</p>
<p><b>Verification done by the AI tool (not by me).</b> (a) nDCG@10 for all 8 methods was recomputed from <code>predictions.csv</code> and raw June transactions with independent code; all 16 means match <code>metrics.json</code> to 6 decimals. (b) <code>protocol_hash</code> matches the starter run; the repo copy of the code on the repo's copy of the data reproduces the baselines and cell 1 exactly, and re-running <code>hybrid_headroom.py</code> reproduces the May pool analysis (one value differs in the 4th decimal). (c) 8 unit tests pass (parsers, status rule, a fake-model run, no rows on or after the cutoff, no ISINs or names in prompts). (d) Encoder statuses agree with the deterministic reading of the prompt's definitions for {agree['agree']} of {agree['agree'] + agree['differ']} rows ({agree_pct:.0f}%).</p>
<div class="pending"><b>My own check.</b> [Write what you checked yourself and the outcome, e.g.: in <code>predictions.csv</code> user 6BAA2D5FBCCEAB57C473, method <code>hyb_c2_encoder</code>, rank 2 is GRS419003009; in <code>transactions.csv</code> this user's only June-2021 Buy is GRS419003009; with one relevant item IDCG = 1 and DCG = 1/log2(2+1) = 0.6309, matching 0.631 in Table 3.]</div>
<div class="two"><div>{contrast_table()}
<div class="caption">Table 2. Paired differences over the 11 users with purchases (10,000 bootstrap resamples, seed 0). Only "2 vs 1" excludes zero; with n = 11 this describes the test, it does not prove an effect.</div></div>
<div>{buyer_table()}
<div class="caption">Table 3. Users with purchases (first 6 ID characters); "in pool (new)": purchases in the pool (never traded before); &dagger; invalid encoder answer, fallback used. The pool held {100 * diag['pool_recall_buyers']:.0f}% of these purchases (about 80% on Mar–May), so a perfect order of the pool would reach {diag['oracle_pool_ndcg_buyers']:.4f}.</div></div></div>
<p><b>4.4 Earlier attempts and no-LLM analyses.</b> Table 4: the two earlier LLM experiments behind this design (&sect;5). Table 5: no-LLM checks on validation cutoffs whose windows end before June (3,000 warm users each; scored on buyers, as non-buyers score 0 for every method).</p>
{earlier_table()}
<div class="caption">Table 4. E1 ran on the course test cutoff before the hybrid was designed, so June was not untouched for the design. E2 was stopped to free the Mac for the hybrid and is scored from its saved answers (<code>analysis/exp2_partial_scores.py</code>); "new caught": new purchases placed in the new-asset slots, whose number came from a propensity model (holdout AUC {prop['holdout_auc']:.2f}).</div>
{validation_table()}
<div class="caption">Table 5. Validation cutoffs, nDCG@10 over buyers, no LLM (<code>analysis/prelim_validation.py</code>, <code>fusion_validation.py</code>). Rank fusion: reciprocal-rank fusion of Latest buy, cell 1, Popularity and CBF; "best of 4": the better list per user with hindsight (an upper bound, not a method).</div>


<h2>5. Discussion</h2>
<p><b>Failure case.</b> In the first smoke test (3 May buyers) the encoder failed for 2 of 3 users: <code>encoder valid False JSON error: Expecting ',' delimiter</code>, with <code>output_tokens</code> mean {s1['diagnostics']['output_tokens']['encoder']['mean']} / max {s1['diagnostics']['output_tokens']['encoder']['max']} at <code>max_new_tokens=800</code>. <b>Root cause:</b> the model wrote indented JSON and long interpretive patterns ("indicating a preference for profitable exits"), so generation hit the limit and the object was cut. <b>Fix + check:</b> prompt v2 asked for one-line JSON, &le;5 short patterns and no interpretation (limit 1000): output fell to mean {s2['diagnostics']['output_tokens']['encoder']['mean']} / max {s2['diagnostics']['output_tokens']['encoder']['max']} tokens, but 1 of 3 answers skipped a row; v3 lists the required IDs in the prompt and the third smoke run had {sum(s3['failures_by_stage'].values())} failures; on the test 2 of 100 encoder answers were invalid. The reranker prompt never changed, and nothing changed after the test run.</p>
<p><b>Earlier attempts that did not work.</b> <i>E1</i> followed Session 1: the LLM ranked 50 fixed candidates from the raw dated history (R) or from a profile it first wrote itself (S). No LLM arm beat the frequency–recency control ({f4(e1['freq_recency']['ndcg_at_10_nonempty'])} vs at best {f4(e1[e1best]['ndcg_at_10_nonempty'])} on buyers), and 8–16 of 100 answers were invalid. Three findings shaped the hybrid: the self-written profile was often wrong (buy count right for {100 * e1acc['times_bought_exact']:.1f}% of assets), yet the same profile computed by code (C) gave the worst arm, so ranking, not the profile, was the bottleneck; shuffled dates (Rp) did not hurt, so the model barely used time; and own assets by frequency were the real signal — hence facts in code and an own-first cascade. <i>E2</i> kept the client's assets first and let the LLM choose which popular never-traded assets fill the slots for new ones. On {e2['users_done']} users the LLM ({f4(e2['methods']['explore_llm']['ndcg_buyers'])}) stayed below the same slots filled by popularity ({f4(e2['methods']['explore_control_popularity']['ndcg_buyers'])}), and hiding asset names did not hurt ({f4(e2['methods']['explore_llm_masked_names']['ndcg_buyers'])}), so the model added no useful knowledge about companies; this moved the LLM's role from choosing new assets to reading the client's own history.</p>
<p><b>What worked.</b> (1) The encoder as a re-describer: cell 2 beat cell 1 for all {c21[3]} users whose order changed ({c21[0]:+.4f}), mostly by demoting fully sold assets (A76 for 6BAA2D) and promoting held ones. (2) Validation with fallbacks kept every user ({sum(fails.values())} failures in 300 calls). (3) The same-pool ablation located the loss in reranking, not in the pool or the encoder.</p>
<p><b>What surprised me / did not work.</b> (1) The encoder's gains partly come from <i>breaking</i> our definitions: only {agree_pct:.0f}% of its statuses match them. For 6BAA2D, A76 (sold all, last buy 84 days ago) should be <i>core</i> and A94 (last buy 204 days ago) <i>lapsed</i>; both deviations moved the June purchase up. On Mar–May the written precedence was slightly better (Table 5), so I read this as luck on 11 users. (2) The reranker broke the own-first default for some clients, not for all: over the 100 users it put a new asset first only {new_first} times and kept {100 * statistics.mean(own_top3):.0f}% of the top-3 places for own assets, yet for {dropped_own} of the {len(in_pool_buyers)} buyers with a purchase in the pool it pushed bought own assets out of the top 10 (cell 3 vs 1: {c31[0]:+.4f}); it won only for the 2 users who bought something new. Encoder evidence reduced but did not remove this (4 vs 3: {c43[0]:+.4f}). (3) Methods disagree: cell 2 and Popularity share only {ov['hyb_c2_encoder | popularity']:.1f} of 10 items, and all eight methods together found {overlap['found_by_union_of_all_methods']} of {overlap['bought_assets']} June purchases against {overlap['found_in_top10']['hyb_c2_encoder']} for the best one. With hindsight, picking the better of four no-LLM lists per user would add {gain_lo:+.2f} to {gain_hi:+.2f} on Mar–May, but rank fusion lost to the rule every month (Table 5): it lifts popular assets above the client's repeats.</p>
<p><b>Limitations and ethics.</b> All June conclusions rest on 11 buyers, and June had already been used by E1. nDCG@10 rewards predicting purchases, not suitable or profitable advice [9]; promoting what the crowd buys can amplify trading that hurts retail investors on average [5]. FAR-Trans is pseudonymised and the model ran locally, so no client data left the machine.</p>
<p><b>Next improvement — what we will test in the next iteration.</b> All tuning on Mar–Apr, checks on 100 May buyers, June run once at the end.</p>
<ol class="next">
<li><b>Per-client switch between repeat and explore.</b> Keep the cascade for own assets and learn how many of the 10 slots go to new assets, from E2's propensity features plus encoder evidence. Ceiling: the hindsight best-of-4 gap above ({gain_lo:+.2f} to {gain_hi:+.2f}).</li>
<li><b>Is the encoder effect real?</b> Cells 1 and 2 on 100 May buyers (one encoder call per user, about 1 h): does {c21[0]:+.4f} survive a larger sample?</li>
<li><b>A narrower reranker.</b> Let the LLM only set the own/new boundary or only order the new candidates, never move own assets below new ones.</li>
<li><b>Audit the encoder.</b> Check 20 encoder answers by hand (confirmed / inaccurate / invented patterns) and test, without LLM, rule variants that copy its systematic "errors" (e.g. a recent loss-making exit counted as an exit).</li>
<li><b>Pool recall and profit.</b> Raise pool recall ({100 * diag['pool_recall_buyers']:.0f}% of June purchases, about 80% on Mar–May), e.g. with assets showing a burst of first-time buyers, and report the 30-day return of each top-10 next to nDCG [9].</li>
</ol>
<div class="pending"><b>Team discussion and my contribution.</b> [What the team (Zlata Kranina, Maria, me) discussed and decided for HW2, open questions; what I proposed, implemented, tested or verified myself; which work was shared (e.g. the course starter and baselines) and which is mine.]</div>

<h2>6. AI Usage Disclosure</h2>
<ul>
<li><b>Tools:</b> Claude Opus 5.5 (Anthropic) in the Claude app with file, shell and Mac-bridge tools; Qwen3-4B-Instruct-2507 and Qwen3-1.7B are the models under test, not assistants.</li>
<li><b>How AI was used:</b> literature search and reference checking; designing the facts, statuses and ablation together with me; code generation (<code>hybrid.py</code>, tests, configs, run scripts, <code>analysis/</code>); no-LLM analyses (Table 5, method overlap), scoring the stopped E2 run, tables and bootstrap; drafting this report from our planning document.</li>
<li><b>Checks the AI performed:</b> listed in &sect;4 (recomputed nDCG, protocol hash, reproduction of the pool analysis, unit tests, status agreement); every reference was opened and its title, authors, venue and year checked against the source page on 2026-11-19.</li>
<li class="fill-li"><b>What I personally verified:</b> <span class="fill">[your hand computation / file checks]</span></li>
<li><b>Trusted without verification:</b> the content of the 100 encoder answers (only automatic checks and status agreement); the bootstrap code; &tau; = 60 for decayed frequency from an earlier analysis; the replay of E2 answers on the 77 completed users.</li>
<li><b>Session log:</b> <span class="fill">[file name]</span> — unmodified export of the AI session.</li>
</ul>

<h2>References</h2>
<p class="ref">[1] J. Sanz-Cruzado, N. Droukas, and R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," arXiv:2407.08692, 2024. [Online]. Available: https://arxiv.org/abs/2407.08692. [Accessed: 2026-11-19].</p>
<p class="ref">[2] S. Jin, "RecSys-LLMs: FAR-Trans starter, provided baselines and LLM baselines (HW2_RESULTS.md)," GitHub repository, commit f685086, 2026. [Online]. Available: https://github.com/dryjins/RecSys-LLMs. [Accessed: 2026-11-19].</p>
<p class="ref">[3] H. Liu, C. Peng, W. A. Xiong, and W. Xiong, "Taming the bias zoo," <i>J. Financial Economics</i>, vol. 143, no. 2, pp. 716–741, 2022. [Online]. Available: https://econpapers.repec.org/RePEc:eee:jfinec:v:143:y:2022:i:2:p:716-741. [Accessed: 2026-11-19].</p>
<p class="ref">[4] B. M. Barber and T. Odean, "All That Glitters: The Effect of Attention and News on the Buying Behavior of Individual and Institutional Investors," <i>Rev. Financial Studies</i>, vol. 21, no. 2, pp. 785–818, 2008, doi: 10.1093/rfs/hhm079. [Online]. Available: https://econpapers.repec.org/RePEc:oup:rfinst:v:21:y:2008:i:2:p:785-818. [Accessed: 2026-11-19].</p>
<p class="ref">[5] B. M. Barber, X. Huang, T. Odean, and C. Schwarz, "Attention-Induced Trading and Returns: Evidence from Robinhood Users," <i>J. Finance</i>, vol. 77, no. 6, pp. 3141–3190, 2022, doi: 10.1111/jofi.13183. [Online]. Available: https://econpapers.repec.org/RePEc:bla:jfinan:v:77:y:2022:i:6:p:3141-3190. [Accessed: 2026-11-19].</p>
<p class="ref">[6] A. Kalda, B. Loos, A. Previtero, and A. Hackethal, "Smart(Phone) Investing? A within Investor-time Analysis of New Technologies and Trading Behavior," NBER Working Paper 28363, 2021. [Online]. Available: https://econpapers.repec.org/RePEc:nbr:nberwo:28363. [Accessed: 2026-11-19].</p>
<p class="ref">[7] M. A. Strahilevitz, T. Odean, and B. M. Barber, "Once Burned, Twice Shy: How Naïve Learning, Counterfactuals, and Regret Affect the Repurchase of Stocks Previously Sold," <i>J. Marketing Research</i>, vol. 48, no. SPL, pp. S102–S120, 2011, doi: 10.1509/jmkr.48.SPL.S102. [Online]. Available: https://digitalcommons.stmarys-ca.edu/school-economics-business-faculty-works/1049. [Accessed: 2026-11-19].</p>
<p class="ref">[8] M. Li, S. Jullien, M. Ariannezhad, and M. de Rijke, "A Next Basket Recommendation Reality Check," <i>ACM Trans. Inf. Syst.</i>, vol. 41, no. 4, Art. 116, 2023, doi: 10.1145/3587153. [Online]. Available: https://arxiv.org/abs/2109.14233. [Accessed: 2026-11-19].</p>
<p class="ref">[9] J. Sanz-Cruzado, R. McCreadie, N. Droukas, C. Macdonald, and I. Ounis, "Investors Are (Not) Always Right: A Comparison of Transaction-Based and Profitability-Based Metrics for Financial Asset Recommendations," <i>ACM Trans. Inf. Syst.</i>, 2025, author version. [Online]. Available: https://javiersanzcruza.github.io/assets/papers/tois2025.pdf. [Accessed: 2026-11-19].</p>
<p class="ref">[10] Y. Hou, J. Zhang, Z. Lin, H. Lu, R. Xie, J. McAuley, and W. X. Zhao, "Large Language Models are Zero-Shot Rankers for Recommender Systems," in <i>Proc. ECIR</i>, 2024. [Online]. Available: https://arxiv.org/abs/2305.08845. [Accessed: 2026-11-19].</p>
<p class="ref">[11] S. Jin, "LLM4Recsys 08. LLM-Based Recommendation and FAR-Trans Project," lecture slides, LLM4Rec course, HSE University, 2026. [Online]. Available: https://docs.google.com/presentation/d/1qwwA0x_MD3N65-qAN1Ew5vnS6_iTDDFAXULuChNWqlE. [Accessed: 2026-11-19].</p>
<p class="ref">[12] A. Yang et al., "Qwen3 Technical Report," arXiv:2505.09388, 2025. [Online]. Available: https://arxiv.org/abs/2505.09388. [Accessed: 2026-11-19].</p>
<p class="ref">[13] Qwen, "Qwen3-4B-Instruct-2507," Hugging Face model card. [Online]. Available: https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507. [Accessed: 2026-11-19].</p>
</body></html>
"""
OUT_HTML.write_text(html, encoding='utf-8')
print('written', OUT_HTML)
print('buyers', len(buyers), 'c4-LB', c4lb, 'c2-c1', c21, 'c4-c3', c43)
