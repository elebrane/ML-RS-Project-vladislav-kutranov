"""HW3 main experiment: a zero-shot LLM ranker fills the new-asset slots, with and without news.

Fixed in every arm: users, cutoff, the client's own-asset block (rule order), the number of new
slots, the 20-asset new block, its initial (popularity) order, the shown order of candidates and
their IDs, the model and decoding. Arms differ only in what the candidate rows contain:

  code (no LLM)   c0_popularity   initial order = matched control (= HW2 rule cascade, cell 1)
                  cnews_attention block by news attention vs the asset's usual level (primary code control)
                  cnews_volume    block by articles in the last 30 days (exploratory)
                  cnews_tone      block by average tone in the last 30 days (exploratory)
  LLM             llm_nonews      base rows only                                  (ablation A1)
                  llm_numbers     + news numbers, no headlines                    (ablation A2)
                  llm_full        + news numbers + latest headlines               (main method M)
                  llm_shuffled    the llm_full news lines moved to other stocks   (placebo A3)
  diagnostics on a subset (not in the main table):
                  llm_full_order2 llm_full with a second shown order               (position sensitivity)
                  llm_full_names  llm_full with names shown and headlines unmasked (name-recall check)

Scope ("llm_scope": "stock_main"): the LLM is called only for clients whose main category is Stock
(the news layer covers listed stocks only; the rule reads past data only); other clients keep C0.
An invalid answer keeps the valid IDs it contains (repair) or falls back to the initial order;
every repair and fallback is counted and no user is dropped. Teaching experiment, not advice.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import common
import news

hybrid = common.hybrid
data_adapter = common.data_adapter
HERE = common.HERE
CODE_ARMS = ('c0_popularity', 'cnews_attention', 'cnews_volume', 'cnews_tone')
LLM_ARMS = {'llm_nonews': {'news': None}, 'llm_numbers': {'news': 'numbers'}, 'llm_full': {'news': 'full'},
            'llm_shuffled': {'news': 'full', 'shuffle': True}}
SUBSET_ARMS = {'llm_full_order2': {'news': 'full', 'order': 'show2'},
               'llm_full_names': {'news': 'full', 'names': True}}
NEWS_ARMS = {'cnews_attention', 'cnews_volume', 'cnews_tone', 'llm_numbers', 'llm_full', 'llm_shuffled',
             'llm_full_order2', 'llm_full_names'}
CONTRASTS = [  # (a, b, question); fixed before any LLM run
    ('llm_full', 'llm_nonews', 'A1: effect of news for the LLM (H1)'),
    ('llm_full', 'llm_numbers', 'A2: headlines beyond the numbers'),
    ('llm_full', 'llm_shuffled', 'A3: news about the right asset vs placebo'),
    ('llm_full', 'c0_popularity', 'method vs matched control'),
    ('llm_nonews', 'c0_popularity', 'LLM reordering without news vs control'),
    ('llm_numbers', 'cnews_attention', 'H2: LLM vs code on the same numbers'),
    ('llm_full', 'cnews_attention', 'H2: LLM with news vs code with news'),
    ('cnews_attention', 'c0_popularity', 'news signal in code (go/no-go)'),
    ('llm_full', 'latest_buy', 'reference: vs best provided baseline'),
]
ID_LINE = re.compile(r'^(A\d{2}) \|', re.M)


# ------------------------------------------------------------------ prompt pieces
def pct(x):
    return 'unknown' if x is None else f'{100 * x:+.1f}%'


def client_block(f):
    sectors = ', '.join(f'{s} {n}' for s, n in f['sector_buys'].most_common(5)) or 'none'
    return (f"- Main asset category: {f['main_category']}; {len(f['assets'])} different assets bought before.\n"
            f"- Buys in the last 90 days: {f['buys_90d']}; in the 90 days before that: {f['buys_prev_90d']}.\n"
            f"- Sectors of past buys (number of buys): {sectors}.\n"
            f"- Profile: {f['profile']}.")


def base_row(cid, isin, assets, state, names=False):
    a = assets[isin]
    cat = a.get('assetCategory') or 'Unknown'
    sector = a.get('sector') or a.get('assetSubCategory') or 'unknown'
    name = f" | name {a.get('assetName')}" if names else ''
    rank = f"rank {state['rank']} in {cat}" if state['rank'] else 'not ranked'
    return (f"{cid} | {cat} | sector {sector}{name} | bank clients who bought it in the last 30 days: "
            f"{state['buyers']} ({rank}) | price change over 30 days: {pct(state['change30'])}")


def derangement(items, rnd):
    if len(items) < 2:
        return list(items)
    while True:
        p = list(items)
        rnd.shuffle(p)
        if all(a != b for a, b in zip(p, items)):
            return p


def build_prompt(template, arm_cfg, u, b, f, ids, shown, assets, states, cards, field, cutoff):
    names = arm_cfg.get('names', False)
    mode = arm_cfg.get('news')
    lines_of = {}
    for i in shown:
        if mode is None or cards.get(i) is None:
            lines_of[i] = []
        else:
            lines_of[i] = news.card_lines(cards[i], field.aliases, b['block'], headlines=(mode == 'full'),
                                          unmask=names)
    if arm_cfg.get('shuffle'):
        stocks = [i for i in shown if cards.get(i) is not None]
        moved = derangement(stocks, random.Random(hybrid.seed_of(u, cutoff, 'shuffle')))
        orig = dict(lines_of)
        for dst, src in zip(stocks, moved):
            lines_of[dst] = orig[src]
    rows = []
    for i in shown:
        rows.append(base_row(ids[i], i, assets, states[i], names))
        rows += lines_of[i]
    k = min(b['slots'], len(b['block']))
    return hybrid.fill(template, slots=k, client_block=client_block(f), candidate_rows='\n'.join(rows)), k


def parse_choice(text, allowed, k):
    text = text or ''
    ids, how = None, 'json'
    s, e = text.find('['), text.rfind(']')
    if 0 <= s < e:
        try:
            arr = json.loads(text[s:e + 1])
            if isinstance(arr, list):
                ids = [x for x in arr if isinstance(x, str)]
        except json.JSONDecodeError:
            ids = None
    if ids is None:
        ids, how = re.findall(r'A\d{2}', text), 'regex'
    valid = [x for x in dict.fromkeys(ids) if x in allowed]
    if not valid:
        return None, 'fallback'
    status = 'ok' if how == 'json' and len(ids) == k and len(valid) == k else 'repaired'
    return valid[:k], status


class FakeGenerator:
    """Stand-in for tests: returns the first k shown IDs (= the shown order)."""
    metadata = {'backend': 'fake'}

    def generate(self, prompt, max_new_tokens=90):
        k = int(re.search(r'fill the remaining (\d+) places', prompt).group(1))
        return {'text': json.dumps(ID_LINE.findall(prompt)[:k]), 'seconds': 0.0, 'output_tokens': 0}


# ------------------------------------------------------------------ statistics
def sign_p(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    p = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * p)


def paired(a, b, n=10000, seed=0):
    d = [x - y for x, y in zip(a, b)]
    rnd = random.Random(seed)
    m = len(d)
    means = sorted(sum(d[rnd.randrange(m)] for _ in range(m)) / m for _ in range(n))
    better, worse = sum(x > 1e-12 for x in d), sum(x < -1e-12 for x in d)
    return {'diff': round(sum(d) / m, 4), 'ci95': [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)],
            'better': better, 'worse': worse, 'sign_p': round(sign_p(better, worse), 4)}


# ------------------------------------------------------------------ run
def run(cfg, generator=None, max_users=None):
    t_all = time.perf_counter()
    cohorts = json.loads((HERE / 'configs' / 'cohorts.json').read_text())
    cohort = cohorts[cfg['cohort']]
    out = HERE / 'outputs' / cfg['cohort'] / cfg['run_name']
    out.mkdir(parents=True, exist_ok=True)
    arms = [a for a in cfg['arms'] if a in CODE_ARMS or a in LLM_ARMS]
    subset_arms = [a for a in cfg.get('subset_arms', []) if a in SUBSET_ARMS]
    unknown = (set(cfg['arms']) - set(arms)) | (set(cfg.get('subset_arms', [])) - set(subset_arms))
    if unknown:
        raise ValueError(f'unknown arms: {sorted(unknown)}')
    if 'c0_popularity' not in arms:
        arms.insert(0, 'c0_popularity')
    t0 = time.perf_counter()
    c = common.load_cohort(cohort, out)
    blocks = common.blocks_for(c)
    starter, assets, _h, truth, contexts, baseline, _e, pop, manifest = c['bundle']
    users = c['users'][:max_users] if max_users else c['users']
    manifest['users'] = users
    protocol = c['protocol']
    frozen_path = HERE / 'outputs' / cfg['cohort'] / 'protocol.json'
    if not max_users and frozen_path.exists():
        frozen = json.loads(frozen_path.read_text())
        if frozen['hw3_protocol_hash'] != protocol['hw3_protocol_hash']:
            raise SystemExit('protocol differs from the frozen outputs/<cohort>/protocol.json')
    need_news = any(a in NEWS_ARMS for a in arms + subset_arms)
    field = cfg.get('_field') or (news.InfoField.load() if need_news else None)
    cutoff = cohort['cutoff']
    market = c['market']
    cat = hybrid.category_of(assets)
    universe = list(assets)
    states = {}
    prep_seconds = time.perf_counter() - t0

    llm_arms = [a for a in arms if a in LLM_ARMS]
    template = (HERE / 'prompts' / cfg.get('prompt', 'block_ranker_v1.txt')).read_text(encoding='utf-8')
    if (llm_arms or subset_arms) and generator is None:
        import self_structured_ranker as ssr
        from runtime import LoggedGenerator
        g = cfg['generation']
        generator = LoggedGenerator(ssr.ModelGenerator(
            g['model'], g['revision'], g.get('device', 'auto'), False, dtype=g.get('dtype', 'float16'),
            temperature=0.0, top_p=1.0, thinking=False, seed=0, context_limit=g.get('context_limit', 8192)),
            HERE / 'outputs' / cfg['cohort'] / 'llm_cache')
    max_new = cfg.get('max_new_tokens', 90)
    subset_n = cfg.get('subset_users', 200)
    subset = [u for u in users if blocks[u]['slots'] > 0 and blocks[u]['main_category'] == 'Stock'][:subset_n]
    subset_set = set(subset) if subset_arms else set()

    preds, details, status = {}, {}, defaultdict(Counter)
    gen_seconds, prompt_chars = Counter(), defaultdict(list)
    news_cov = Counter()
    sub_preds = defaultdict(dict)
    t_loop = time.perf_counter()
    for n_user, u in enumerate(users, 1):
        b, f = blocks[u], c['facts'][u]
        block = b['block']
        P = {'c0_popularity': common.compose(b, block)}
        cards = {}
        if field is not None:
            cards = {i: field.card(i, cutoff) for i in block if field.tracked(i)}
            if b['slots'] > 0:
                news_cov['blocks_with_slots'] += 1
                news_cov['blocks_with_tracked_stocks'] += bool(cards)
                news_cov['tracked_candidates'] += len(cards)
                news_cov['with_articles_30d'] += sum(cd['articles30'] > 0 for cd in cards.values())
                news_cov['with_headlines'] += sum(bool(cd['headlines']) for cd in cards.values())
        pos = {i: k for k, i in enumerate(block)}

        def by(score):
            return sorted(block, key=lambda i: (-score(i), pos[i]))
        if 'cnews_attention' in arms:
            P['cnews_attention'] = common.compose(b, by(lambda i: cards[i]['ratio'] if cards.get(i) else 0.0))
        if 'cnews_volume' in arms:
            P['cnews_volume'] = common.compose(b, by(lambda i: cards[i]['articles30'] if cards.get(i) else 0))
        if 'cnews_tone' in arms:
            P['cnews_tone'] = common.compose(b, by(lambda i: (cards[i]['tone30'] or 0.0) if cards.get(i) else 0.0))
        info = {'slots': b['slots'], 'main_category': b['main_category'], 'block': block, 'llm': {}}
        in_scope = cfg.get('llm_scope', 'all') == 'all' or b['main_category'] == 'Stock'
        todo = [(a, LLM_ARMS[a]) for a in llm_arms] if in_scope else []
        if u in subset_set:
            todo += [(a, SUBSET_ARMS[a]) for a in subset_arms]
        if todo and b['slots'] > 0 and block:
            ids = hybrid.assign_ids(u, cutoff, block)
            isin_of = {v: k for k, v in ids.items()}
            for i in block:
                if i not in states:
                    states[i] = market.state(i, market.cutoff, cat, universe)
            shown = list(block)
            random.Random(hybrid.seed_of(u, cutoff, 'show')).shuffle(shown)
            info['ids'] = {ids[i]: i for i in shown}
            for arm, acfg in todo:
                sh = shown
                if acfg.get('order'):
                    sh = list(block)
                    random.Random(hybrid.seed_of(u, cutoff, acfg['order'])).shuffle(sh)
                prompt, k = build_prompt(template, acfg, u, b, f, ids, sh, assets, states, cards, field, cutoff)
                prompt_chars[arm].append(len(prompt))
                try:
                    resp = generator.generate(prompt, max_new_tokens=max_new)
                    chosen, st = parse_choice(resp['text'], set(isin_of), k)
                    gen_seconds[arm] += resp.get('seconds', 0.0) or 0.0
                    raw = resp['text']
                except (ValueError, RuntimeError) as exc:
                    chosen, st, raw = None, 'fallback', f'{type(exc).__name__}: {exc}'
                status[arm][st] += 1
                ranking = common.compose(b, [isin_of[x] for x in chosen]) if chosen else list(P['c0_popularity'])
                if arm in SUBSET_ARMS:
                    sub_preds[arm][u] = ranking
                else:
                    P[arm] = ranking
                    if arm == 'llm_full' and u in subset_set:
                        sub_preds['llm_full'][u] = ranking
                info['llm'][arm] = {'status': st, 'raw': raw, 'k': k}
        else:
            for arm in llm_arms:
                P[arm] = list(P['c0_popularity'])
                status[arm]['no_call_out_of_scope' if not in_scope else
                            'no_call_no_slots' if b['slots'] == 0 else 'no_call_empty_block'] += 1
        for ranking in P.values():
            starter.validate_ranking(ranking, contexts[u]['eligible_asset_ids'])
        preds[u] = P
        details[u] = info
        if n_user % 100 == 0 or n_user == len(users):
            print(f'[{time.strftime("%H:%M:%S")}] {n_user}/{len(users)} users', flush=True)
    loop_seconds = time.perf_counter() - t_loop
    summary = evaluate(cfg, cohort, arms, subset_arms, subset, users, preds, sub_preds, blocks, c, protocol)
    summary['meta'].update({
        'llm_status': {a: dict(v) for a, v in status.items()},
        'news_coverage': dict(news_cov),
        'prompt_chars': {a: {'mean': round(statistics.mean(v)), 'max': max(v)} for a, v in prompt_chars.items() if v},
        'generator': dict(getattr(generator, 'metadata', {}) or {}) if generator else {'backend': 'none'},
        'prompt': cfg.get('prompt', 'block_ranker_v1.txt'), 'max_new_tokens': max_new,
        'seconds': {'preparation_cohort_and_blocks': round(prep_seconds, 1),
                    'recommendation_loop_wall': round(loop_seconds, 1),
                    'generation_by_arm_incl_cached': {a: round(s, 1) for a, s in gen_seconds.items()},
                    'total_wall': round(time.perf_counter() - t_all, 1)}})
    (out / 'hw3_summary.json').write_text(json.dumps(summary, indent=1) + '\n')
    (out / 'details.json').write_text(json.dumps(details, ensure_ascii=False) + '\n', encoding='utf-8')
    args = c['args']
    args.output_dir = out
    data_adapter.save_results(args, c['bundle'], {u: preds[u] for u in users}, [], 'hw3_news_rag', summary['meta'])
    print_table(summary)
    return summary


def evaluate(cfg, cohort, arms, subset_arms, subset, users, preds, sub_preds, blocks, c, protocol):
    """Truth is read here only."""
    starter, assets, _h, truth, contexts, baseline, _e, pop, manifest = c['bundle']
    methods = ['popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'] + [a for a in arms if a in preds[users[0]]]
    per_user = {m: [] for m in methods}
    groups = {'buyers': [], 'bought_new': [], 'with_slots': []}
    oracle = []
    for u in users:
        rel = truth.get(u, set()) & set(contexts[u]['eligible_asset_ids'])
        allp = dict(baseline[u])
        allp.update(preds[u])
        for m in methods:
            per_user[m].append(starter.ndcg10(allp[m], rel))
        b = blocks[u]
        groups['buyers'].append(bool(rel))
        groups['bought_new'].append(bool(rel - c['facts'][u]['touched']))
        groups['with_slots'].append(b['slots'] > 0)
        oracle.append(starter.ndcg10(common.compose(b, [i for i in b['block'] if i in rel]), rel))

    def mean_where(vals, mask):
        xs = [v for v, m in zip(vals, mask) if m]
        return (round(statistics.mean(xs), 4) if xs else None), len(xs)
    table = {}
    for m in methods + ['oracle_block']:
        vals = oracle if m == 'oracle_block' else per_user[m]
        row = {'all': round(statistics.mean(vals), 4), 'n_all': len(vals)}
        for g, mask in groups.items():
            row[g], row['n_' + g] = mean_where(vals, mask)
        table[m] = row
    contrasts = []
    for a, b2, q in CONTRASTS:
        if a in per_user and b2 in per_user:
            r = {'a': a, 'b': b2, 'question': q, 'all_users': paired(per_user[a], per_user[b2])}
            ba = [x for x, m in zip(per_user[a], groups['buyers']) if m]
            bb = [x for x, m in zip(per_user[b2], groups['buyers']) if m]
            r['buyers'] = paired(ba, bb) if ba else None
            contrasts.append(r)
    diagnostics = {}
    for arm in subset_arms:
        us = [u for u in subset if u in sub_preds.get(arm, {}) and u in sub_preds.get('llm_full', {})]
        if not us:
            continue
        rel = {u: truth.get(u, set()) & set(contexts[u]['eligible_asset_ids']) for u in us}
        a_vals = [starter.ndcg10(sub_preds['llm_full'][u], rel[u]) for u in us]
        b_vals = [starter.ndcg10(sub_preds[arm][u], rel[u]) for u in us]
        taus = [t for t in (hybrid.kendall_tau(sub_preds['llm_full'][u], sub_preds[arm][u]) for u in us) if t is not None]
        diagnostics[arm] = {'users': len(us), 'llm_full_minus_it': paired(a_vals, b_vals),
                            'identical_lists': sum(sub_preds['llm_full'][u] == sub_preds[arm][u] for u in us),
                            'mean_kendall_tau': round(statistics.mean(taus), 4) if taus else None}
    changed = {a: sum(preds[u][a] != preds[u]['c0_popularity'] for u in users)
               for a in arms if a != 'c0_popularity' and a in preds[users[0]]}
    meta = {'run_name': cfg['run_name'], 'cohort': cohort, 'arms': arms, 'subset_arms': subset_arms,
            'subset_users': len(subset) if subset_arms else 0, 'protocol_hash': protocol['protocol_hash'],
            'hw3_protocol_hash': protocol['hw3_protocol_hash'], 'users': len(users),
            'lists_changed_vs_c0': changed}
    return {'table': table, 'contrasts': contrasts, 'subset_diagnostics': diagnostics, 'meta': meta}


def print_table(s):
    m = s['meta']
    print(f"\n{m['run_name']} | {m['cohort']['name']} | users {m['users']} | "
          f"protocol {m['protocol_hash'][:12]} / hw3 {m['hw3_protocol_hash'][:12]}")
    print(f"{'method':18s} {'all':>7s} {'buyers':>7s} {'n':>5s} {'boughtNew':>9s} {'n':>4s}")
    nan = float('nan')
    for name, r in s['table'].items():
        print(f"{name:18s} {r['all']:7.4f} {r['buyers'] if r['buyers'] is not None else nan:7.4f} "
              f"{r['n_buyers']:5d} {r['bought_new'] if r['bought_new'] is not None else nan:9.4f} {r['n_bought_new']:4d}")
    for ct in s['contrasts']:
        x = ct['all_users']
        print(f"  {ct['a']} - {ct['b']}: {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] "
              f"{x['better']}/{x['worse']} p={x['sign_p']:.3f}  ({ct['question']})")
    if m.get('llm_status'):
        print('  llm status:', m['llm_status'])
    if m.get('news_coverage'):
        print('  news coverage:', m['news_coverage'])
    for arm, dg in s.get('subset_diagnostics', {}).items():
        x = dg['llm_full_minus_it']
        print(f"  subset {arm}: users {dg['users']}, identical {dg['identical_lists']}, tau {dg['mean_kendall_tau']}, "
              f"llm_full - it {x['diff']:+.4f} [{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--config', required=True)
    p.add_argument('--fake-llm', action='store_true')
    p.add_argument('--max-users', type=int, default=0)
    a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text())
    run(cfg, FakeGenerator() if a.fake_llm else None, a.max_users or None)


if __name__ == '__main__':
    main()
