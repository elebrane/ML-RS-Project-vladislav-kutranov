"""How much room does the four-source pool leave above the provided baselines? No LLM involved.

Validation cutoffs only (the June test month is never read for this). Users: buyers among the first
--users warm users of the starter's deterministic order. Non-buyers score 0 for every method, so the
ordering of methods on all users equals their ordering on buyers; all-user means are buyer means
times the buyer share.
  oracle_pool       perfect order of the whole four-source pool: ceiling for any reranker of it
  oracle_a2_top20   perfect order of the 20 best A2 candidates: ceiling for the A5-style reranker
"""
from __future__ import annotations

import json
import statistics
from collections import Counter

import data_adapter
import hybrid_v1 as hybrid  # the four-source pool of the first hybrid design


def main():
    p = data_adapter.arguments('Headroom of the hybrid pool (no LLM)')
    p.add_argument('--users', type=int, default=1000)
    args = p.parse_args()
    args.max_users = args.users
    bundle = data_adapter.load(args)
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = bundle
    users = [u for u in manifest['users'] if truth.get(u)]
    market = hybrid.Market(args.data_dir, args.cutoff, users)
    fb = starter.order(pop)
    rows = []
    for u in users:
        elig = set(contexts[u]['eligible_asset_ids'])
        tr = truth[u] & elig
        if not tr:
            continue
        f = hybrid.client_facts(u, market, assets, elig)
        pool = hybrid.build_pool(u, f, market, assets, history, contexts[u])
        ids = [e['isin'] for e in pool]
        o1 = hybrid.rank_by_weights(pool, hybrid.LEX)
        o2 = hybrid.rank_by_weights(pool, hybrid.weights_of(hybrid.formula_weights(f)))
        own = set(history.get(u, {}))
        methods = {k: v for k, v in baseline[u].items()}
        methods['A1_rule'] = hybrid.fill10(o1, fb)
        methods['A2_formula'] = hybrid.fill10(o2, fb)
        methods['oracle_pool'] = hybrid.fill10([i for i in ids if i in tr] + ids, fb)
        methods['oracle_a2_top20'] = hybrid.fill10([i for i in o2[:20] if i in tr] + o2, fb)
        r = {m: starter.ndcg10(rk, tr) for m, rk in methods.items()}
        new_tr = tr - own
        r.update(n_own=len(own), n_truth=len(tr), has_repeat=bool(tr & own), has_new=bool(new_tr),
                 new_in_pool=bool(new_tr & set(ids)),
                 new_sources=sorted({e['source'] for e in pool if e['isin'] in new_tr}))
        rows.append(r)
    names = [k for k in rows[0] if isinstance(rows[0][k], float)]
    out = {'cutoff': args.cutoff, 'warm_users': len(manifest['users']), 'buyers': len(rows),
           'buyer_share': round(len(rows) / len(manifest['users']), 3),
           'mean_ndcg_buyers': {m: round(statistics.mean(r[m] for r in rows), 4) for m in names}}
    out['mean_ndcg_all_users'] = {m: round(v * out['buyer_share'], 4) for m, v in out['mean_ndcg_buyers'].items()}
    for grp, sel in (('only_repeats', lambda r: not r['has_new']), ('bought_new', lambda r: r['has_new'])):
        sub = [r for r in rows if sel(r)]
        out[f'mean_ndcg_{grp}'] = {'n': len(sub), **{m: round(statistics.mean(r[m] for r in sub), 4)
                                                      for m in names}} if sub else {'n': 0}
    nb = [r for r in rows if r['has_new']]
    out['bought_new_share'] = round(len(nb) / len(rows), 3)
    out['new_purchase_found_in_pool'] = round(sum(r['new_in_pool'] for r in nb) / max(1, len(nb)), 3)
    out['new_found_by_source'] = dict(Counter(s for r in nb for s in r['new_sources']))
    out['own_assets_ge_10'] = round(sum(r['n_own'] >= 10 for r in rows) / len(rows), 3)
    best_base = max(('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'),
                    key=lambda m: out['mean_ndcg_buyers'][m])
    out['A2_vs_best_baseline'] = {'baseline': best_base, **dict(Counter(
        'win' if r['A2_formula'] > r[best_base] + 1e-9 else ('loss' if r['A2_formula'] < r[best_base] - 1e-9 else 'tie')
        for r in rows))}
    path = args.output_dir / f'headroom_{args.cutoff}.json'
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
