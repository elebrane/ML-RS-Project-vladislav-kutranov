"""Can combining methods help? Reciprocal-rank fusion of no-LLM methods on validation cutoffs.

Validation cutoffs only (2021-03/04/05; windows end before June). 3,000 warm users per cutoff, scored on
buyers (non-buyers score 0 for every method). 'oracle_best_of' picks, per user, the better list with
hindsight: an upper bound, not a method.
Usage (from far-trans-llm): PYTHONPATH=. python analysis/fusion_validation.py OUT.json
"""
import json
import statistics
import sys
from pathlib import Path

import data_adapter
import hybrid


def rrf(lists, k=60):
    score = {}
    for lst in lists:
        for r, i in enumerate(lst):
            score[i] = score.get(i, 0.0) + 1.0 / (k + r + 1)
    return [i for i, _ in sorted(score.items(), key=lambda x: (-x[1], x[0]))][:10]


def run(cutoff):
    args = hybrid.parse_args(['--data-dir', '../Data FAR-Trans', '--output-dir', '/tmp/fusion',
                              '--cutoff', cutoff, '--max-users', '3000', '--no-llm'])
    starter, assets, _h, truth, _c, base, _e, pop, manifest = data_adapter.load(args)
    users = [u for u in manifest['users'] if truth.get(u)]
    market = hybrid.Market(args.data_dir, cutoff, users)
    cfg = {'decay_tau': 60.0, 'own_max': 10, 'popular_new_max': 10}
    popo = [i for i in starter.order(pop) if i in assets]
    res = {}
    for u in users:
        f = hybrid.client_facts(u, market, assets, cfg)
        pool = hybrid.build_pool(f, market, assets, cfg, popo)
        A = f['assets']
        c1 = hybrid.fill10(sorted(pool['own'], key=lambda i: (-A[i]['decayed_frequency'],
                                  A[i]['last_buy_days_ago'], i)) + pool['new'] + pool['fill'], popo)
        L = {'latest_buy': base[u]['latest_buy'], 'popularity': base[u]['popularity'],
             'cbf': base[u]['cbf'], 'cell1': c1}
        L['rrf(cell1,popularity)'] = hybrid.fill10(rrf([c1, L['popularity']]), popo)
        L['rrf(latest_buy,popularity)'] = hybrid.fill10(rrf([L['latest_buy'], L['popularity']]), popo)
        L['rrf(all four)'] = hybrid.fill10(rrf([L['latest_buy'], c1, L['popularity'], L['cbf']]), popo)
        s = {m: starter.ndcg10(v, truth[u]) for m, v in L.items()}
        s['oracle_best_of(cell1,popularity)'] = max(s['cell1'], s['popularity'])
        s['oracle_best_of(all four)'] = max(s['latest_buy'], s['cell1'], s['popularity'], s['cbf'])
        s['common_items(cell1,popularity)'] = len(set(c1) & set(L['popularity']))
        s['common_items(latest_buy,popularity)'] = len(set(L['latest_buy']) & set(L['popularity']))
        for m, v in s.items():
            res.setdefault(m, []).append(v)
    out = {m: round(statistics.mean(v), 4) for m, v in res.items()}
    out.update(cutoff=cutoff, buyers=len(users))
    return out


if __name__ == '__main__':
    rows = [run(c) for c in ('2021-03-01', '2021-04-01', '2021-05-01')]
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(json.dumps(r))
