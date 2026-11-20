"""No-LLM preliminary experiments on validation cutoffs (all windows end before June 2021).

For each cutoff: 3,000 warm users (starter order), users with purchases only (empty truth scores 0
for every method). Reports nDCG@10 of Latest buy, Latest buy or sell, cell 1 (decayed-frequency
cascade) and two rule-status cascades (core-first vs exited-first precedence), plus pool recall and
the nDCG of a perfect ordering of the pool.
Usage (from far-trans-llm): PYTHONPATH=. python analysis/prelim_validation.py OUT.json
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import data_adapter
import hybrid

RANK = ['core', 'new', 'lapsed', 'one_off', 'exited_gain', 'exited_loss']


def exited_first(f):
    if f['position'] == 'sold all':
        return 'exited_gain' if (f['last_sale_vs_avg_buy'] or 0) > 0 else 'exited_loss'
    return hybrid.rule_status(f)          # same definitions otherwise


def run(cutoff):
    args = hybrid.parse_args(['--data-dir', '../Data FAR-Trans', '--output-dir', '/tmp/prelim',
                              '--cutoff', cutoff, '--max-users', '3000', '--no-llm'])
    starter, assets, _h, truth, _ctx, base, _e, pop, manifest = data_adapter.load(args)
    users = [u for u in manifest['users'] if truth.get(u)]
    market = hybrid.Market(args.data_dir, cutoff, users)
    market.assert_no_future()
    cfg = {'decay_tau': 60.0, 'own_max': 10, 'popular_new_max': 10}
    popo = [i for i in starter.order(pop) if i in assets]
    rk = {s: k for k, s in enumerate(RANK)}
    res = {k: [] for k in ('latest_buy', 'latest_buy_or_sell', 'cell1', 'core_first', 'exited_first',
                           'pool_recall', 'oracle_pool')}
    for u in users:
        f = hybrid.client_facts(u, market, assets, cfg)
        pool = hybrid.build_pool(f, market, assets, cfg, popo)
        A, tail, rel = f['assets'], pool['new'] + pool['fill'], truth[u]
        allp = pool['own'] + tail
        c1 = sorted(pool['own'], key=lambda i: (-A[i]['decayed_frequency'], A[i]['last_buy_days_ago'], i))
        res['cell1'].append(starter.ndcg10(hybrid.fill10(c1 + tail, popo), rel))
        for name, fn in (('core_first', hybrid.rule_status), ('exited_first', exited_first)):
            own = sorted(pool['own'], key=lambda i: (rk[fn(A[i])], -A[i]['decayed_frequency'],
                                                     A[i]['last_buy_days_ago'], i))
            res[name].append(starter.ndcg10(hybrid.fill10(own + tail, popo), rel))
        res['latest_buy'].append(starter.ndcg10(base[u]['latest_buy'], rel))
        res['latest_buy_or_sell'].append(starter.ndcg10(base[u]['latest_buy_or_sell'], rel))
        res['pool_recall'].append(len(rel & set(allp)) / len(rel))
        res['oracle_pool'].append(starter.ndcg10(hybrid.fill10([i for i in allp if i in rel] + allp, popo), rel))
    out = {k: round(statistics.mean(v), 4) for k, v in res.items()}
    out.update(cutoff=cutoff, warm_users=len(manifest['users']), buyers=len(users))
    return out


if __name__ == '__main__':
    rows = [run(c) for c in ('2021-03-01', '2021-04-01', '2021-05-01')]
    Path(sys.argv[1]).write_text(json.dumps(rows, indent=2) + '\n')
    for r in rows:
        print(r)
