"""HW3 shared pieces: cohorts, frozen protocol, the new-asset block and list assembly.

Reuses the HW2 code in ../../week8/far-trans-llm (starter loader, Market, client facts) unchanged.
Everything a method reads is dated strictly before the cutoff; truth is read only for evaluation.
Teaching experiment on historical data, not investment advice.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FTL = (HERE / '..' / '..' / 'week8' / 'far-trans-llm').resolve()
DATA = (HERE / '..' / '..' / 'week8' / 'Data FAR-Trans').resolve()
if str(FTL) not in sys.path:
    sys.path.insert(0, str(FTL))

import data_adapter  # noqa: E402  (course file, unchanged)
import hybrid  # noqa: E402  (HW2 file, unchanged)

FACT_CFG = {'decay_tau': 60.0, 'own_max': 10, 'popular_new_max': 10}
BLOCK_SIZE = 20


def load_cohort(cohort, output_dir):
    """cohort: {'name', 'cutoff', 'max_users', 'active_days' (None = no filter)}.

    Users = the starter's deterministic warm order, first `max_users`, then (optionally) only those
    with at least one Buy in the `active_days` before the cutoff. The rule reads past rows only.
    """
    args = hybrid.parse_args(['--data-dir', str(DATA), '--output-dir', str(output_dir),
                              '--cutoff', cohort['cutoff'], '--max-users', str(cohort['max_users']),
                              '--no-llm'])
    bundle = data_adapter.load(args)
    starter, assets, _h, truth, contexts, baseline, _e, pop, manifest = bundle
    warm = list(manifest['users'])
    market = hybrid.Market(args.data_dir, cohort['cutoff'], warm)
    market.assert_no_future()
    facts = {u: hybrid.client_facts(u, market, assets, FACT_CFG) for u in warm}
    days = cohort.get('active_days')
    if days:
        users = [u for u in warm
                 if min((a['last_buy_days_ago'] for a in facts[u]['assets'].values()), default=10 ** 9) <= days]
    else:
        users = warm
    manifest['users'] = users
    protocol = hybrid.protocol_for(args, assets, users, manifest)
    hw3 = dict(protocol)
    hw3.pop('protocol_hash')
    hw3['hw3_selection'] = {'name': cohort['name'], 'first_warm_users': cohort['max_users'],
                            'active_days_before_cutoff': days,
                            'rule_reads': 'past Buy rows only (no truth, no scores)'}
    hw3['hw3_block'] = {'own': 'pool own (top 10 by decayed frequency, tau 60) in rule order',
                        'new_block': f'top {BLOCK_SIZE} never-traded assets of the main category by '
                                     'distinct bank buyers in the 30 days before the cutoff',
                        'slots': '10 - number of own assets; rest filled from starter popularity'}
    protocol['hw3_protocol_hash'] = hashlib.sha256(json.dumps(hw3, sort_keys=True).encode()).hexdigest()
    protocol['hw3_selection'] = hw3['hw3_selection']
    protocol['hw3_block'] = hw3['hw3_block']
    return {'args': args, 'bundle': bundle, 'market': market, 'facts': facts, 'users': users,
            'protocol': protocol, 'cohort': cohort}


def blocks_for(c):
    """Per user: own (rule order), the new block (initial = popularity order), slots, fills."""
    starter, assets, _h, truth, contexts, baseline, _e, pop, manifest = c['bundle']
    market = c['market']
    popo = [i for i in starter.order(pop) if i in assets]
    cat = hybrid.category_of(assets)
    table = market.ranks(market.cutoff, cat, list(assets))
    out = {}
    for u in c['users']:
        f = c['facts'][u]
        A = f['assets']
        pool = hybrid.build_pool(f, market, assets, FACT_CFG, popo)
        own = sorted(pool['own'], key=lambda i: (-A[i]['decayed_frequency'], A[i]['last_buy_days_ago'], i))
        new = [i for i in assets if i not in f['touched'] and cat(i) == f['main_category']
               and table.get(i, (None, 0))[1] > 0]
        new.sort(key=lambda i: (table[i][0], i))
        new = new[:BLOCK_SIZE]
        out[u] = {'own': own, 'block': new, 'slots': max(0, 10 - len(own)),
                  'main_category': f['main_category'], 'popo': popo}
    return out


def compose(b, block_order):
    """Final top-10: own assets, then the block in the given order, then starter popularity."""
    order = list(block_order) + [i for i in b['block'] if i not in block_order]
    return hybrid.fill10(b['own'] + order, b['popo'])


def ndcg(starter, ranking, rel):
    return starter.ndcg10(ranking, rel)
