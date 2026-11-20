"""Score the stopped Experiment 2 run (explore_slots, cutoff 2021-05-01) on the users it completed.

Rebuilds the deterministic pipeline (own assets, k, shortlist) with explore_slots.prepare and
replays the saved LLM answers from outputs/explore_q4b_val/arm_responses.jsonl; no model is called.
Usage (from far-trans-llm): PYTHONPATH=. python analysis/exp2_partial_scores.py OUT.json
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import explore_slots as es

args = es.parse_args(['--config', 'configs/explore_q4b_val.json'])
bundle = es.data_adapter.load(args)
starter, assets, _h, truth, contexts, base, events, _pop, manifest = bundle
by_customer = es.customer_events(events)
prop = es.fit_propensity(by_customer, args.cutoff, args.horizon_days, args.propensity_start)
users = es.prepare(args, bundle, prop, by_customer)

recs = defaultdict(dict)
for line in Path(args.output_dir / 'arm_responses.jsonl').read_text().splitlines():
    r = json.loads(line)
    recs[r['user_id']][r['arm']] = r
done = [u for u in users if len(recs.get(u, {})) == 4]
seeds = {'llm': args.shuffle_seed, 'llm_masked_names': args.shuffle_seed, 'llm_shuffled_shortlist': args.second_seed}
masks = {'llm': False, 'llm_masked_names': True, 'llm_shuffled_shortlist': False}
scores = defaultdict(list)
invalid = defaultdict(int)
new_hits = defaultdict(int)
new_total = 0
for u in done:
    p, k, rel = users[u], users[u]['k'], truth.get(u, set())
    preds = {'freq_recency_cbf': es.compose(p['own'], p['cbf_new'], max(0, 10 - len(p['own'])), p['fill'])}
    for arm, r in recs[u].items():
        chosen = list(p['shortlist'])
        if arm in seeds and k > 0 and r['ids'] is not None:
            _d, ids = es.payload(p, assets, es.shown_order(p, seeds[arm]), masks[arm], args.holdings_shown)
            isin_of = {v: i for i, v in ids.items()}
            picked = [isin_of[i] for i in r['ids']]
            chosen = picked + [i for i in p['shortlist'] if i not in picked]
        if r.get('valid') is False or r.get('error'):
            invalid[arm] += 1
        rk = es.compose(p['own'], chosen, k, p['fill'])
        preds[es.METHOD[arm]] = rk
        if rel:
            new_hits[es.METHOD[arm]] += len({i for i in rel if i not in p['traded']} & set(rk[10 - k:] if k else []))
    for name in ('popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'):
        preds[name] = base[u][name]
    if rel:
        new_total += len({i for i in rel if i not in p['traded']})
    for name, rk in preds.items():
        scores[name].append(starter.ndcg10(rk, rel))
buyers = [i for i, u in enumerate(done) if truth.get(u)]
out = {'cutoff': args.cutoff, 'users_done': len(done), 'users_planned': len(users),
       'buyers': len(buyers), 'new_assets_bought': new_total, 'invalid_by_arm': dict(invalid),
       'methods': {n: {'ndcg_all': round(sum(v) / len(v), 4),
                       'ndcg_buyers': round(sum(v[i] for i in buyers) / len(buyers), 4),
                       'new_caught': new_hits.get(n)} for n, v in scores.items()}}
Path(sys.argv[1]).write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
