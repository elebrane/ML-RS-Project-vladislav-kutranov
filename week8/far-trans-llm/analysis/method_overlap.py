"""How much do the top-10 lists of different methods overlap, and could combining them help?

Descriptive only. Reads a run's predictions.csv and rebuilds the truth with the starter
(same cutoff, users and window as the run).
Usage (from far-trans-llm): PYTHONPATH=. python analysis/method_overlap.py RUN_DIR CUTOFF MAX_USERS OUT.json
"""
import csv
import itertools
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import data_adapter
import hybrid

run, cutoff, n_users, out_path = Path(sys.argv[1]), sys.argv[2], sys.argv[3], Path(sys.argv[4])
args = hybrid.parse_args(['--data-dir', '../Data FAR-Trans', '--output-dir', '/tmp/overlap',
                          '--cutoff', cutoff, '--max-users', n_users, '--no-llm'])
starter, assets, _h, truth, _c, _b, _e, _p, manifest = data_adapter.load(args)
lists = defaultdict(lambda: defaultdict(list))
for r in csv.DictReader(open(run / 'predictions.csv')):
    lists[r['method']][r['user_id']].append((int(r['rank']), r['ISIN']))
lists = {m: {u: [i for _, i in sorted(v)] for u, v in us.items()} for m, us in lists.items()}
methods = sorted(lists)
users = manifest['users']
buyers = [u for u in users if truth.get(u)]
ndcg = {m: {u: starter.ndcg10(lists[m][u], truth.get(u, set())) for u in users} for m in methods}
overlap = {f'{a} | {b}': round(statistics.mean(len(set(lists[a][u]) & set(lists[b][u])) for u in users), 2)
           for a, b in itertools.combinations(methods, 2)}
hits = {m: {u: len(set(lists[m][u]) & truth[u]) for u in buyers} for m in methods}
pairs = {}
for a, b in itertools.combinations(methods, 2):
    best = statistics.mean(max(ndcg[a][u], ndcg[b][u]) for u in buyers)
    only_a = sum(1 for u in buyers if ndcg[a][u] > ndcg[b][u])
    only_b = sum(1 for u in buyers if ndcg[b][u] > ndcg[a][u])
    pairs[f'{a} | {b}'] = {'oracle_best_of_two_buyers': round(best, 4), 'a_better': only_a, 'b_better': only_b}
oracle_all = statistics.mean(max(ndcg[m][u] for m in methods) for u in buyers)
union_found = sum(len(truth[u] & set().union(*(lists[m][u] for m in methods))) for u in buyers)
out = {'run': run.name, 'users': len(users), 'buyers': len(buyers),
       'bought_assets': sum(len(truth[u]) for u in buyers),
       'found_in_top10': {m: sum(hits[m].values()) for m in methods},
       'found_by_union_of_all_methods': union_found,
       'mean_ndcg_buyers': {m: round(statistics.mean(ndcg[m][u] for u in buyers), 4) for m in methods},
       'oracle_best_method_per_buyer': round(oracle_all, 4),
       'mean_common_items_in_top10': overlap, 'pairs': pairs}
out_path.write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps({k: v for k, v in out.items() if k not in ('pairs', 'mean_common_items_in_top10')}, indent=1))
