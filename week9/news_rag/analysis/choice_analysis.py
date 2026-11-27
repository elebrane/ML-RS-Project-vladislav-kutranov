"""What did the LLM choose instead of the popularity order? (failure analysis; reads one run's details.json)

For every client the LLM was called for: overlap of its k picks with C0's top-k, share of picks that are
simply the first k candidates in the shown (random) order, the popularity rank of picks, whether the
pick's sector is among the client's past sectors, and the 30-day price change of picks. Truth is used only
for the last block (which purchases the LLM dropped or added).
"""
import json
import statistics
import sys
from pathlib import Path

import common

run = Path(sys.argv[1])                     # e.g. outputs/D1_may/a1_now
arm = sys.argv[2] if len(sys.argv) > 2 else 'llm_nonews'
cohort_name = run.parent.name
cohorts = json.loads((common.HERE / 'configs' / 'cohorts.json').read_text())
c = common.load_cohort(cohorts[cohort_name], run)
blocks = common.blocks_for(c)
starter, assets, _h, truth, contexts, *_ = c['bundle']
details = json.loads((run / 'details.json').read_text())
cat = common.hybrid.category_of(assets)
market = c['market']
st = {}
def state(i):
    if i not in st:
        st[i] = market.state(i, market.cutoff, cat, list(assets))
    return st[i]
ov, firstk, rk_llm, rk_c0, sec_llm, sec_c0, px_llm, px_c0 = [], [], [], [], [], [], [], []
dropped = added = 0
users = 0
for u, info in details.items():
    rec = info.get('llm', {}).get(arm)
    if not rec or rec['status'] == 'fallback':
        continue
    ids = info['ids']
    k = rec['k']
    try:
        picks = [ids[x] for x in json.loads(rec['raw'][rec['raw'].find('['):rec['raw'].rfind(']') + 1]) if x in ids][:k]
    except Exception:
        continue
    if not picks:
        continue
    users += 1
    b = blocks[u]
    c0 = b['block'][:k]
    shown = list(ids.values())
    ov.append(len(set(picks) & set(c0)) / k)
    firstk.append(len(set(picks) & set(shown[:k])) / k)
    mysec = {s for s, n in c['facts'][u]['sector_buys'].items()}
    for i in picks:
        rk_llm.append(state(i)['rank'] or 99)
        sec_llm.append(common.hybrid.sector_of(assets, i) in mysec)
        if state(i)['change30'] is not None:
            px_llm.append(state(i)['change30'])
    for i in c0:
        rk_c0.append(state(i)['rank'] or 99)
        sec_c0.append(common.hybrid.sector_of(assets, i) in mysec)
        if state(i)['change30'] is not None:
            px_c0.append(state(i)['change30'])
    rel = truth.get(u, set())
    dropped += len((set(c0) - set(picks)) & rel)
    added += len((set(picks) - set(c0)) & rel)
out = {'arm': arm, 'run': str(run), 'clients': users,
       'mean_overlap_with_C0_topk': round(statistics.mean(ov), 3),
       'mean_share_equal_to_first_k_shown': round(statistics.mean(firstk), 3),
       'median_popularity_rank_of_picks': {'llm': statistics.median(rk_llm), 'c0': statistics.median(rk_c0)},
       'share_in_client_past_sectors': {'llm': round(statistics.mean(sec_llm), 3), 'c0': round(statistics.mean(sec_c0), 3)},
       'mean_price_change_30d_of_picks': {'llm': round(statistics.mean(px_llm), 4), 'c0': round(statistics.mean(px_c0), 4)},
       'bought_assets_dropped_from_C0_slots': dropped, 'bought_assets_added_by_llm': added}
(run / f'choice_analysis_{arm}.json').write_text(json.dumps(out, indent=1) + '\n')
print(json.dumps(out, indent=1))
