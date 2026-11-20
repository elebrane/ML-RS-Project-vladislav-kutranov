"""Rule-status cascades vs decayed frequency on validation cutoffs (no LLM, no June data)."""
import sys, json, statistics
import hybrid, data_adapter
cutoff = sys.argv[1]
args = hybrid.parse_args(['--data-dir', '../Data FAR-Trans', '--output-dir', 'outputs/_scratch/tmp_so',
                          '--cutoff', cutoff, '--max-users', '3000', '--no-llm'])
starter, assets, _h, truth, contexts, base, _e, pop, manifest = data_adapter.load(args)
users = [u for u in manifest['users'] if truth.get(u)]
m = hybrid.Market(args.data_dir, cutoff, users)
cfg = {'decay_tau': 60.0, 'own_max': 10, 'popular_new_max': 10}
popo = [i for i in starter.order(pop) if i in assets]
def status_a(f):   # current prompt: exited first
    return hybrid.rule_status(f)
def status_b(f):   # core first, then exited
    if f['buys'] >= 2 and f['last_buy_days_ago'] <= 90: return 'core'
    if f['position'] == 'sold all': return 'exited_gain' if (f['last_sale_vs_avg_buy'] or 0) > 0 else 'exited_loss'
    if f['first_buy_days_ago'] <= 60: return 'new'
    if f['buys'] >= 2: return 'lapsed'
    return 'one_off'
ranks = {'r1 core,new,lapsed,one_off,eg,el': ['core','new','lapsed','one_off','exited_gain','exited_loss'],
         'r2 core,new,eg,lapsed,one_off,el': ['core','new','exited_gain','lapsed','one_off','exited_loss'],
         'r3 core,new,eg,el,lapsed,one_off': ['core','new','exited_gain','exited_loss','lapsed','one_off']}
res = {'c1': [], 'latest_buy': []}
for u in users:
    f = hybrid.client_facts(u, m, assets, cfg); pool = hybrid.build_pool(f, m, assets, cfg, popo)
    A = f['assets']; tail = pool['new'] + pool['fill']; rel = truth[u]
    c1 = sorted(pool['own'], key=lambda i: (-A[i]['decayed_frequency'], A[i]['last_buy_days_ago'], i))
    res['c1'].append(starter.ndcg10(hybrid.fill10(c1 + tail, popo), rel))
    res['latest_buy'].append(starter.ndcg10(base[u]['latest_buy'], rel))
    for sname, sf in (('A exited-first', status_a), ('B core-first', status_b)):
        for rname, order in ranks.items():
            rk = {s: k for k, s in enumerate(order)}
            own = sorted(pool['own'], key=lambda i: (rk[sf(A[i])], -A[i]['decayed_frequency'], A[i]['last_buy_days_ago'], i))
            res.setdefault(f'{sname} | {rname}', []).append(starter.ndcg10(hybrid.fill10(own + tail, popo), rel))
print(cutoff, 'buyers', len(users))
for k, v in res.items(): print(f'  {statistics.mean(v):.4f}  {k}')
