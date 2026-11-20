"""Hybrid FAR-Trans HW2: LLM encoder (cited evidence) + LLM reranker over one fixed pool.

RQ: We test whether a zero-shot hybrid LLM pipeline -- an encoder that turns each client's
pre-cutoff transactions into cited structured evidence, and a reranker that orders a fixed pool
of the client's own and currently popular assets using that evidence -- improves nDCG@10 over the
provided Latest buy baseline for the 100 warm FAR-Trans users at the 1 June 2021 cutoff.

2x2 ablation, all cells see the same fact table and the same candidate pool:
  c1 rule      code facts; own assets by decayed frequency, then popular new assets (no LLM)
  c2 encoder   LLM evidence; own assets by status rank (ties: decayed frequency); same cascade
  c3 reranker  code fact table -> LLM reranker
  c4 both      LLM evidence -> LLM reranker
Invalid encoder output: c2 falls back to the c1 order, c4 gives the reranker the c3 input.
Invalid reranker output: c3 falls back to the c1 order, c4 to the c2 order. Every failure is
counted and no user is dropped. Facts use only rows strictly before the cutoff; asset names and
ISINs never reach the model (random per-client IDs). Truth is used only for evaluation and, on
validation cutoffs, to pick smoke-test users. Teaching experiment, not investment advice.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import hashlib
import json
import math
import random
import re
import statistics
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import data_adapter

HERE = Path(__file__).resolve().parent
STATUS_VOCAB = ('core', 'exited_gain', 'exited_loss', 'new', 'lapsed', 'one_off')
DEFAULT_STATUS_RANK = ('core', 'new', 'lapsed', 'one_off', 'exited_gain', 'exited_loss')
CELLS = {'c1': 'hyb_c1_rule', 'c2': 'hyb_c2_encoder', 'c3': 'hyb_c3_reranker', 'c4': 'hyb_c4_both'}
ISIN_RE = re.compile(r'\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b')
ID_RE = re.compile(r'^(A\d{2}) \|', re.M)


def d(value):
    return date.fromisoformat(str(value)[:10])


def fill(template, **values):
    """Replace {name} placeholders only for the given names (prompts contain JSON braces)."""
    for key, value in values.items():
        template = template.replace('{' + key + '}', str(value))
    return template


def pct(x):
    return '-' if x is None else f'{100 * x:+.1f}%'


def seed_of(*parts):
    return int(hashlib.sha256('|'.join(map(str, parts)).encode()).hexdigest(), 16)


# ------------------------------------------------------------------ market data
class Market:
    """Everything the method reads from raw files, strictly before the cutoff."""

    def __init__(self, data_dir, cutoff, users, popularity_days=30, spike=None):
        self.cutoff = d(cutoff)
        self.popularity_days = popularity_days
        self.spike_cfg = spike or {'recent_days': 7, 'baseline_weeks': 4,
                                   'min_ratio': 3.0, 'min_buyers': 5}
        users = set(users)
        self.max_input_day = date.min
        seen, buys = set(), defaultdict(list)
        self.user_rows = defaultdict(list)          # customer -> rows incl. initial positions
        with open(Path(data_dir) / 'transactions.csv', newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                key = tuple(sorted(row.items()))
                if key in seen:
                    continue
                seen.add(key)
                day = d(row['timestamp'])
                if day >= self.cutoff:
                    continue
                self.max_input_day = max(self.max_input_day, day)
                if row['customerID'] in users:
                    self.user_rows[row['customerID']].append(row)
                if int(row['transactionID']) >= 0 and row['transactionType'] == 'Buy':
                    buys[row['ISIN']].append((day, row['customerID']))
        self.buy_days, self.buy_cust = {}, {}
        for isin, lst in buys.items():
            lst.sort()
            self.buy_days[isin] = [x[0] for x in lst]
            self.buy_cust[isin] = [x[1] for x in lst]
        prices = defaultdict(list)
        with open(Path(data_dir) / 'close_prices.csv', newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                day = d(row['timestamp'])
                if day < self.cutoff and float(row['closePrice']) > 0:
                    prices[row['ISIN']].append((day, float(row['closePrice'])))
                    self.max_input_day = max(self.max_input_day, day)
        self.px_days, self.px_vals, self.first_priced = {}, {}, {}
        for isin, lst in prices.items():
            lst.sort()
            self.px_days[isin] = [x[0] for x in lst]
            self.px_vals[isin] = [x[1] for x in lst]
            self.first_priced[isin] = lst[0][0]
        self.profiles = {}
        with open(Path(data_dir) / 'customer_information.csv', newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                if row['customerID'] in users and d(row['timestamp']) < self.cutoff:
                    old = self.profiles.get(row['customerID'])
                    if old is None or row['timestamp'] > old['timestamp']:
                        self.profiles[row['customerID']] = row
        self._ranks, self._spikes = {}, {}

    def assert_no_future(self):
        if self.max_input_day >= self.cutoff:
            raise AssertionError(f'input row dated {self.max_input_day} is not before {self.cutoff}')
        for rows in self.user_rows.values():
            for row in rows:
                if d(row['timestamp']) >= self.cutoff:
                    raise AssertionError('future transaction in client rows')
        for row in self.profiles.values():
            if d(row['timestamp']) >= self.cutoff:
                raise AssertionError('future profile record')

    def count(self, isin, start, end):
        """Distinct bank customers buying isin on days in [start, end)."""
        days = self.buy_days.get(isin)
        if not days:
            return 0
        lo, hi = bisect.bisect_left(days, start), bisect.bisect_left(days, end)
        return len(set(self.buy_cust[isin][lo:hi]))

    def price(self, isin, day, limit=10):
        days = self.px_days.get(isin)
        if not days:
            return None
        i = bisect.bisect_right(days, day) - 1
        if i < 0 or (day - days[i]).days > limit:
            return None
        return self.px_vals[isin][i]

    def change30(self, isin, day):
        """Price change over 30 days, using closes strictly before `day`."""
        a = self.price(isin, day - timedelta(days=1))
        b = self.price(isin, day - timedelta(days=31))
        return None if a is None or b is None else a / b - 1

    def ranks(self, day, category_of, universe):
        """isin -> (rank inside its category, distinct buyers) over the popularity window."""
        if day in self._ranks:
            return self._ranks[day]
        start = day - timedelta(days=self.popularity_days)
        counts, by_cat = {}, defaultdict(list)
        for isin in universe:
            first = self.first_priced.get(isin)
            if first is None or first >= day:
                continue
            counts[isin] = self.count(isin, start, day)
            by_cat[category_of(isin)].append(isin)
        table = {}
        for lst in by_cat.values():
            lst.sort(key=lambda i: (-counts[i], i))
            for rank, isin in enumerate(lst, 1):
                table[isin] = (rank, counts[isin])
        self._ranks[day] = table
        return table

    def spike(self, isin, day):
        key = (isin, day)
        if key not in self._spikes:
            c = self.spike_cfg
            end = day - timedelta(days=c['recent_days'])
            recent = self.count(isin, end, day)
            weeks = c['baseline_weeks']
            base = sum(self.count(isin, end - timedelta(days=7 * (k + 1)), end - timedelta(days=7 * k))
                       for k in range(weeks)) / weeks
            self._spikes[key] = recent >= c['min_buyers'] and recent >= c['min_ratio'] * base
        return self._spikes[key]

    def state(self, isin, day, category_of, universe):
        rank, buyers = self.ranks(day, category_of, universe).get(isin, (None, 0))
        return {'rank': rank, 'buyers': buyers, 'spike': self.spike(isin, day),
                'change30': self.change30(isin, day)}


# ------------------------------------------------------------------ per-client facts
def category_of(assets):
    return lambda i: (assets.get(i) or {}).get('assetCategory') or 'Unknown'


def sector_of(assets, isin):
    a = assets.get(isin) or {}
    return a.get('sector') or a.get('assetSubCategory') or 'Unknown'


def rule_status(f):
    """Deterministic reading of the prompt's status definitions (diagnostics only).

    core comes before the exits: active traders often sell out and buy again within weeks
    (rule cascades on 2021-03/04/05 validation cutoffs: core-first >= exited-first every month).
    """
    if f['buys'] >= 2 and f['last_buy_days_ago'] <= 90:
        return 'core'
    if f['position'] == 'sold all':
        return 'exited_gain' if (f['last_sale_vs_avg_buy'] or 0) > 0 else 'exited_loss'
    if f['first_buy_days_ago'] <= 60:
        return 'new'
    if f['buys'] >= 2:
        return 'lapsed'
    return 'one_off'


def risk_text(prof, cutoff):
    risk = prof.get('riskLevel') or ''
    qd = prof.get('lastQuestionnaireDate') or ''
    if risk.startswith('Predicted_'):
        return 'bank estimate ' + risk[len('Predicted_'):]
    if risk in ('', 'Not_Available') or not qd or d(qd) >= cutoff:
        return 'unknown at cutoff'
    return f'{risk} (questionnaire {qd[:10]})'


def client_facts(uid, market, assets, cfg):
    cutoff, cat = market.cutoff, category_of(assets)
    universe = list(assets)
    rows = sorted(market.user_rows.get(uid, []),
                  key=lambda r: (r['timestamp'], int(r['transactionID'])))
    buys, sells = defaultdict(list), defaultdict(list)
    units, cost, bunits, last_sale_px = Counter(), Counter(), Counter(), {}
    touched = set()
    for r in rows:
        isin, u, real = r['ISIN'], float(r['units'] or 0), int(r['transactionID']) >= 0
        touched.add(isin)
        if r['transactionType'] == 'Buy':
            units[isin] += u
            if real:
                buys[isin].append(d(r['timestamp']))
                cost[isin] += float(r['totalValue'] or 0)
                bunits[isin] += u
        else:
            units[isin] -= u
            if real:
                sells[isin].append(d(r['timestamp']))
                if u > 0:
                    last_sale_px[isin] = float(r['totalValue'] or 0) / u
    per = {}
    for isin, days in buys.items():
        if isin not in assets:
            continue
        days = sorted(days)
        ages = [(cutoff - x).days for x in days]
        gaps = [(b - a).days for a, b in zip(days, days[1:])]
        sold = sorted(sells.get(isin, []))
        position = ('sold all' if sold and units[isin] <= 1e-9 else
                    'sold part' if sold else 'holds')
        sale = None
        if isin in last_sale_px and bunits[isin] > 0 and cost[isin] > 0:
            sale = last_sale_px[isin] / (cost[isin] / bunits[isin]) - 1
        f = {'isin': isin, 'category': cat(isin), 'sector': sector_of(assets, isin),
             'buys': len(days), 'buys_90d': sum(a < 90 for a in ages),
             'buys_30d': sum(a < 30 for a in ages),
             'first_buy_days_ago': ages[0], 'last_buy_days_ago': ages[-1],
             'usual_gap_days': round(statistics.median(gaps[-5:])) if gaps else None,
             'sells': len(sold), 'last_sell_days_ago': (cutoff - sold[-1]).days if sold else None,
             'position': position, 'last_sale_vs_avg_buy': sale,
             'decayed_frequency': sum(math.exp(-a / cfg['decay_tau']) for a in ages),
             'at_first_buy': market.state(isin, days[0], cat, universe),
             'now': market.state(isin, cutoff, cat, universe)}
        f['rule_status'] = rule_status(f)
        per[isin] = f
    order = sorted(per, key=lambda i: (-per[i]['decayed_frequency'], per[i]['last_buy_days_ago'], i))
    all_days = [x for i in per for x in buys[i]]
    b90 = sum((cutoff - x).days < 90 for x in all_days)
    bprev = sum(90 <= (cutoff - x).days < 180 for x in all_days)
    cats = Counter(per[i]['category'] for i in per for _ in buys[i])
    main = sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cats else 'Unknown'
    prof = market.profiles.get(uid, {})
    profile = (f"customer type {prof.get('customerType') or 'unknown'}; "
               f"risk profile {risk_text(prof, cutoff)}; "
               f"investment capacity {prof.get('investmentCapacity') or 'unknown'}")
    sectors = Counter(per[i]['sector'] for i in per for _ in buys[i])
    return {'assets': per, 'order': order, 'buys_90d': b90, 'buys_prev_90d': bprev,
            'main_category': main, 'profile': profile, 'touched': touched,
            'sector_buys': sectors}


# ------------------------------------------------------------------ pool and ids
def build_pool(facts, market, assets, cfg, popularity_order):
    own = facts['order'][:cfg['own_max']]
    cat = category_of(assets)
    table = market.ranks(market.cutoff, cat, list(assets))
    new = [i for i in assets if i not in facts['touched'] and cat(i) == facts['main_category']
           and table.get(i, (None, 0))[1] > 0]
    new.sort(key=lambda i: (table[i][0], i))
    new = new[:cfg['popular_new_max']]
    fills = []
    for isin in popularity_order:
        if len(own) + len(new) + len(fills) >= 10:
            break
        if isin not in own and isin not in new and isin not in fills:
            fills.append(isin)
    return {'own': own, 'new': new, 'fill': fills}


def assign_ids(uid, cutoff, isins):
    numbers = list(range(10, 100))
    random.Random(seed_of(uid, cutoff, 'ids')).shuffle(numbers)
    if len(isins) > len(numbers):
        raise ValueError('too many assets for two-digit ids')
    return {isin: f'A{n}' for isin, n in zip(isins, numbers)}


def rank_text(s):
    if s['rank'] is None:
        return 'not ranked, spike no, price 30d ' + pct(s['change30'])
    top = 'top-10' if s['rank'] <= 10 else 'not top-10'
    return (f"category rank {s['rank']} ({top}), spike {'yes' if s['spike'] else 'no'}, "
            f"price 30d {pct(s['change30'])}")


TABLE_HEADER = ('id | category, sector | buys all/90d/30d | first buy, last buy (days ago) | '
                'usual gap (days) | sells, last sell (days ago) | position | last sale vs avg buy price | '
                'decayed frequency | at first buy: popularity | at cutoff: popularity')


def fact_row(fid, f):
    gap = '-' if f['usual_gap_days'] is None else f['usual_gap_days']
    last_sell = '-' if f['last_sell_days_ago'] is None else f['last_sell_days_ago']
    return (f"{fid} | {f['category']}, {f['sector']} | {f['buys']}/{f['buys_90d']}/{f['buys_30d']} | "
            f"{f['first_buy_days_ago']}, {f['last_buy_days_ago']} | {gap} | {f['sells']}, {last_sell} | "
            f"{f['position']} | {pct(f['last_sale_vs_avg_buy'])} | {f['decayed_frequency']:.2f} | "
            f"{rank_text(f['at_first_buy'])} | {rank_text(f['now'])}")


def fact_table(facts, ids, max_rows):
    rows = [TABLE_HEADER]
    shown = facts['order'][:max_rows]
    rows += [fact_row(ids[i], facts['assets'][i]) for i in shown]
    rest = facts['order'][max_rows:]
    if rest:
        n = sum(facts['assets'][i]['buys'] for i in rest)
        last = min(facts['assets'][i]['last_buy_days_ago'] for i in rest)
        rows.append(f'+ {len(rest)} more assets: {n} buys in total, latest {last} days ago')
    return '\n'.join(rows), shown


def candidate_row(fid, isin, kind, facts, assets, market, encoder=None):
    cat = category_of(assets)
    if kind == 'own':
        f = facts['assets'][isin]
        head = f"{fid} | own | {f['category']}, {f['sector']}"
        if encoder is not None:
            st = encoder['asset_status'][fid]
            return f"{head} | status {st['status']}: {st.get('note', '')} | now: {rank_text(f['now'])}"
        gap = '-' if f['usual_gap_days'] is None else f"{f['usual_gap_days']}d"
        return (f"{head} | buys {f['buys']}/{f['buys_90d']}/{f['buys_30d']} (all/90d/30d), last buy "
                f"{f['last_buy_days_ago']}d ago, usual gap {gap}, decayed frequency "
                f"{f['decayed_frequency']:.2f} | {f['position']}, last sale vs avg buy "
                f"{pct(f['last_sale_vs_avg_buy'])} | now: {rank_text(f['now'])}")
    s = market.state(isin, market.cutoff, cat, list(assets))
    sector = sector_of(assets, isin)
    return (f"{fid} | new | {cat(isin)}, {sector} | bank buyers in last 30 days: {s['buyers']}; "
            f"{rank_text(s)} | client buys in this sector: {facts['sector_buys'].get(sector, 0)}")


# ------------------------------------------------------------------ prompts and parsing
def load_prompts(prompts_dir, encoder_name, reranker_name):
    enc = (Path(prompts_dir) / encoder_name).read_text(encoding='utf-8')
    rer = (Path(prompts_dir) / reranker_name).read_text(encoding='utf-8')
    return {'encoder': enc, 'reranker': rer,
            'sha256': {encoder_name: hashlib.sha256(enc.encode()).hexdigest(),
                       reranker_name: hashlib.sha256(rer.encode()).hexdigest()}}


def parse_encoder(raw, row_ids, max_patterns):
    """(object, None) for a valid answer, else (None, reason)."""
    text = raw or ''
    start, end = text.find('{'), text.rfind('}')
    if start < 0 or end <= start:
        return None, 'no JSON object'
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f'JSON error: {exc.msg}'
    if not isinstance(obj, dict):
        return None, 'not an object'
    pats, status, unknowns = obj.get('observed_patterns'), obj.get('asset_status'), obj.get('unknowns')
    if not isinstance(pats, list) or not isinstance(status, dict) or not isinstance(unknowns, list):
        return None, 'missing or mistyped field'
    if len(pats) > max_patterns:
        return None, f'{len(pats)} patterns > {max_patterns}'
    allowed = set(row_ids)
    for p in pats:
        if not isinstance(p, dict) or not isinstance(p.get('pattern'), str) or not p['pattern'].strip():
            return None, 'bad pattern'
        sup = p.get('supporting_assets')
        if not isinstance(sup, list) or not sup or any(s not in allowed for s in sup):
            return None, 'pattern cites an id outside the rows'
    if set(status) != allowed:
        return None, f'status for {len(set(status) & allowed)} of {len(allowed)} rows'
    for value in status.values():
        if not isinstance(value, dict) or value.get('status') not in STATUS_VOCAB:
            return None, 'status outside the vocabulary'
        if not isinstance(value.get('note', ''), str):
            return None, 'note is not text'
    return obj, None


def parse_ranking(raw, allowed):
    text = raw or ''
    start, end = text.find('['), text.rfind(']')
    if start < 0 or end <= start:
        return None, 'no JSON array'
    try:
        arr = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return None, f'JSON error: {exc.msg}'
    if not isinstance(arr, list) or any(not isinstance(x, str) for x in arr):
        return None, 'not a list of ids'
    if len(arr) != 10:
        return None, f'{len(arr)} ids instead of 10'
    if len(set(arr)) != 10:
        return None, 'duplicate ids'
    if any(x not in allowed for x in arr):
        return None, 'id outside the candidates'
    return arr, None


class FakeGenerator:
    """Deterministic stand-in for pipeline checks: valid encoder JSON, reranker keeps shown order."""
    metadata = {'backend': 'fake'}

    def __init__(self):
        self.prompts = []

    def generate(self, prompt, max_new_tokens=80):
        self.prompts.append(prompt)
        if prompt.startswith('You summarize'):
            ids = ID_RE.findall(prompt)
            obj = {'observed_patterns': [{'pattern': 'fake pattern', 'supporting_assets': ids[:1]}],
                   'asset_status': {i: {'status': 'core', 'note': 'fake'} for i in ids},
                   'unknowns': ['goals']}
            text = json.dumps(obj)
        else:
            text = json.dumps(re.findall(r'^(A\d{2}) \| (?:own|new)', prompt, re.M)[:10])
        return {'text': text, 'seconds': 0.0, 'output_tokens': 0}


# ------------------------------------------------------------------ metrics helpers
def fill10(order, fallback):
    out = []
    for i in list(order) + list(fallback):
        if i not in out:
            out.append(i)
        if len(out) == 10:
            break
    return out


def kendall_tau(a, b):
    common = [x for x in a if x in b]
    pa, pb = {x: i for i, x in enumerate(a)}, {x: i for i, x in enumerate(b)}
    conc = disc = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            s = (pa[common[i]] - pa[common[j]]) * (pb[common[i]] - pb[common[j]])
            conc += s > 0
            disc += s < 0
    return (conc - disc) / (conc + disc) if conc + disc else None


def pick_users(users, truth, n, how, seed):
    pool = [u for u in users if truth.get(u)] if how == 'buyers' else list(users)
    if not n or n >= len(pool):
        return pool
    chosen = set(random.Random(seed).sample(sorted(pool), n))
    return [u for u in users if u in chosen]


def protocol_for(args, assets, users, manifest):
    """Same fields as starter.py run writes, so the hash matches for the same users and cutoff."""
    protocol = {'teaching_defaults_not_official_grading_protocol': True,
                'cutoff': args.cutoff, 'end_exclusive': manifest['end_exclusive'],
                'users': users, 'cohort_selection': 'warm',
                'eligible_assets': sorted(assets), 'exclude_seen': False,
                'candidate_count': args.candidate_count, 'k': 10,
                'empty_truth_policy': 'zero in primary mean; nonempty mean also reported',
                'initial_positions': 'excluded', 'duplicate_policy': 'exact full rows only',
                'profile_use': 'display only in classical baselines',
                'asset_feature_fields': ['assetCategory', 'assetSubCategory', 'sector',
                                         'industry', 'marketID']}
    protocol['source_sha256'] = dict(manifest['data_sha256'])
    protocol['protocol_hash'] = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    return protocol


# ------------------------------------------------------------------ run
def run(args, bundle, generator=None):
    starter, assets, _history, truth, contexts, _baseline, _events, pop, manifest = bundle
    started_all = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cells = [c for c in args.cells.split(',') if c]
    if not cells or any(c not in CELLS for c in cells):
        raise ValueError(f'--cells must be a comma list from {sorted(CELLS)}')
    if args.no_llm:
        cells = ['c1']
    perms = [int(p) for p in str(args.permutations).split(',') if p]
    if not perms or perms[0] != 1:
        raise ValueError('--permutations must start with 1 (the metric permutation)')
    status_rank = [s for s in args.status_rank.split(',') if s]
    if sorted(status_rank) != sorted(STATUS_VOCAB):
        raise ValueError('--status-rank must list every status once')
    users = pick_users(manifest['users'], truth, args.pick, args.pick_from, args.pick_seed)
    manifest['users'] = users
    protocol = protocol_for(args, assets, users, manifest)
    (args.output_dir / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    protocol_match = None
    if args.compare_protocol:
        other = json.loads(Path(args.compare_protocol).read_text(encoding='utf-8'))
        protocol_match = other.get('protocol_hash') == protocol['protocol_hash']
        print(f"protocol_hash {protocol['protocol_hash'][:12]} vs {args.compare_protocol}: "
              f"{'MATCH' if protocol_match else 'DIFFERENT'}", flush=True)

    prompts = load_prompts(args.prompts_dir, args.encoder_prompt, args.reranker_prompt)
    spike = {'recent_days': args.spike_recent_days, 'baseline_weeks': args.spike_baseline_weeks,
             'min_ratio': args.spike_min_ratio, 'min_buyers': args.spike_min_buyers}
    market = Market(args.data_dir, args.cutoff, users, args.popularity_days, spike)
    market.assert_no_future()
    cfg = {'decay_tau': args.decay_tau, 'own_max': args.own_max,
           'popular_new_max': args.popular_new_max}
    popularity_order = [i for i in starter.order(pop) if i in assets]
    needs_llm = any(c in cells for c in ('c2', 'c3', 'c4'))
    if needs_llm and generator is None:
        import self_structured_ranker as ssr
        from runtime import LoggedGenerator
        generator = LoggedGenerator(ssr.ModelGenerator(
            args.model, args.revision, args.device, args.allow_download, dtype=args.dtype,
            temperature=args.temperature, top_p=args.top_p, thinking=args.thinking,
            seed=args.sample_seed, context_limit=args.context_limit), args.output_dir)

    predictions, failures, details = {}, [], {}
    seconds = Counter()
    stability, agreement, out_tokens = defaultdict(list), Counter(), defaultdict(list)
    for n_user, uid in enumerate(users, 1):
        facts = client_facts(uid, market, assets, cfg)
        pool = build_pool(facts, market, assets, cfg, popularity_order)
        pool_isins = pool['own'] + pool['new'] + pool['fill']
        table_isins = facts['order'][:args.max_asset_rows]
        ids = assign_ids(uid, args.cutoff, list(dict.fromkeys(table_isins + pool_isins)))
        isin_of = {v: k for k, v in ids.items()}
        kind = {i: 'own' for i in pool['own']}
        kind.update({i: 'new' for i in pool['new'] + pool['fill']})
        table, shown = fact_table(facts, ids, args.max_asset_rows)
        f = facts['assets']
        c1 = sorted(pool['own'], key=lambda i: (-f[i]['decayed_frequency'], f[i]['last_buy_days_ago'], i))
        tail = pool['new'] + pool['fill']
        preds = {CELLS['c1']: fill10(c1 + tail, popularity_order)}
        rel = truth.get(uid, set())
        info = {'pool': {k: [{'id': ids[i], 'isin': i} for i in v] for k, v in pool.items()},
                'table_ids': [ids[i] for i in shown], 'buys_90d': facts['buys_90d'],
                'buys_prev_90d': facts['buys_prev_90d'], 'main_category': facts['main_category'],
                'rule_status': {ids[i]: f[i]['rule_status'] for i in shown},
                'truth_size': len(rel),
                'truth_in_pool': sorted(ids[i] for i in rel if i in ids and i in pool_isins)}

        encoder, error = None, None
        if 'c2' in cells or 'c4' in cells:
            prompt = fill(prompts['encoder'], cutoff=args.cutoff, profile=facts['profile'],
                          buys_last_90d=facts['buys_90d'], buys_prev_90d=facts['buys_prev_90d'],
                          main_category=facts['main_category'], max_rows=args.max_asset_rows,
                          spike_ratio=f'{args.spike_min_ratio:g}', fact_table=table,
                          status_ids=', '.join(ids[i] for i in shown))
            t0 = time.perf_counter()
            try:
                resp = generator.generate(prompt, max_new_tokens=args.encoder_max_new_tokens)
                encoder, error = parse_encoder(resp['text'], [ids[i] for i in shown], args.max_patterns)
                out_tokens['encoder'].append(resp.get('output_tokens', 0))
                info['encoder'] = {'raw': resp['text'], 'valid': encoder is not None, 'error': error,
                                   'seconds': round(resp.get('seconds', 0.0), 2)}
            except (ValueError, RuntimeError) as exc:
                error = f'{type(exc).__name__}: {exc}'
                info['encoder'] = {'raw': None, 'valid': False, 'error': error}
            seconds['encoder'] += time.perf_counter() - t0
            if encoder is None:
                failures.append({'user_id': uid, 'stage': 'encoder', 'error': error})
            else:
                for i in shown:
                    same = encoder['asset_status'][ids[i]]['status'] == f[i]['rule_status']
                    agreement['agree' if same else 'differ'] += 1

        if 'c2' in cells:
            if encoder is not None:
                rank_of = {s: k for k, s in enumerate(status_rank)}
                own2 = sorted(pool['own'], key=lambda i: (rank_of[encoder['asset_status'][ids[i]]['status']],
                                                          -f[i]['decayed_frequency'],
                                                          f[i]['last_buy_days_ago'], i))
                preds[CELLS['c2']] = fill10(own2 + tail, popularity_order)
            else:
                preds[CELLS['c2']] = list(preds[CELLS['c1']])

        for cell in [c for c in ('c3', 'c4') if c in cells]:
            use_enc = cell == 'c4' and encoder is not None
            head = (f"Profile: {facts['profile']}\nActivity: {facts['buys_90d']} buys in the last 90 days, "
                    f"{facts['buys_prev_90d']} in the 90 days before. Main asset category: "
                    f"{facts['main_category']}.\n")
            if use_enc:
                block = head + ("Evidence about the client's past behaviour (JSON):\n"
                                + json.dumps(encoder, ensure_ascii=False, indent=1))
            else:
                block = head + 'Assets the client has bought (one row each):\n' + table
            rows = {ids[i]: candidate_row(ids[i], i, kind[i], facts, assets, market,
                                          encoder if use_enc else None) for i in pool_isins}
            if cell == 'c4' and CELLS['c2'] in preds:
                fallback = preds[CELLS['c2']]
            else:
                fallback = preds[CELLS['c1']]
            for perm in perms:
                order = [ids[i] for i in pool_isins]
                random.Random(seed_of(uid, args.cutoff, perm)).shuffle(order)
                prompt = fill(prompts['reranker'], client_block=block,
                              candidate_rows='\n'.join(rows[x] for x in order))
                t0 = time.perf_counter()
                try:
                    resp = generator.generate(prompt, max_new_tokens=args.reranker_max_new_tokens)
                    picked, rerr = parse_ranking(resp['text'], set(order))
                    out_tokens['reranker'].append(resp.get('output_tokens', 0))
                    record = {'raw': resp['text'], 'valid': picked is not None, 'error': rerr,
                              'shown_order': order, 'seconds': round(resp.get('seconds', 0.0), 2)}
                except (ValueError, RuntimeError) as exc:
                    picked, rerr = None, f'{type(exc).__name__}: {exc}'
                    record = {'raw': None, 'valid': False, 'error': rerr, 'shown_order': order}
                seconds[cell if perm == 1 else f'{cell}_p{perm}'] += time.perf_counter() - t0
                if picked is None:
                    failures.append({'user_id': uid, 'stage': f'reranker_{cell}_p{perm}', 'error': rerr})
                    ranking = list(fallback)
                else:
                    ranking = [isin_of[x] for x in picked]
                preds[CELLS[cell] if perm == 1 else f'{CELLS[cell]}_p{perm}'] = ranking
                info[f'{cell}_p{perm}'] = record
            if len(perms) > 1:
                a, b = preds[CELLS[cell]], preds[f'{CELLS[cell]}_p{perms[1]}']
                stability[cell].append({'overlap10': len(set(a) & set(b)) / 10,
                                        'kendall_tau': kendall_tau(a, b),
                                        'ndcg_diff': starter.ndcg10(a, rel) - starter.ndcg10(b, rel)})
        info['rankings'] = {}
        for name, ranking in preds.items():
            starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
            info['rankings'][name] = [ids.get(i, i) for i in ranking]
        predictions[uid] = preds
        details[uid] = info
        if n_user % 10 == 0 or n_user == len(users):
            print(f'[{time.strftime("%H:%M:%S")}] {n_user}/{len(users)} users', flush=True)

    # evaluation-only diagnostics
    diag = {'users': len(users), 'buyers': sum(1 for u in users if truth.get(u))}
    recall, oracle = [], []
    for uid in users:
        rel = truth.get(uid, set())
        if not rel:
            continue
        pool = [x['isin'] for v in details[uid]['pool'].values() for x in v]
        recall.append(len(rel & set(pool)) / len(rel))
        oracle.append(starter.ndcg10(fill10([i for i in pool if i in rel] + pool, popularity_order), rel))
    diag['pool_recall_buyers'] = round(statistics.mean(recall), 4) if recall else None
    diag['oracle_pool_ndcg_buyers'] = round(statistics.mean(oracle), 4) if oracle else None
    diag['status_agreement_with_rule'] = dict(agreement)
    diag['stability'] = {c: {k: (round(statistics.mean([r[k] for r in v if r[k] is not None]), 4)
                                 if any(r[k] is not None for r in v) else None)
                             for k in ('overlap10', 'kendall_tau', 'ndcg_diff')}
                         for c, v in stability.items()}
    diag['output_tokens'] = {k: {'mean': round(statistics.mean(v), 1), 'max': max(v)}
                             for k, v in out_tokens.items() if v}
    (args.output_dir / 'details.json').write_text(json.dumps(details, ensure_ascii=False, indent=1) + '\n',
                                                 encoding='utf-8')
    gen_meta = generator.metadata if generator is not None else {'backend': 'no_llm'}
    runtime_by_method = {CELLS['c1']: 0.0}
    if 'c2' in cells:
        runtime_by_method[CELLS['c2']] = seconds['encoder']
    if 'c3' in cells:
        runtime_by_method[CELLS['c3']] = seconds['c3']
    if 'c4' in cells:
        runtime_by_method[CELLS['c4']] = seconds['encoder'] + seconds['c4']
    meta = dict(gen_meta, method='hybrid_v2', run_name=args.run_name or args.output_dir.name,
                cells=cells, permutations=perms, cutoff=args.cutoff, pick=args.pick,
                pick_from=args.pick_from, decay_tau=args.decay_tau, own_max=args.own_max,
                popular_new_max=args.popular_new_max, popularity_days=args.popularity_days,
                spike=spike, max_asset_rows=args.max_asset_rows, max_patterns=args.max_patterns,
                status_rank=status_rank, encoder_max_new_tokens=args.encoder_max_new_tokens,
                reranker_max_new_tokens=args.reranker_max_new_tokens,
                prompt_files={'encoder': args.encoder_prompt, 'reranker': args.reranker_prompt},
                prompt_sha256=prompts['sha256'], protocol_hash=protocol['protocol_hash'],
                protocol_matches_reference=protocol_match,
                seconds={k: round(v, 1) for k, v in seconds.items()},
                runtime_seconds_by_method={k: round(v, 1) for k, v in runtime_by_method.items()},
                failures_by_stage=dict(Counter(x['stage'] for x in failures)),
                total_wall_seconds=round(time.perf_counter() - started_all, 1),
                diagnostics=diag)
    data_adapter.save_results(args, bundle, predictions, failures, 'hybrid_v2', meta)
    (args.output_dir / 'diagnostics.json').write_text(json.dumps(diag, indent=2) + '\n')
    print(json.dumps({'failures_by_stage': meta['failures_by_stage'], 'seconds': meta['seconds'],
                      'diagnostics': diag}, indent=2))
    return meta


# ------------------------------------------------------------------ command line
def build_parser():
    p = data_adapter.arguments('Hybrid LLM encoder + reranker over one fixed pool (2x2 ablation)')
    for action in p._actions:
        if action.dest == 'device':
            action.choices = ['auto', 'cpu', 'cuda', 'mps']
        if action.dest == 'model':
            action.required = False
            action.default = 'Qwen/Qwen3-4B-Instruct-2507'
    p.add_argument('--config', type=Path, help='JSON config (configs/hybrid_*.json)')
    p.add_argument('--run-name')
    p.add_argument('--cells', default='c1,c2,c3,c4')
    p.add_argument('--permutations', default='1', help='comma list; 1 = metric, 2 = stability check')
    p.add_argument('--pick', type=int, default=0, help='0 = all users of the sample')
    p.add_argument('--pick-from', choices=['all', 'buyers'], default='all',
                   help='buyers = users with purchases in the window (validation cutoffs only)')
    p.add_argument('--pick-seed', type=int, default=0)
    p.add_argument('--decay-tau', type=float, default=60.0)
    p.add_argument('--own-max', type=int, default=10)
    p.add_argument('--popular-new-max', type=int, default=10)
    p.add_argument('--popularity-days', type=int, default=30)
    p.add_argument('--spike-recent-days', type=int, default=7)
    p.add_argument('--spike-baseline-weeks', type=int, default=4)
    p.add_argument('--spike-min-ratio', type=float, default=3.0)
    p.add_argument('--spike-min-buyers', type=int, default=5)
    p.add_argument('--max-asset-rows', type=int, default=12)
    p.add_argument('--max-patterns', type=int, default=8)
    p.add_argument('--status-rank', default=','.join(DEFAULT_STATUS_RANK))
    p.add_argument('--prompts-dir', type=Path, default=HERE / 'prompts')
    p.add_argument('--encoder-prompt', default='encoder_v3.txt')
    p.add_argument('--reranker-prompt', default='reranker_v1.txt')
    p.add_argument('--encoder-max-new-tokens', type=int, default=1000)
    p.add_argument('--reranker-max-new-tokens', type=int, default=100)
    p.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    p.add_argument('--temperature', type=float, default=0.0)
    p.add_argument('--top-p', type=float, default=1.0)
    p.add_argument('--sample-seed', type=int, default=0)
    p.add_argument('--thinking', action='store_true')
    p.add_argument('--context-limit', type=int, default=8192)
    p.add_argument('--compare-protocol', help='protocol.json of a starter run to compare hashes with')
    p.add_argument('--no-llm', action='store_true', help='only c1 and the four baselines')
    p.add_argument('--fake-llm', action='store_true', help='pipeline check with a deterministic fake model')
    return p


CONFIG_TOP = ('run_name', 'data_dir', 'output_dir', 'cells', 'permutations', 'compare_protocol')
CONFIG_SECTIONS = ('protocol', 'facts', 'pool', 'encoder', 'reranker', 'generation')


def parse_args(argv=None):
    """Command-line flags win over --config values; --config wins over defaults."""
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        config = json.loads(known.config.read_text(encoding='utf-8'))
        unknown_top = sorted(set(config) - set(CONFIG_TOP) - set(CONFIG_SECTIONS) - {'note'})
        if unknown_top:
            raise ValueError(f'Unknown config sections: {unknown_top}')
        flat = {k: config[k] for k in CONFIG_TOP if k in config}
        for section in CONFIG_SECTIONS:
            flat.update(config.get(section) or {})
        for key in ('cells', 'permutations', 'status_rank'):
            if isinstance(flat.get(key), list):
                flat[key] = ','.join(map(str, flat[key]))
        actions = {a.dest: a for a in parser._actions}
        unknown = sorted(set(flat) - set(actions))
        if unknown:
            raise ValueError(f'Unknown config keys: {unknown}')
        for key, value in flat.items():
            if key in ('data_dir', 'output_dir', 'starter_dir', 'prompts_dir') and value is not None:
                value = Path(value)          # relative to the working folder
            choices = actions[key].choices
            if choices and value not in choices:
                raise ValueError(f'Config {key}={value!r} is not one of {list(choices)}')
            actions[key].default = value
            actions[key].required = False
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.fake_llm and args.no_llm:
        raise SystemExit('--fake-llm and --no-llm are exclusive')
    random.seed(0)
    bundle = data_adapter.load(args)
    run(args, bundle, FakeGenerator() if args.fake_llm else None)


if __name__ == '__main__':
    main()
