"""Hybrid LLM encoder + LLM reranker over a four-source candidate pool (HW2, revised question).

RQ: We test whether an LLM encoder that turns a client's transaction history into weights over
four candidate sources (own history, current bank-wide popularity, similarity to held assets,
recent price gainers), combined with an LLM reranker of the resulting pool, improves nDCG@10 over
the next 30 days over the four provided baselines for the warm FAR-Trans evaluation users.

Sources (all facts strictly before the cutoff):
  OWN             assets the client bought; latest buy first, ties by bank buyers in 7 days
  POPULAR         5 never-bought assets with most distinct bank buyers in 30 days, client's main category
  SIMILAR         5 never-bought assets from the starter's CBF candidate order
  RECENT_GAINERS  5 never-bought assets with the largest 30-day price change
Score of an asset = W[source] / rank inside its source; the Top-10 are the best scores.
  A1 rule           fixed lexicographic weights OWN > POPULAR > SIMILAR > RECENT_GAINERS (no LLM)
  A2 formula        weights from the client's shrunk past purchase mix (no LLM)
  A3 llm_encoder    weights from the LLM encoder (same facts as A2)
  A4 llm_reranker   LLM picks and orders 10 of the A1 pool
  A5 hybrid         LLM picks and orders 10 of the 20 best A3 candidates, sees the encoder profile
Invalid LLM output: encoder -> A2 weights, reranker -> fill from the no-LLM order; counted as failure.
Truth is used only for evaluation and for choosing smoke-test users on a validation cutoff.
Teaching experiment, not investment advice.
"""
from __future__ import annotations

import bisect
import csv
import json
import random
import re
import time
from collections import Counter, defaultdict
from datetime import date, timedelta
from pathlib import Path

import data_adapter
import self_structured_ranker as ssr

SOURCES = ('OWN', 'POPULAR', 'SIMILAR', 'RECENT_GAINERS')
LEX = {'OWN': 1.0, 'POPULAR': 1e-3, 'SIMILAR': 1e-6, 'RECENT_GAINERS': 1e-9}
# Bank-wide reference, monthly cutoffs 2020-07..2021-05 (eda/encoder_prep.py). For a May-2021
# validation cutoff these numbers include May itself; for the June test cutoff they are clean.
PRIOR = {'new_share': 0.33, 'POPULAR': 0.52, 'SIMILAR': 0.56, 'RECENT_GAINERS': 0.12}
SHRINK = 3.0
METHODS = {'A1': 'hyb_a1_rule', 'A2': 'hyb_a2_formula', 'A3': 'hyb_a3_llm_encoder',
           'A4': 'hyb_a4_llm_reranker', 'A5': 'hyb_a5_hybrid'}

ENCODER_PROMPT = """You describe how one retail client of a Greek bank has chosen securities in the past.
Use only the facts below. Do not give investment advice and do not forecast prices.

TASK
Estimate where this client's purchases in the next {horizon} days ({cutoff} to {end}) will most likely come from.
Sources:
- OWN: buying again an asset the client bought before
- POPULAR: a new asset that many clients of the bank are buying right now
- SIMILAR: a new asset from a sub-category the client already holds
- RECENT_GAINERS: a new asset whose price rose strongly over the last 30 days

REFERENCE: all clients of the bank, before {cutoff}
- 67% of purchases are repeats of an asset the client bought before.
- Of new purchases, 52% were among the 10 most bought assets of their category in the previous
  30 days, 56% were in a sub-category the client already held, and 12% were in the top 10% by
  30-day return (10% would be expected by chance).
- A client's own past mix predicts the next one: clients whose past purchases were mostly repeats
  keep repeating (88% repeats); clients with few past repeats mostly buy new assets (39% repeats).
- When the client has fewer than 3 purchases, stay close to the reference. Move away from it as
  the history grows.

CLIENT on {cutoff}
{client_block}

LAST PURCHASES, newest first (up to 15)
date | asset | category | repeat | popular then | similar | recent gainer then
{history_rows}

OUTPUT
Return only one JSON object and nothing else:
{{"new_share": s, "popular": p, "similar": q, "recent_gainers": r, "confidence": "low" | "medium" | "high"}}
- new_share: expected share of NEW assets among the client's next purchases, from 0 to 1
- popular, similar, recent_gainers: how those new purchases split between the three sources;
  three numbers from 0 to 1 that sum to 1
- confidence: "low" if the client has fewer than 3 purchases"""

RERANKER_PROMPT = """You rank securities for one retail client of a Greek bank by how likely the client is to buy
each one in the next {horizon} days. Use only the facts below. This is purchase prediction, not
investment advice.

TASK
Below are {n} candidate assets from four sources. Choose the {k} the client is most
likely to buy between {cutoff} and {end} and order them from most to least likely.

WHAT IS KNOWN ABOUT CLIENTS OF THIS BANK (before {cutoff})
- Most purchases are repeats. Among the client's own assets, the one bought most recently, or
  bought several times in the last 90 days, is the most likely repeat.
- New purchases mostly follow what many clients of the bank are buying now in the client's main
  category. The total number of buyers matters more than a sudden jump in buyers.
- Being a recent price gainer on its own rarely leads to a purchase.

CLIENT on {cutoff}
{client_block}{profile_line}

CANDIDATES
id | source | asset | category / sub-category | own: buys 90d / all, days since last buy, holds now, last sale | bank buyers 7d / 30d | buyer jump 7d | 30d price change
{rows}

OUTPUT
Return only a JSON array of exactly {k} different ids from the list above, most likely first,
for example ["C03", "C01", "C07", ...]. No other text."""


def d(s):
    return date.fromisoformat(s[:10])


# ------------------------------------------------------------------ market data
class Market:
    """Everything the method needs from raw files, strictly before the cutoff."""

    def __init__(self, data_dir, cutoff, users):
        self.cutoff = d(cutoff)
        users = set(users)
        seen = set()
        buys = defaultdict(list)            # isin -> [(day, customer)] positive-ID buys, all customers
        self.user_rows = defaultdict(list)  # customer -> raw rows (incl. initial positions)
        cust_days = set()                   # (day, isin, customer) for customer-day counts
        with open(Path(data_dir) / 'transactions.csv', newline='', encoding='utf-8') as fh:
            for row in csv.DictReader(fh):
                key = tuple(sorted(row.items()))
                if key in seen:
                    continue
                seen.add(key)
                day = d(row['timestamp'])
                if day >= self.cutoff:
                    continue
                if row['customerID'] in users:
                    self.user_rows[row['customerID']].append(row)
                if int(row['transactionID']) >= 0 and row['transactionType'] == 'Buy':
                    buys[row['ISIN']].append((day, row['customerID']))
                    cust_days.add((day, row['ISIN'], row['customerID']))
        self.daily = defaultdict(Counter)   # day -> isin -> customers buying that day
        for day, isin, _ in cust_days:
            self.daily[day][isin] += 1
        self._memo = {}
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
        self._cache = {}

    def buyers(self, isin, start, end, distinct=True):
        """Bank customers buying isin on days in [start, end); customer-days if not distinct."""
        key = (isin, start, end, distinct)
        if key in self._memo:
            return self._memo[key]
        days = self.buy_days.get(isin)
        if not days:
            return 0
        lo, hi = bisect.bisect_left(days, start), bisect.bisect_left(days, end)
        cust = self.buy_cust[isin][lo:hi]
        value = len(set(cust)) if distinct else len(set(zip(days[lo:hi], cust)))
        self._memo[key] = value
        return value

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

    def day_flags(self, day, category_of):
        """As of `day`: category top-10 by 30-day customer-days, 30-day change, its 90th percentile."""
        if day in self._cache:
            return self._cache[day]
        valid = [i for i, f in self.first_priced.items() if f < day]
        window = Counter()
        for k in range(1, 31):
            window.update(self.daily.get(day - timedelta(days=k), {}))
        counts = {i: window.get(i, 0) for i in valid}
        top = set()
        by_cat = defaultdict(list)
        for i in valid:
            by_cat[category_of(i)].append(i)
        for lst in by_cat.values():
            top.update(sorted(lst, key=lambda i: (-counts[i], i))[:10])
        ch = {i: self.change30(i, day) for i in valid}
        vals = sorted(v for v in ch.values() if v is not None)
        thr = vals[int(0.9 * (len(vals) - 1))] if len(vals) > 10 else float('inf')
        self._cache[day] = (top, ch, thr)
        return self._cache[day]


# ------------------------------------------------------------------ per-client facts
def cat_of(assets):
    return lambda i: (assets.get(i) or {}).get('assetCategory') or 'Unknown'


def client_facts(uid, market, assets, eligible):
    rows = market.user_rows.get(uid, [])
    cat = cat_of(assets)
    ev = sorted({(d(r['timestamp']), r['ISIN']) for r in rows
                 if r['transactionType'] == 'Buy' and int(r['transactionID']) >= 0 and r['ISIN'] in eligible})
    seen_isin, seen_sub, events = {}, {}, []
    for day, isin in ev:
        sub = (assets.get(isin) or {}).get('assetSubCategory') or 'Unknown'
        repeat = isin in seen_isin and seen_isin[isin] < day
        similar = (not repeat) and sub in seen_sub and seen_sub[sub] < day
        top, ch, thr = market.day_flags(day, cat)
        events.append({'day': day, 'isin': isin, 'repeat': repeat, 'similar': similar,
                       'popular': (not repeat) and isin in top,
                       'gainer': (not repeat) and ch.get(isin) is not None and ch[isin] >= thr})
        seen_isin.setdefault(isin, day)
        seen_sub.setdefault(sub, day)
    new = [e for e in events if not e['repeat']]
    cats = Counter(cat(e['isin']) for e in events)
    main = sorted(cats.items(), key=lambda kv: (-kv[1], kv[0]))[0][0] if cats else 'Unknown'
    sells = [r for r in rows if r['transactionType'] == 'Sell']
    prof = market.profiles.get(uid, {})
    risk = prof.get('riskLevel') or 'unknown'
    qd = prof.get('lastQuestionnaireDate') or ''
    if risk.startswith('Predicted_'):
        risk = 'bank estimate: ' + risk[len('Predicted_'):]
    elif risk in ('', 'Not_Available') or not qd or d(qd) >= market.cutoff:
        risk = 'unknown on cutoff'
    units, cost, bunits, last_sale = Counter(), Counter(), Counter(), {}
    for r in sorted(rows, key=lambda r: (r['timestamp'], int(r['transactionID']))):
        u = float(r['units'] or 0)
        if r['transactionType'] == 'Buy':
            units[r['ISIN']] += u
            if int(r['transactionID']) >= 0:
                cost[r['ISIN']] += float(r['totalValue'] or 0)
                bunits[r['ISIN']] += u
        else:
            units[r['ISIN']] -= u
            if int(r['transactionID']) >= 0 and u > 0:
                last_sale[r['ISIN']] = float(r['totalValue'] or 0) / u
    sale = {}
    for isin, p in last_sale.items():
        avg = cost[isin] / bunits[isin] if bunits[isin] else None
        sale[isin] = 'none' if avg is None else ('gain' if p > avg else 'loss')
    return {'events': events, 'new': new, 'main': main, 'cats': cats, 'n_sells': len(sells),
            'customer_type': prof.get('customerType') or 'unknown', 'risk': risk,
            'holds': {i: units[i] > 1e-9 for i in units}, 'last_sale': sale}


def mix(f):
    n, nn = len(f['events']), len(f['new'])
    k = {s: sum(e[key] for e in f['new']) for s, key in
         (('POPULAR', 'popular'), ('SIMILAR', 'similar'), ('RECENT_GAINERS', 'gainer'))}
    return n, nn, k


def formula_weights(f):
    n, nn, k = mix(f)
    s = (nn + SHRINK * PRIOR['new_share']) / (n + SHRINK)
    raw = {src: (k[src] + SHRINK * PRIOR[src]) / (nn + SHRINK) for src in k}
    tot = sum(raw.values())
    return {'new_share': s, **{src.lower(): raw[src] / tot for src in raw}}


def weights_of(enc):
    s = enc['new_share']
    return {'OWN': 1 - s, 'POPULAR': s * enc['popular'], 'SIMILAR': s * enc['similar'],
            'RECENT_GAINERS': s * enc['recent_gainers']}


def client_block(f, market):
    n, nn, k = mix(f)
    ev = f['events']
    if not ev:
        return f"customer type: {f['customer_type']}; risk profile on file: {f['risk']}\npurchases: none"

    def share(x):
        return f'{x / nn:.2f}' if nn else 'n/a'
    others = [c for c, _ in sorted(f['cats'].items(), key=lambda kv: (-kv[1], kv[0])) if c != f['main']]
    return '\n'.join([
        f"customer type: {f['customer_type']}; risk profile on file: {f['risk']}",
        f"purchases: {n} buys of {len({e['isin'] for e in ev})} assets since {ev[0]['day']}; "
        f"last buy {(market.cutoff - ev[-1]['day']).days} days ago; sells: {f['n_sells']}",
        f"past mix: repeats {(n - nn) / n:.2f}; of {nn} new purchases: popular {share(k['POPULAR'])}, "
        f"similar {share(k['SIMILAR'])}, recent gainers {share(k['RECENT_GAINERS'])}",
        f"main category: {f['main']} ({f['cats'][f['main']] / n:.2f} of purchases); "
        f"other categories: {', '.join(others) or 'none'}"])


def name_of(assets, isin, width=40):
    return ((assets.get(isin) or {}).get('assetName') or isin)[:width]


def encoder_prompt(f, market, assets, horizon, end):
    def yn(v):
        return 'yes' if v else 'no'
    rows = [f"{e['day']} | {name_of(assets, e['isin'])} | {cat_of(assets)(e['isin'])} | {yn(e['repeat'])} | "
            f"{yn(e['popular'])} | {yn(e['similar'])} | {yn(e['gainer'])}"
            for e in reversed(f['events'][-15:])]
    return ENCODER_PROMPT.format(horizon=horizon, cutoff=market.cutoff, end=end,
                                 client_block=client_block(f, market),
                                 history_rows='\n'.join(rows) or '(none)')


# ------------------------------------------------------------------ candidate pool
def build_pool(uid, f, market, assets, history, context):
    c = market.cutoff
    own_set = set(history.get(uid, {}))
    last = {}
    for e in f['events']:
        last[e['isin']] = e['day']
    b7 = {i: market.buyers(i, c - timedelta(days=7), c) for i in own_set}
    own = sorted(own_set, key=lambda i: (-(last.get(i, date.min).toordinal()), -b7[i], i))[:10]
    cat = cat_of(assets)
    elig = [i for i in context['eligible_asset_ids'] if i not in own_set]
    pop_counts = {i: market.buyers(i, c - timedelta(days=30), c) for i in elig}
    popular = sorted([i for i in elig if cat(i) == f['main'] and pop_counts[i] > 0],
                     key=lambda i: (-pop_counts[i], i))[:5]
    if len(popular) < 5:
        popular += sorted([i for i in elig if i not in popular and pop_counts[i] > 0],
                          key=lambda i: (-pop_counts[i], i))[:5 - len(popular)]
    similar = [a['id'] for a in context['candidate_assets'] if a['id'] not in own_set][:5]
    ch = {i: market.change30(i, c) for i in elig}
    gainers = sorted([i for i in elig if ch[i] is not None], key=lambda i: (-ch[i], i))[:5]
    pool, seen = [], set()
    for src, lst in zip(SOURCES, (own, popular, similar, gainers)):
        r = 0
        for i in lst:
            if i in seen:
                continue
            r += 1
            pool.append({'isin': i, 'source': src, 'rank': r})
            seen.add(i)
    return pool


def rank_by_weights(pool, w):
    scored = sorted(enumerate(pool), key=lambda t: (-w[t[1]['source']] / t[1]['rank'], t[0]))
    return [e['isin'] for _, e in scored]


def fill10(order, fallback):
    out = []
    for i in list(order) + list(fallback):
        if i not in out:
            out.append(i)
        if len(out) == 10:
            break
    return out


def candidate_rows(pool_isins, src_of, f, market, assets):
    c = market.cutoff
    own_ev = defaultdict(list)
    for e in f['events']:
        own_ev[e['isin']].append(e['day'])
    rows, ids = [], {}
    for n, i in enumerate(pool_isins, 1):
        cid = f'C{n:02d}'
        ids[cid] = i
        a = assets.get(i) or {}
        if src_of[i] == 'OWN':
            days = own_ev.get(i, [])
            b90 = sum(1 for x in days if x >= c - timedelta(days=90))
            since = (c - max(days)).days if days else 'n/a'
            own = (f"{b90} / {len(days)}, {since} days, {'yes' if f['holds'].get(i) else 'no'}, "
                   f"{f['last_sale'].get(i, 'none')}")
        else:
            own = '-'
        b7 = market.buyers(i, c - timedelta(days=7), c)
        b30 = market.buyers(i, c - timedelta(days=30), c)
        b90 = market.buyers(i, c - timedelta(days=90), c)
        base = (b90 - b7) / 83 * 7
        jump = f'{b7 / base:.1f}x' if base > 0 else ('new' if b7 else 'n/a')
        ch = market.change30(i, c)
        chs = f'{ch * 100:+.1f}%' if ch is not None else 'n/a'
        rows.append(f"{cid} | {src_of[i]} | {name_of(assets, i)} | {a.get('assetCategory', '?')} / "
                    f"{a.get('assetSubCategory', '?')} | {own} | {b7} / {b30} | {jump} | {chs}")
    return '\n'.join(rows), ids


def reranker_prompt(block, profile_line, rows, n, k, horizon, cutoff, end):
    return RERANKER_PROMPT.format(horizon=horizon, n=n, k=k, cutoff=cutoff, end=end,
                                  client_block=block, profile_line=profile_line, rows=rows)


# ------------------------------------------------------------------ parsing
def parse_encoder(raw):
    """(normalised encoder output or None, repaired flag)."""
    m = re.search(r'\{.*?\}', raw or '', re.S)
    if not m:
        return None, False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None, False
    if 'recent_gainers' not in obj and 'return' in obj:
        obj['recent_gainers'] = obj['return']
    try:
        s = float(obj['new_share'])
        parts = {k: float(obj[k]) for k in ('popular', 'similar', 'recent_gainers')}
    except (KeyError, TypeError, ValueError):
        return None, False
    if not 0 <= s <= 1 or any(v < 0 for v in parts.values()) or sum(parts.values()) <= 0:
        return None, False
    tot = sum(parts.values())
    out = {'new_share': s, **{k: v / tot for k, v in parts.items()}, 'confidence': obj.get('confidence')}
    return out, abs(tot - 1) > 0.05


def parse_ranking(raw, allowed, k):
    """(valid ids in answer order, ok): ok means exactly k distinct allowed ids came first."""
    ids = re.findall(r'C\d{2}', raw or '')
    picked = []
    for i in ids:
        if i in allowed and i not in picked:
            picked.append(i)
    ok = len(ids) >= k and len(set(ids[:k])) == k and all(i in allowed for i in ids[:k])
    return picked[:k], ok


class FakeGenerator:
    """Deterministic stand-in for pipeline checks: fixed encoder answer, reranker keeps the order."""
    metadata = {'backend': 'fake'}

    def generate(self, prompt, max_new_tokens=80):
        if prompt.startswith('You describe'):
            text = '{"new_share": 0.4, "popular": 0.5, "similar": 0.3, "recent_gainers": 0.2, "confidence": "medium"}'
        else:
            text = json.dumps(re.findall(r'^(C\d{2}) \|', prompt, re.M)[:10])
        return {'text': text, 'seconds': 0.0}


# ------------------------------------------------------------------ run
def pick_users(users, truth, n, how, seed):
    pool = [u for u in users if truth.get(u)] if how == 'buyers' else list(users)
    if not n or n >= len(pool):
        return pool
    chosen = set(random.Random(seed).sample(sorted(pool), n))
    return [u for u in users if u in chosen]


def run(args, bundle, generator=None):
    starter, assets, history, truth, contexts, baseline, events, pop, manifest = bundle
    args.output_dir.mkdir(parents=True, exist_ok=True)
    users = pick_users(manifest['users'], truth, args.pick, args.pick_from, args.pick_seed)
    manifest['users'] = users
    manifest['user_selection'] = {'pick': args.pick, 'pick_from': args.pick_from, 'seed': args.pick_seed}
    t0 = time.perf_counter()
    market = Market(args.data_dir, args.cutoff, users)
    print(f'market loaded in {time.perf_counter() - t0:.0f}s; users: {len(users)}', flush=True)
    end = (market.cutoff + timedelta(days=args.horizon_days)).isoformat()
    popular_fallback = starter.order(pop)
    if generator is None and not args.no_llm:
        from runtime import LoggedGenerator
        generator = LoggedGenerator(ssr.ModelGenerator(
            args.model, args.revision, args.device, args.allow_download, dtype=args.dtype,
            temperature=0.0, top_p=1.0, thinking=False, seed=0,
            context_limit=args.context_limit), args.output_dir)
    log = args.output_dir / 'llm_calls.jsonl'
    if log.exists():
        log.unlink()
    predictions, failures, details, timing = {}, [], {}, Counter()
    for uid in users:
        f = client_facts(uid, market, assets, set(contexts[uid]['eligible_asset_ids']))
        pool = build_pool(uid, f, market, assets, history, contexts[uid])
        src_of = {e['isin']: e['source'] for e in pool}
        enc2 = formula_weights(f)
        order1, order2 = rank_by_weights(pool, LEX), rank_by_weights(pool, weights_of(enc2))
        preds = {METHODS['A1']: fill10(order1, popular_fallback),
                 METHODS['A2']: fill10(order2, popular_fallback)}
        det = {'n_buys': len(f['events']), 'n_new': len(f['new']), 'main_category': f['main'],
               'formula': {k: round(v, 3) for k, v in enc2.items()},
               'pool': [[e['source'], e['rank'], e['isin'], name_of(assets, e['isin'], 30)] for e in pool]}
        if generator is not None:
            block = client_block(f, market)
            prompt = encoder_prompt(f, market, assets, args.horizon_days, end)
            t = time.perf_counter()
            resp = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
            timing['A3'] += time.perf_counter() - t
            enc3, repaired = parse_encoder(resp['text'])
            ok3 = enc3 is not None
            if not ok3:
                failures.append({'user_id': uid, 'arm': 'A3', 'error': 'invalid encoder output'})
                enc3 = dict(enc2, confidence=None)
            ssr.append_jsonl(log, {'user_id': uid, 'arm': 'A3', 'prompt': prompt, 'raw': resp['text'],
                                   'valid': ok3, 'repaired': repaired, 'seconds': resp.get('seconds')})
            order3 = rank_by_weights(pool, weights_of(enc3))
            preds[METHODS['A3']] = fill10(order3, popular_fallback)
            det['encoder'] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in enc3.items()}
            det['encoder_valid'] = ok3
            profile = (f"\nPROFILE from the encoder: expected share of new assets {enc3['new_share']:.2f}; "
                       f"new purchases split popular {enc3['popular']:.2f}, similar {enc3['similar']:.2f}, "
                       f"recent gainers {enc3['recent_gainers']:.2f}")
            for arm, cand, control, line in (('A4', order1, order1, ''), ('A5', order3[:20], order3, profile)):
                rows, ids = candidate_rows(cand, src_of, f, market, assets)
                k = min(10, len(cand))
                prompt = reranker_prompt(block, line, rows, len(cand), k, args.horizon_days, market.cutoff, end)
                t = time.perf_counter()
                resp = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
                timing[arm] += time.perf_counter() - t
                picked, ok = parse_ranking(resp['text'], set(ids), k)
                if not ok:
                    failures.append({'user_id': uid, 'arm': arm, 'error': 'invalid reranker output'})
                preds[METHODS[arm]] = fill10([ids[i] for i in picked], list(control) + popular_fallback)
                ssr.append_jsonl(log, {'user_id': uid, 'arm': arm, 'prompt': prompt, 'raw': resp['text'],
                                       'valid': ok, 'filled': 10 - len(picked), 'seconds': resp.get('seconds')})
                det[arm + '_valid'] = ok
        for ranking in preds.values():
            starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
        predictions[uid] = preds
        details[uid] = det
        print(f'[{time.strftime("%H:%M:%S")}] {uid[:6]} done', flush=True)
    (args.output_dir / 'details.json').write_text(
        json.dumps(details, ensure_ascii=False, indent=1, default=str), encoding='utf-8')
    meta = dict(generator.metadata if generator is not None else {'backend': 'no_llm'},
                method='hybrid', cutoff=args.cutoff, prior=PRIOR, shrink=SHRINK,
                max_new_tokens=args.max_new_tokens,
                seconds_by_arm={k: round(v, 1) for k, v in timing.items()},
                invalid_by_arm=dict(Counter(x['arm'] for x in failures)),
                prompts={'encoder': ENCODER_PROMPT, 'reranker': RERANKER_PROMPT})
    data_adapter.save_results(args, bundle, predictions, failures, 'hybrid', meta)
    per = json.loads((args.output_dir / 'per_user.json').read_text(encoding='utf-8'))
    table = defaultdict(dict)
    for r in per:
        table[r['user_id']][r['method']] = r['ndcg_at_10']
    print('\nper-user nDCG@10:')
    for uid in users:
        print(f'  {uid[:6]} purchases in window={len(truth.get(uid, set()))}: ' +
              ', '.join(f'{m}={v:.3f}' for m, v in sorted(table[uid].items())))
    return meta


def build_parser():
    p = data_adapter.arguments('Hybrid LLM encoder + reranker over a four-source pool')
    p.add_argument('--pick', type=int, default=0, help='0 = all users of the sample')
    p.add_argument('--pick-from', choices=['all', 'buyers'], default='all',
                   help='buyers = users with purchases in the window (validation smoke tests only)')
    p.add_argument('--pick-seed', type=int, default=0)
    p.add_argument('--max-new-tokens', type=int, default=80)
    p.add_argument('--dtype', default='auto')
    p.add_argument('--context-limit', type=int, default=8192)
    p.add_argument('--no-llm', action='store_true', help='only A1, A2 and the baselines')
    p.add_argument('--fake-llm', action='store_true', help='pipeline check with a deterministic fake model')
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.fake_llm and args.no_llm:
        raise SystemExit('--fake-llm and --no-llm are exclusive')
    bundle = data_adapter.load(args)
    run(args, bundle, FakeGenerator() if args.fake_llm else None)


if __name__ == '__main__':
    main()
