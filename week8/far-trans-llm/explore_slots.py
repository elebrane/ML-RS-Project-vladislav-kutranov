"""Repeat-first Top-10 with LLM-chosen exploration slots (HW2, revised research question).

RQ: We test whether letting an LLM choose which never-traded assets fill the exploration
slots of a repeat-first Top-10 - picking from a shortlist of recently popular assets in the
customer's own categories - improves nDCG@10 over filling the same slots by popularity and
category match, for the common FAR-Trans evaluation users.

Pipeline per user. Steps 1-3 are code and identical in every arm; only step 4 differs.
 1. Repeat part: assets the user bought before the cutoff (latest ``history_events`` Buy
    events), ordered by buy count, then last buy date, then ISIN (freq_recency).
 2. Exploration slots: k Top-10 places for assets the user never bought or sold. Free
    places (fewer than ten own assets) always go to new assets. A propensity model
    P(new asset | the customer buys in the next 30 days), fitted only on 30-day windows that
    end on or before the cutoff, may take more places from the bottom of the own list:
    k = max(10 - n_own, round(10 * p)).
 3. Shortlist: eligible assets the user never traded that somebody bought in the previous
    365 days, ordered by: in the user's main category first, then distinct buyers in the last
    30 days, then in the last 365 days, then ISIN. The first ``shortlist_size`` are kept.
 4. Who fills the k slots from the shortlist:
      control_popularity      the shortlist order itself (no LLM)
      llm                     the LLM; shortlist shown in a seeded random order with neutral ids
      llm_masked_names        as llm, asset names replaced by neutral labels
      llm_shuffled_shortlist  as llm with a second random order (position sensitivity)
The final list is the own assets first (repeat-first), then the k chosen new assets.
Also reported: freq_recency_cbf (the earlier HW2 control: own assets, then CBF-new assets)
and the four starter baselines.

Truth is used only for evaluation and diagnostics, never in a prompt or a feature.
Teaching experiment, not investment advice.
"""
from __future__ import annotations

import argparse
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
import self_structured_ranker as ssr

ARMS = ('control_popularity', 'llm', 'llm_masked_names', 'llm_shuffled_shortlist')
LLM_ARMS = ('llm', 'llm_masked_names', 'llm_shuffled_shortlist')
METHOD = {'control_popularity': 'explore_control_popularity', 'llm': 'explore_llm',
          'llm_masked_names': 'explore_llm_masked_names',
          'llm_shuffled_shortlist': 'explore_llm_shuffled_shortlist'}
START = '2019-01-01'          # exploration history is counted from here (2018 = burn-in)
FEATURES = ('log_days_since_buy', 'log_days_since_txn', 'log_buy_days', 'log_distinct',
            'log_buys_30d', 'log_sells_30d', 'past_new_share', 'few_past_buys')

INSTRUCTION = """You are ranking investment assets for one bank customer. This is an experiment, not investment advice. Use only the supplied facts.
The customer has never bought or sold any asset in SHORTLIST. Choose the {k} SHORTLIST assets this customer is most likely to buy for the first time in the {horizon} days after the cutoff date, most likely first.
HOLDINGS lists assets the customer traded before the cutoff, most recent first. buyers_7d and buyers_30d count distinct customers of the bank who bought the asset in the 7 and 30 days before the cutoff.
Answer with only a JSON array of exactly {k} ids from SHORTLIST, for example ["N03", "N11"]."""
DEFAULT_PROMPTS = {'instruction': INSTRUCTION}
PROMPTS = dict(DEFAULT_PROMPTS)


def set_prompts(overrides):
    unknown = sorted(set(overrides or {}) - set(DEFAULT_PROMPTS))
    if unknown:
        raise ValueError(f'Unknown prompt keys: {unknown}')
    PROMPTS.clear()
    PROMPTS.update(DEFAULT_PROMPTS)
    PROMPTS.update(overrides or {})


# ---------------------------------------------------------------- dates
def add_days(day, n):
    return (date.fromisoformat(day) + timedelta(days=n)).isoformat()


def days_between(earlier, later):
    return (date.fromisoformat(later) - date.fromisoformat(earlier)).days


def month_starts(first, last_inclusive):
    out, d = [], date.fromisoformat(first)
    last = date.fromisoformat(last_inclusive)
    while d <= last:
        out.append(d.isoformat())
        d = date(d.year + (d.month == 12), d.month % 12 + 1, 1)
    return out


# ---------------------------------------------------------------- propensity model
def window_rows(evs, cutoffs, horizon, with_labels=True):
    """Features (and 30-day labels) of one customer at each cutoff, from earlier events only.

    ``evs`` is the customer's sorted list of (date, isin, kind). Cutoffs where the customer
    has no Buy before the cutoff are skipped (the course evaluates warm users only).
    Labels: any_buy = a Buy in [cutoff, cutoff + horizon); new_buy = such a Buy of an asset
    the customer never traded before the cutoff.
    """
    i, n = 0, len(evs)
    last_buy = last_txn = None
    buy_days, first, buy_events = set(), {}, set()
    new_events = 0
    out = []
    for k in cutoffs:
        while i < n and evs[i][0] < k:
            day, isin, kind = evs[i]
            first.setdefault(isin, day)
            if kind == 'Buy':
                last_buy = day
                buy_days.add(day)
                if day >= START and (day, isin) not in buy_events:
                    buy_events.add((day, isin))
                    new_events += first[isin] == day
            last_txn = day
            i += 1
        if last_buy is None:
            continue
        lo = add_days(k, -30)
        buys30 = sells30 = 0
        j = i - 1
        while j >= 0 and evs[j][0] >= lo:
            if evs[j][2] == 'Buy':
                buys30 += 1
            else:
                sells30 += 1
            j -= 1
        hist = len(buy_events)
        feats = [math.log1p(days_between(last_buy, k)), math.log1p(days_between(last_txn, k)),
                 math.log1p(len(buy_days)), math.log1p(len(first)), math.log1p(buys30),
                 math.log1p(sells30), new_events / hist if hist >= 3 else 0.0,
                 1.0 if hist < 3 else 0.0]
        any_buy = new_buy = 0
        if with_labels:
            end, j = add_days(k, horizon), i
            while j < n and evs[j][0] < end:
                if evs[j][2] == 'Buy':
                    any_buy = 1
                    if evs[j][1] not in first:
                        new_buy = 1
                j += 1
        out.append((k, feats, any_buy, new_buy))
    return out


def fit_logistic(X, y, lam=1.0, iters=50):
    import numpy as np
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mu, sd = X.mean(axis=0), X.std(axis=0)
    sd[sd == 0] = 1.0
    Z = np.hstack([np.ones((len(X), 1)), (X - mu) / sd])
    w = np.zeros(Z.shape[1])
    reg = lam * np.eye(Z.shape[1])
    reg[0, 0] = 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-Z @ w))
        grad = Z.T @ (p - y) + reg @ w
        hess = (Z * (p * (1 - p))[:, None]).T @ Z + reg
        step = np.linalg.solve(hess, grad)
        w -= step
        if np.abs(step).max() < 1e-8:
            break
    return {'mean': mu.tolist(), 'scale': sd.tolist(), 'weights': w.tolist()}


def predict_logistic(model, feats):
    z = model['weights'][0] + sum(w * (f - m) / s for w, f, m, s in zip(
        model['weights'][1:], feats, model['mean'], model['scale']))
    return 1 / (1 + math.exp(-z))


def auc(labels, scores):
    pairs = sorted(zip(scores, labels))
    ranks, i = [0.0] * len(pairs), 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        for t in range(i, j + 1):
            ranks[t] = (i + j) / 2 + 1
        i = j + 1
    pos = sum(lab for _s, lab in pairs)
    neg = len(pairs) - pos
    if not pos or not neg:
        return None
    return (sum(r for r, (_s, lab) in zip(ranks, pairs) if lab) - pos * (pos + 1) / 2) / (pos * neg)


def fit_propensity(by_customer, cutoff, horizon, first_cutoff):
    """P(new asset | buys) on windows whose label period ends on or before ``cutoff``."""
    last = add_days(cutoff, -horizon)
    cutoffs = [k for k in month_starts(first_cutoff, last) if add_days(k, horizon) <= cutoff]
    if len(cutoffs) < 4:
        raise ValueError('Need at least four training cutoffs before the evaluation cutoff')
    rows = []
    for evs in by_customer.values():
        rows.extend(r for r in window_rows(evs, cutoffs, horizon) if r[2])
    hold = set(cutoffs[-3:])
    train = [r for r in rows if r[0] not in hold]
    test = [r for r in rows if r[0] in hold]
    check = fit_logistic([r[1] for r in train], [r[3] for r in train])
    holdout_auc = auc([r[3] for r in test], [predict_logistic(check, r[1]) for r in test])
    model = fit_logistic([r[1] for r in rows], [r[3] for r in rows])
    model.update({'features': list(FEATURES), 'target': 'new asset in the window | any Buy',
                  'training_cutoffs': [cutoffs[0], cutoffs[-1]], 'training_windows': len(rows),
                  'positive_share': round(sum(r[3] for r in rows) / len(rows), 4),
                  'holdout_cutoffs': sorted(hold), 'holdout_windows': len(test),
                  'holdout_auc': None if holdout_auc is None else round(holdout_auc, 4)})
    return model


# ---------------------------------------------------------------- per-user parts
def customer_events(events):
    return {cid: sorted(evs) for cid, evs in events.items()}


def buyer_counts(by_customer, cutoff, days):
    lo = add_days(cutoff, -days)
    buyers = defaultdict(set)
    for cid, evs in by_customer.items():
        for day, isin, kind in evs:
            if kind == 'Buy' and lo <= day < cutoff:
                buyers[isin].add(cid)
    return Counter({isin: len(c) for isin, c in buyers.items()})


def own_order(window):
    counts, last = Counter(i for _d, i in window), {}
    for day, isin in window:
        last[isin] = max(day, last.get(isin, day))
    return sorted(counts, key=lambda i: (-counts[i], -date.fromisoformat(last[i]).toordinal(), i)), counts


def main_category(traded, assets, buy_counts):
    cats = Counter()
    weight = Counter()
    for isin in traded:
        cat = assets.get(isin, {}).get('assetCategory') or 'unknown'
        cats[cat] += 1
        weight[cat] += buy_counts.get(isin, 0)
    if not cats:
        return None
    return sorted(cats, key=lambda c: (-cats[c], -weight[c], c))[0]


def shortlist_for(eligible, traded, assets, main_cat, b30, b365, size):
    pool = [i for i in eligible if i not in traded and b365.get(i, 0) > 0]
    pool.sort(key=lambda i: (0 if assets.get(i, {}).get('assetCategory') == main_cat else 1,
                             -b30.get(i, 0), -b365.get(i, 0), i))
    return pool[:size]


def slots(n_own, p):
    return max(10 - min(n_own, 10), min(10, int(round(10 * p))))


def compose(own, chosen, k, fill):
    """Own assets first, then the chosen new ones; topped up from ``fill`` to ten unique."""
    ranking = own[:10 - k] + [i for i in chosen[:k] if i not in own[:10 - k]]
    for isin in list(own[10 - k:]) + list(fill):
        if len(ranking) >= 10:
            break
        if isin not in ranking:
            ranking.append(isin)
    if len(ranking) < 10:
        raise ValueError('Fewer than ten assets for the final list')
    return ranking[:10]


def prepare(args, bundle, propensity, by_customer):
    starter, assets, history, _truth, contexts, _baseline, events, pop, manifest = bundle
    cutoff = args.cutoff
    b7, b30, b365 = (buyer_counts(by_customer, cutoff, d) for d in (7, args.popularity_days, 365))
    features = {isin: starter.tokens(row) for isin, row in assets.items()}
    users = {}
    for uid in manifest['users']:
        evs = by_customer.get(uid, [])
        window = ssr.buy_window(events.get(uid, ()), cutoff, assets, args.history_events)
        own, counts = own_order(window)
        own = own[:args.own_max]
        traded, last_trade = set(), {}
        for day, isin, _kind in evs:
            traded.add(isin)
            last_trade[isin] = max(day, last_trade.get(isin, day))
        recent = sorted(traded, key=lambda i: (-date.fromisoformat(last_trade[i]).toordinal(), i))
        eligible = contexts[uid]['eligible_asset_ids']
        cat = main_category(traded, assets, history.get(uid, Counter()))
        short = shortlist_for(eligible, traded, assets, cat, b30, b365, args.shortlist_size)
        feats = window_rows(evs, [cutoff], args.horizon_days, with_labels=False)
        p = predict_logistic(propensity, feats[0][1]) if feats else 1.0
        k = min(slots(len(own), p), len(short))
        hist_counts = history.get(uid, Counter())
        scores = starter.content_scores(hist_counts, features)
        if not hist_counts or not any(scores.values()):
            scores = pop
        cbf = starter.order({i: scores[i] for i in eligible})
        by_pop = starter.order({i: pop[i] for i in eligible})
        users[uid] = {'uid': uid, 'cutoff': cutoff, 'own': own, 'buy_counts': counts,
                      'traded': traded, 'last_trade': last_trade, 'recent': recent,
                      'main_category': cat, 'shortlist': short, 'p_new': p, 'k': k,
                      'cbf_new': [i for i in cbf if i not in traded],
                      'fill': [i for i in by_pop if i not in own],
                      'b7': {i: b7.get(i, 0) for i in short}, 'b30': {i: b30.get(i, 0) for i in short}}
    return users


# ---------------------------------------------------------------- prompts and parsing
def shown_order(parts, seed):
    return sorted(parts['shortlist'], key=lambda i: (ssr.hashed(f'{seed}:{parts["uid"]}', i), i))


def payload(parts, assets, shown, mask, holdings_shown):
    cutoff = parts['cutoff']
    ids = {isin: f'N{n:02d}' for n, isin in enumerate(shown, 1)}
    holdings = []
    for n, isin in enumerate(parts['recent'][:holdings_shown], 1):
        card = ssr.asset_card(isin, assets, cutoff, False, f'H{n:02d}')
        card.pop('id')
        if mask:
            card['name'] = f'Held asset {n}'
        card['times_bought'] = parts['buy_counts'].get(isin, 0)
        card['last_trade_date'] = parts['last_trade'][isin]
        holdings.append(card)
    shortlist = []
    for isin in shown:
        card = ssr.asset_card(isin, assets, cutoff, mask, ids[isin])
        card['buyers_7d'] = parts['b7'][isin]
        card['buyers_30d'] = parts['b30'][isin]
        shortlist.append(card)
    data = {'cutoff_date_exclusive': cutoff, 'HOLDINGS': holdings, 'SHORTLIST': shortlist}
    return data, ids


def compose_prompt(data, k, horizon):
    instruction = PROMPTS['instruction'].replace('{k}', str(k)).replace('{horizon}', str(horizon))
    return '\n'.join([instruction, 'DATA:', json.dumps(data, ensure_ascii=False)])


def parse_ids(raw, allowed, k):
    """Return (ids in answer order, valid, repaired). Valid = exactly k unique allowed ids."""
    if type(raw) is not str:
        raise ValueError('Response must be text')
    text = raw.strip()
    repaired = False
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        repaired = True
        start, end = text.find('['), text.rfind(']')
        value = None
        if 0 <= start < end:
            try:
                value = json.loads(text[start:end + 1])
            except json.JSONDecodeError:
                value = None
        if value is None:
            value = re.findall(r'\bN\d{2}\b', text)
    if type(value) is not list:
        return [], False, repaired
    items = [v.strip() for v in value if type(v) is str]
    clean = []
    for v in items:
        if v in allowed and v not in clean:
            clean.append(v)
    valid = len(items) == len(value) == k and len(clean) == k
    return clean[:k], valid, repaired


# ---------------------------------------------------------------- run
def run(args, bundle, generator=None):
    starter, assets, _history, truth, contexts, _baseline, events, _pop, manifest = bundle
    arms = [a for a in args.arms.split(',') if a]
    if not arms or any(a not in ARMS for a in arms):
        raise ValueError(f'--arms must be a comma list from {ARMS}')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started_all = time.perf_counter()
    by_customer = customer_events(events)
    propensity = fit_propensity(by_customer, args.cutoff, args.horizon_days, args.propensity_start)
    (args.output_dir / 'propensity.json').write_text(json.dumps(propensity, indent=2) + '\n')
    users = prepare(args, bundle, propensity, by_customer)
    diag = diagnostics(users, truth, contexts)
    (args.output_dir / 'diagnostics.json').write_text(json.dumps(diag, indent=2) + '\n')
    print(json.dumps({'propensity_holdout_auc': propensity['holdout_auc'],
                      'training_windows': propensity['training_windows'], **diag}, indent=2))
    seeds = {'llm': args.shuffle_seed, 'llm_masked_names': args.shuffle_seed,
             'llm_shuffled_shortlist': args.second_seed}
    masks = {'llm': False, 'llm_masked_names': True, 'llm_shuffled_shortlist': False}

    if args.prepare_only:
        rows = []
        for uid, p in users.items():
            if p['k'] == 0:
                continue
            for arm in [a for a in arms if a in LLM_ARMS]:
                data, _ids = payload(p, assets, shown_order(p, seeds[arm]), masks[arm], args.holdings_shown)
                prompt = compose_prompt(data, p['k'], args.horizon_days)
                rows.append({'user_id': uid, 'arm': arm, 'k': p['k'], 'chars': len(prompt), 'prompt': prompt})
        ssr.write_jsonl(args.output_dir / 'prompts.jsonl', rows)
        summary = {'stage': 'prepare_only_no_inference', 'users': len(users), 'llm_calls': len(rows),
                   'prompt_chars': {'median': statistics.median([r['chars'] for r in rows]) if rows else 0,
                                    'max': max([r['chars'] for r in rows], default=0)},
                   'diagnostics': diag, 'propensity': propensity, 'prompts': dict(PROMPTS)}
        (args.output_dir / 'prepare_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        return summary

    if generator is None and any(a in LLM_ARMS for a in arms):
        from runtime import LoggedGenerator
        generator = LoggedGenerator(ssr.ModelGenerator(
            args.model, args.revision, args.device, args.allow_download, dtype=args.dtype,
            temperature=args.temperature, top_p=args.top_p, thinking=args.thinking,
            seed=args.sample_seed, context_limit=args.context_limit), args.output_dir)
    log = args.output_dir / 'arm_responses.jsonl'
    if log.exists():
        log.unlink()
    predictions, failures, timing = {}, [], Counter()
    valid = {a: set() for a in arms}
    chosen_new = defaultdict(dict)
    for n_user, (uid, p) in enumerate(users.items(), 1):
        preds = {'freq_recency_cbf': compose(p['own'], p['cbf_new'], max(0, 10 - len(p['own'])), p['fill'])}
        k = p['k']
        for arm in arms:
            record = {'user_id': uid, 'arm': arm, 'k': k, 'raw_response': None, 'ids': None,
                      'valid': None, 'parse_repaired': False, 'filled_from_control': 0,
                      'error': None, 'seconds': 0.0}
            t0 = time.perf_counter()
            chosen = list(p['shortlist'])
            if arm in LLM_ARMS and k > 0:
                try:
                    data, ids = payload(p, assets, shown_order(p, seeds[arm]), masks[arm], args.holdings_shown)
                    isin_of = {v: i for i, v in ids.items()}
                    response = generator.generate(compose_prompt(data, k, args.horizon_days),
                                                  max_new_tokens=args.max_new_tokens)
                    record['raw_response'] = response['text']
                    record['seconds'] = response.get('seconds', 0.0)
                    picked, ok, repaired = parse_ids(response['text'], set(ids.values()), k)
                    record.update(ids=picked, valid=ok, parse_repaired=repaired)
                    picked_isin = [isin_of[i] for i in picked]
                    if not ok:
                        failures.append({'user_id': uid, 'arm': arm, 'error': 'invalid answer'})
                        if args.on_invalid == 'error':
                            raise ValueError(f'{uid}/{arm}: invalid answer {response["text"]!r}')
                    else:
                        valid[arm].add(uid)
                    rest = [i for i in p['shortlist'] if i not in picked_isin]
                    record['filled_from_control'] = k - len(picked_isin)
                    chosen = picked_isin + rest
                except (ValueError, RuntimeError) as exc:
                    if args.on_invalid == 'error':
                        ssr.append_jsonl(log, record)
                        raise
                    record['error'] = f'{type(exc).__name__}: {exc}'
                    failures.append({'user_id': uid, 'arm': arm, 'error': record['error']})
                    record['filled_from_control'] = k
            else:
                valid[arm].add(uid)      # control, or no exploration slot for this user
            ranking = compose(p['own'], chosen, k, p['fill'])
            starter.validate_ranking(ranking, contexts[uid]['eligible_asset_ids'])
            preds[METHOD[arm]] = ranking
            chosen_new[arm][uid] = ranking[10 - k:] if k else []
            timing[arm] += time.perf_counter() - t0
            record['wall_seconds'] = round(time.perf_counter() - t0, 3)
            ssr.append_jsonl(log, record)
        predictions[uid] = preds
        if n_user % 10 == 0 or n_user == len(users):
            print(f'[{time.strftime("%H:%M:%S")}] {n_user}/{len(users)} users', flush=True)

    details = slot_report(users, truth, chosen_new, arms)
    (args.output_dir / 'slot_hits.json').write_text(json.dumps(details, indent=2) + '\n')
    valid_only = {}
    base_method = METHOD['control_popularity'] if 'control_popularity' in arms else 'freq_recency_cbf'
    for arm in arms:
        subset = sorted(valid[arm])
        valid_only[METHOD[arm]] = {'valid_users': len(subset), 'paired': {}}
        for name in sorted({METHOD[arm], base_method, 'freq_recency_cbf'}):
            recs = [{'ndcg_at_10': starter.ndcg10(predictions[u][name], truth.get(u, set())),
                     'relevant_count': len(truth.get(u, set()))} for u in subset]
            valid_only[METHOD[arm]]['paired'][name] = starter.aggregate(recs)
    (args.output_dir / 'metrics_valid_only.json').write_text(json.dumps(valid_only, indent=2) + '\n')
    gen_meta = generator.metadata if generator is not None else {'backend': 'no_llm_arms'}
    metadata = dict(gen_meta, method='explore_slots', arms=arms, cutoff=args.cutoff,
                    history_events=args.history_events, own_max=args.own_max,
                    shortlist_size=args.shortlist_size, popularity_days=args.popularity_days,
                    holdings_shown=args.holdings_shown, shuffle_seed=args.shuffle_seed,
                    second_seed=args.second_seed, max_new_tokens=args.max_new_tokens,
                    wall_seconds_by_arm={a: round(t, 1) for a, t in timing.items()},
                    total_wall_seconds=round(time.perf_counter() - started_all, 1),
                    invalid_by_arm={a: sum(1 for f in failures if f['arm'] == a) for a in arms},
                    diagnostics=diag, propensity={k: propensity[k] for k in propensity
                                                  if k not in ('mean', 'scale', 'weights')},
                    run_name=args.run_name or args.output_dir.name, prompts=dict(PROMPTS),
                    prompts_edited=sorted(k for k in PROMPTS if PROMPTS[k] != DEFAULT_PROMPTS[k]))
    data_adapter.save_results(args, bundle, predictions, failures, 'explore_slots', metadata)
    return metadata


def diagnostics(users, truth, contexts):
    """Evaluation-only facts; nothing here enters a prompt."""
    t_users = [u for u in users if truth.get(u)]
    rows = []
    for uid in t_users:
        p = users[uid]
        rel = truth[uid] & set(contexts[uid]['eligible_asset_ids'])
        new = {i for i in rel if i not in p['traded']}
        rows.append({'truth': len(rel), 'new': len(new), 'new_in_shortlist': len(new & set(p['shortlist'])),
                     'repeat_in_own_top': len((rel - new) & set(p['own'][:10 - p['k']]))})
    ks = [p['k'] for p in users.values()]
    return {'users': len(users), 'users_with_future_buys': len(t_users),
            'future_buy_assets': sum(r['truth'] for r in rows),
            'future_new_assets': sum(r['new'] for r in rows),
            'future_new_in_shortlist': sum(r['new_in_shortlist'] for r in rows),
            'future_repeat_in_own_slots': sum(r['repeat_in_own_top'] for r in rows),
            'k_distribution': dict(sorted(Counter(ks).items())),
            'median_p_new': round(statistics.median(p['p_new'] for p in users.values()), 4),
            'users_with_llm_call': sum(1 for k in ks if k > 0)}


def slot_report(users, truth, chosen_new, arms):
    out = {}
    for arm in arms:
        hits = new_total = 0
        for uid, p in users.items():
            new_truth = {i for i in truth.get(uid, set()) if i not in p['traded']}
            new_total += len(new_truth)
            hits += len(new_truth & set(chosen_new[arm].get(uid, [])))
        out[METHOD[arm]] = {'new_assets_bought': new_total, 'caught_in_new_slots': hits}
    return out


# ---------------------------------------------------------------- arguments and config
def build_parser():
    parser = data_adapter.arguments('Repeat-first Top-10 with LLM-chosen exploration slots')
    for action in parser._actions:
        if action.dest == 'device':
            action.choices = ['auto', 'cpu', 'cuda', 'mps']
        if action.dest == 'on_invalid':
            action.default = 'fallback'
            action.help = 'fallback = fill missing slots from the control order and count the case'
    parser.add_argument('--config', type=Path, help='JSON config (see configs/explore_*.json)')
    parser.add_argument('--run-name')
    parser.add_argument('--arms', default=','.join(ARMS))
    parser.add_argument('--history-events', type=int, default=50)
    parser.add_argument('--own-max', type=int, default=30)
    parser.add_argument('--propensity-start', default='2019-07-01',
                        help='First monthly training cutoff of the propensity model')
    parser.add_argument('--shortlist-size', type=int, default=20)
    parser.add_argument('--popularity-days', type=int, default=30)
    parser.add_argument('--holdings-shown', type=int, default=10)
    parser.add_argument('--shuffle-seed', type=int, default=0)
    parser.add_argument('--second-seed', type=int, default=1)
    parser.add_argument('--max-new-tokens', type=int, default=96)
    parser.add_argument('--dtype', choices=['auto', 'bfloat16', 'float16', 'float32'], default='auto')
    parser.add_argument('--temperature', type=float, default=0.0)
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--sample-seed', type=int, default=0)
    parser.add_argument('--thinking', action='store_true')
    parser.add_argument('--context-limit', type=int, default=4096)
    parser.add_argument('--prepare-only', action='store_true',
                        help='Fit the propensity model and write prompts only; no LLM is loaded')
    return parser


CONFIG_TOP = ('run_name', 'data_dir', 'output_dir')
CONFIG_SECTIONS = ('protocol', 'pipeline', 'llm_input', 'generation')


def parse_args(argv=None):
    """Command-line flags win over --config values; --config wins over defaults."""
    parser = build_parser()
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument('--config', type=Path)
    known, _ = pre.parse_known_args(argv)
    if known.config:
        config = json.loads(known.config.read_text(encoding='utf-8'))
        unknown_top = sorted(set(config) - set(CONFIG_TOP) - set(CONFIG_SECTIONS) - {'arms', 'prompts', 'note'})
        if unknown_top:
            raise ValueError(f'Unknown config sections: {unknown_top}')
        flat = {k: config[k] for k in CONFIG_TOP if k in config}
        for section in CONFIG_SECTIONS:
            flat.update(config.get(section) or {})
        if 'arms' in config:
            flat['arms'] = ','.join(config['arms']) if isinstance(config['arms'], list) else config['arms']
        actions = {a.dest: a for a in parser._actions}
        unknown = sorted(set(flat) - set(actions))
        if unknown:
            raise ValueError(f'Unknown config keys: {unknown}')
        for key, value in flat.items():
            if key in ('data_dir', 'output_dir', 'starter_dir') and value is not None:
                value = Path(value)          # relative paths are relative to the working folder
            choices = actions[key].choices
            if choices and value not in choices:
                raise ValueError(f'Config {key}={value!r} is not one of {list(choices)}')
            actions[key].default = value
            actions[key].required = False
        set_prompts(config.get('prompts'))
    else:
        set_prompts({})
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.shortlist_size < 10 or args.history_events < 1:
        raise ValueError('shortlist-size must be >= 10 and history-events positive')
    random.seed(0)
    result = run(args, data_adapter.load(args))
    if args.prepare_only:
        print(json.dumps({k: result[k] for k in ('stage', 'users', 'llm_calls', 'prompt_chars')}, indent=2))
    else:
        print(json.dumps({'invalid_by_arm': result['invalid_by_arm'],
                          'wall_seconds_by_arm': result['wall_seconds_by_arm'],
                          'total_wall_seconds': result['total_wall_seconds']}, indent=2))


if __name__ == '__main__':
    main()
