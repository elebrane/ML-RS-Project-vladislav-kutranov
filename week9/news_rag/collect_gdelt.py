"""Collect the GDELT DOC 2.0 news layer for the candidate stocks (run on the Mac; stdlib only).

Per asset (heaviest first): daily article counts (timelinevolraw) and daily average tone
(timelinetone) over --start ... --end, and the article list in the 30 days before each cutoff
(artlist, newest first; GDELT itself is asked to stop at 21:00 UTC on cutoff - 1). Every response is
saved unchanged under news/raw/<isin>/ with the URL and fetch time, so the run can be stopped and
restarted at any time and every number in the report traces back to a saved response.
GDELT asks for at most one request every 5 seconds; on HTTP 429 the script waits longer and retries.
"""
from __future__ import annotations

import argparse
import json
import ssl
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEWS = HERE / 'news'
RAW = NEWS / 'raw'
API = 'https://api.gdeltproject.org/api/v2/doc/doc'
UA = 'Mozilla/5.0 (FAR-Trans course project; teaching use)'
BACKOFF = [20, 40, 80, 160, 300, 300]


def ssl_context():
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        return ssl.create_default_context()


CTX = ssl_context()


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with (NEWS / 'collect.log').open('a', encoding='utf-8') as fh:
        fh.write(line + '\n')


def query_for(phrases, filt):
    """Quote multi-word phrases only; GDELT rejects short quoted phrases, so short single words are
    dropped when a longer alternative exists and otherwise left unquoted."""
    longer = [p for p in phrases if ' ' in p or len(p) >= 5]
    use = longer or phrases
    terms = [f'"{p}"' if ' ' in p else p for p in use]
    q = terms[0] if len(terms) == 1 else '(' + ' OR '.join(terms) + ')'
    return (q + ' ' + filt).strip()


def fetch(params, pause):
    url = API + '?' + urllib.parse.urlencode(params)
    for attempt, wait in enumerate([0] + BACKOFF):
        if wait:
            log(f'  wait {wait} s (attempt {attempt + 1})')
            time.sleep(wait)
        try:
            req = urllib.request.Request(url, headers={'User-Agent': UA})
            with urllib.request.urlopen(req, timeout=90, context=CTX) as r:
                body = r.read().decode('utf-8', errors='replace')
                status = r.status
        except urllib.error.HTTPError as e:
            status, body = e.code, ''
        except Exception as e:  # network errors: retry with backoff
            log(f'  network error: {type(e).__name__}: {e}')
            continue
        if 'too short' in body[:200] or 'too common' in body[:200]:
            log(f'  GDELT rejected the query: {body[:80]!r}')
            return url, {'_error': body[:200]}
        if status == 429 or body.startswith('Please limit'):
            continue
        if status != 200:
            log(f'  HTTP {status}')
            continue
        time.sleep(pause)
        text = body.strip()
        if not text:
            return url, {}
        try:
            return url, json.loads(text)
        except json.JSONDecodeError:
            log(f'  not JSON: {text[:120]!r}')
            return url, {'_error': 'not json', '_body': text[:2000]}
    return url, None


def save(path, url, params, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({'url': url, 'params': params,
                                'fetched_at': datetime.now(timezone.utc).isoformat(),
                                'response': data}, ensure_ascii=False) + '\n', encoding='utf-8')


def done(path):
    if not path.exists():
        return False
    try:
        return '_error' not in (json.loads(path.read_text()).get('response') or {})
    except (json.JSONDecodeError, AttributeError):
        return False


def gdelt_dt(d, hour=0):
    return d.strftime('%Y%m%d') + f'{hour:02d}0000'


def timeline(isin, mode, q, start, end, pause, depth=0):
    """Daily timeline; if GDELT answers at a coarser resolution, split the window in two."""
    path = RAW / isin / f'{mode}_{start.isoformat()}_{end.isoformat()}.json'
    if done(path):
        return True
    params = {'query': q, 'mode': mode, 'format': 'json',
              'startdatetime': gdelt_dt(start), 'enddatetime': gdelt_dt(end)}
    url, data = fetch(params, pause)
    if data is None or '_error' in data:
        log(f'  FAILED {mode} {start}..{end}')
        return False
    res = (data.get('query_details') or {}).get('date_resolution') if isinstance(data, dict) else None
    if res not in (None, 'day') and depth < 4 and (end - start).days > 20:
        mid = start + (end - start) / 2
        log(f'  resolution {res}: splitting {start}..{end}')
        return (timeline(isin, mode, q, start, mid, pause, depth + 1)
                and timeline(isin, mode, q, mid, end, pause, depth + 1))
    save(path, url, params, data)
    return True


def artlist(isin, q, cutoff, pause, maxrecords):
    path = RAW / isin / f'artlist_{cutoff.isoformat()}.json'
    if done(path):
        return True
    params = {'query': q, 'mode': 'artlist', 'format': 'json', 'sort': 'datedesc',
              'maxrecords': str(maxrecords),
              'startdatetime': gdelt_dt(cutoff - timedelta(days=31)),
              'enddatetime': gdelt_dt(cutoff - timedelta(days=1), 21)}
    url, data = fetch(params, pause)
    if data is None or '_error' in data:
        log(f'  FAILED artlist {cutoff}')
        return False
    save(path, url, params, data)
    return True


def total(data):
    try:
        return sum(p['value'] for p in data['timeline'][0]['data'])
    except (KeyError, IndexError, TypeError):
        return 0


def probe(rows, pause):
    """Pick the source filter once: Greek press if GDELT supports it for these names, else none."""
    state_path = RAW / 'state.json'
    if state_path.exists():
        return json.loads(state_path.read_text())['filter']
    probes = [r for r in rows if r['isin'] in ('GRS393503008', 'GRS014003032', 'GRS003003035')] or rows[:3]
    s, e = date(2021, 3, 1), date(2021, 5, 1)
    result = {}
    for filt in ('sourcecountry:greece', 'sourcelang:greek', ''):
        counts = []
        for r in probes:
            params = {'query': query_for(r['query_phrases'], filt), 'mode': 'timelinevolraw', 'format': 'json',
                      'startdatetime': gdelt_dt(s), 'enddatetime': gdelt_dt(e)}
            url, data = fetch(params, pause)
            counts.append(total(data) if data and '_error' not in data else 0)
            save(RAW / '_probe' / f"{r['isin']}_{filt.replace(':', '_') or 'none'}.json", url, params, data)
        result[filt or 'none'] = counts
        log(f'probe {filt or "no filter"}: articles Mar-Apr 2021 = {counts}')
        if sum(c > 0 for c in counts) >= 2:
            RAW.mkdir(parents=True, exist_ok=True)
            state_path.write_text(json.dumps({'filter': filt, 'probe': result}, indent=1) + '\n')
            return filt
    raise SystemExit('GDELT returned no articles for the probe names with any filter; stop and check.')


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    p.add_argument('--start', default='2020-11-01')
    p.add_argument('--end', default='2021-07-01')
    p.add_argument('--cutoffs', default='2021-04-01,2021-05-01,2021-06-01')
    p.add_argument('--pause', type=float, default=7.0)
    p.add_argument('--maxrecords', type=int, default=75)
    p.add_argument('--limit', type=int, default=0, help='only the first N assets (smoke test)')
    p.add_argument('--isins', default='', help='comma list: only these assets, in this order')
    p.add_argument('--modes', default='timelinevolraw,timelinetone,artlist')
    p.add_argument('--until', default='', help='stop starting new requests after HH:MM local time')
    a = p.parse_args(argv)
    rows = json.loads((NEWS / 'aliases.json').read_text(encoding='utf-8'))
    rows.sort(key=lambda r: -r['weight'])
    if a.limit:
        rows = rows[:a.limit]
    if a.isins:
        by = {r['isin']: r for r in rows}
        rows = [by[i] for i in a.isins.split(',') if i in by]
    modes = set(a.modes.split(','))
    RAW.mkdir(parents=True, exist_ok=True)
    log(f'start: {len(rows)} assets, timelines {a.start}..{a.end}, headline cutoffs {a.cutoffs}')
    filt = probe(rows, a.pause)
    log(f'source filter: {filt or "none"}')
    start, end = date.fromisoformat(a.start), date.fromisoformat(a.end)
    cutoffs = [date.fromisoformat(c) for c in a.cutoffs.split(',') if c]
    failed = []
    for n, r in enumerate(rows, 1):
        q = query_for(r['query_phrases'], filt)
        log(f"{n}/{len(rows)} {r['isin']} {r['name'][:40]} | {q}")
        if a.until and datetime.now().strftime('%H:%M') >= a.until:
            log(f'time limit {a.until} reached; stopping')
            failed += [x['isin'] for x in rows[n - 1:]]
            break
        ok = True
        if 'timelinevolraw' in modes:
            ok &= timeline(r['isin'], 'timelinevolraw', q, start, end, a.pause)
        if 'timelinetone' in modes:
            ok &= timeline(r['isin'], 'timelinetone', q, start, end, a.pause)
        if 'artlist' in modes:
            for c in cutoffs:
                ok &= artlist(r['isin'], q, c, a.pause, a.maxrecords)
        if not ok:
            failed.append(r['isin'])
    (NEWS / 'collect_summary.json').write_text(json.dumps(
        {'finished_at': datetime.now(timezone.utc).isoformat(), 'assets': len(rows), 'failed': failed,
         'filter': filt, 'start': a.start, 'end': a.end, 'cutoffs': a.cutoffs}, indent=1) + '\n')
    log(f'done; failed assets: {failed}')
    print('STATUS: ' + ('NEWS_DONE' if not failed else 'NEWS_PARTIAL'))
    return 0 if not failed else 3


if __name__ == '__main__':
    sys.exit(main())
