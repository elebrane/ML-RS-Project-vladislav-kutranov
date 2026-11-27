"""Turn the saved GDELT responses into the daily information field and the headline store.

news/daily_infofield.csv  isin, day (UTC), articles, tone          (from timelinevolraw/timelinetone)
news/headlines.jsonl      one row per (asset, article): source_id, url, domain, language, seendate,
                          title, title_match (an alias of the asset is in the original-language title)
news/coverage.json        coverage of the candidate stocks at each cohort cutoff
news/linking_check_sample.csv  50 random matched headlines to label by hand (precision of the linking)
"""
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import news

HERE = Path(__file__).resolve().parent
NEWS = HERE / 'news'
RAW = NEWS / 'raw'


def series(resp):
    out = {}
    try:
        for p in resp['timeline'][0]['data']:
            out[p['date'][:4] + '-' + p['date'][4:6] + '-' + p['date'][6:8]] = p['value']
    except (KeyError, IndexError, TypeError):
        pass
    return out


def main():
    aliases = news.Aliases.load()
    daily, rows, missing = defaultdict(dict), [], []
    for isin in aliases.rows:
        d = RAW / isin
        if not d.exists():
            missing.append(isin)
            continue
        vol, tone = {}, {}
        vol_files = [f for f in sorted(d.glob('timelinevolraw_*.json'))
                     if '_error' not in (json.loads(f.read_text())['response'] or {})]
        if not vol_files:
            missing.append(isin)
            continue
        for f in vol_files:
            vol.update(series(json.loads(f.read_text())['response']))
        for f in sorted(d.glob('timelinetone_*.json')):
            resp = json.loads(f.read_text())['response'] or {}
            if '_error' not in resp:
                tone.update(series(resp))
        for day in sorted(set(vol) | set(tone)):
            daily[isin][day] = (int(vol.get(day, 0) or 0), tone.get(day))
        seen = set()
        for f in sorted(d.glob('artlist_*.json')):
            resp = json.loads(f.read_text())['response'] or {}
            if '_error' in resp:
                continue
            for a in resp.get('articles', []) or []:
                url = a.get('url') or ''
                if not url or url in seen:
                    continue
                seen.add(url)
                rows.append({'isin': isin, 'source_id': 'N' + hashlib.sha1(url.encode()).hexdigest()[:10],
                             'url': url, 'domain': a.get('domain'), 'language': a.get('language'),
                             'sourcecountry': a.get('sourcecountry'), 'seendate': a.get('seendate'),
                             'title': (a.get('title') or '').strip(),
                             'title_match': aliases.matches(isin, a.get('title') or ''),
                             'file': f.name})
    with (NEWS / 'daily_infofield.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['isin', 'day', 'articles', 'tone'])
        for isin in sorted(daily):
            for day, (a, t) in sorted(daily[isin].items()):
                w.writerow([isin, day, a, '' if t is None else t])
    with (NEWS / 'headlines.jsonl').open('w', encoding='utf-8') as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + '\n')
    matched = [r for r in rows if r['title_match']]
    rnd = random.Random(0)
    sample = rnd.sample(matched, min(50, len(matched)))
    with (NEWS / 'linking_check_sample.csv').open('w', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        w.writerow(['isin', 'asset_name', 'seendate', 'domain', 'title', 'url', 'about_this_company_1_0'])
        for r in sample:
            w.writerow([r['isin'], aliases.rows[r['isin']]['name'], r['seendate'], r['domain'], r['title'], r['url'], ''])
    # coverage at each cohort cutoff, over the stocks that appear in clients' blocks
    field = news.InfoField.load(NEWS)
    cov = {}
    for blocks_path in sorted((HERE / 'outputs').glob('*/blocks.json')):
        proto = json.loads((blocks_path.parent / 'protocol.json').read_text())
        cutoff = proto['cutoff']
        blocks = json.loads(blocks_path.read_text())
        stocks = sorted({i for b in blocks.values() if b['slots'] > 0 for i in b['block'] if field.tracked(i)})
        cards = {i: field.card(i, cutoff) for i in stocks}
        cov[blocks_path.parent.name] = {
            'cutoff': cutoff, 'tracked_stocks_in_blocks': len(stocks),
            'with_articles_30d': sum(c['articles30'] > 0 for c in cards.values()),
            'with_headline': sum(bool(c['headlines']) for c in cards.values()),
            'median_articles_30d': sorted(c['articles30'] for c in cards.values())[len(cards) // 2] if cards else None}
    summary = {'assets': len(aliases.rows), 'assets_without_raw': missing,
               'daily_rows': sum(len(v) for v in daily.values()), 'headlines': len(rows),
               'headlines_title_matched': len(matched),
               'title_match_rate': round(len(matched) / len(rows), 3) if rows else None,
               'coverage': cov}
    (NEWS / 'coverage.json').write_text(json.dumps(summary, indent=1) + '\n')
    print(json.dumps(summary, indent=1))
    if missing:
        print(f'note: {len(missing)} low-weight assets not collected; they are treated as untracked (no news lines)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
