"""Freeze the HW3 cohorts before any comparison: protocol.json, blocks.json, candidate stock list.

Writes outputs/<cohort>/protocol.json and blocks.json (no truth), and news/candidate_stocks.json
(every Stock that appears in some client's new-asset block), which the news collector reads.
"""
import json
import sys
import time
from collections import Counter
from pathlib import Path

import common

HERE = common.HERE


def main(names):
    cohorts = json.loads((HERE / 'configs' / 'cohorts.json').read_text())
    stocks = {}
    for name in names or list(cohorts):
        t0 = time.time()
        out = HERE / 'outputs' / name
        out.mkdir(parents=True, exist_ok=True)
        c = common.load_cohort(cohorts[name], out)
        blocks = common.blocks_for(c)
        assets = c['bundle'][1]
        (out / 'protocol.json').write_text(json.dumps(c['protocol'], indent=2) + '\n')
        (out / 'blocks.json').write_text(json.dumps(blocks, indent=1) + '\n')
        n_slots = sum(b['slots'] > 0 for b in blocks.values())
        cats = Counter(b['main_category'] for b in blocks.values() if b['slots'] > 0)
        for b in blocks.values():
            if b['slots'] > 0:
                for i in b['block']:
                    if assets[i].get('assetCategory') == 'Stock':
                        s = stocks.setdefault(i, {'isin': i, 'name': assets[i].get('assetName'),
                                                  'short': assets[i].get('assetShortName'),
                                                  'sector': assets[i].get('sector'),
                                                  'industry': assets[i].get('industry'), 'clients': Counter()})
                        s['clients'][name] += 1
        print(f"{name}: cutoff {c['cohort']['cutoff']}, users {len(c['users'])}, with new slots {n_slots}, "
              f"main category of those {dict(cats)}, protocol_hash {c['protocol']['protocol_hash'][:12]}, "
              f"hw3_protocol_hash {c['protocol']['hw3_protocol_hash'][:12]}, {time.time() - t0:.0f} s", flush=True)
    rows = sorted(stocks.values(), key=lambda s: -sum(s['clients'].values()))
    for s in rows:
        s['clients'] = dict(s['clients'])
    (HERE / 'news' / 'candidate_stocks.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
    print(f'candidate stocks: {len(rows)}')


if __name__ == '__main__':
    main(sys.argv[1:])
