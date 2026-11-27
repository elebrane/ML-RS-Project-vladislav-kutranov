"""Collect every outputs/<cohort>/<run>/hw3_summary.json into outputs/summary.md (tables for the report)."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
NAMES = {'popularity': 'Popularity', 'cbf': 'CBF', 'latest_buy': 'Latest buy', 'latest_buy_or_sell': 'Latest buy or sell',
         'c0_popularity': 'C0 initial order (matched control)', 'cnews_attention': 'C-news: attention (code)',
         'cnews_volume': 'C-news: volume (code, expl.)', 'cnews_tone': 'C-news: tone (code, expl.)',
         'llm_nonews': 'A1 LLM, no news', 'llm_numbers': 'A2 LLM, numbers only', 'llm_full': 'M LLM + news',
         'llm_shuffled': 'A3 LLM, shuffled news (placebo)', 'oracle_block': 'ceiling: perfect block order'}


def f4(x):
    return '—' if x is None else f'{x:.4f}'


def main():
    out = ['# HW3 development results (not official benchmark scores)', '']
    for path in sorted(HERE.glob('outputs/*/*/hw3_summary.json')):
        if path.parent.name.startswith('_'):
            continue
        s = json.loads(path.read_text())
        m = s['meta']
        out += [f"## {m['cohort']['name']} / {m['run_name']}",
                f"cutoff {m['cohort']['cutoff']}, users {m['users']}, protocol_hash `{m['protocol_hash'][:12]}`, "
                f"hw3_protocol_hash `{m['hw3_protocol_hash'][:12]}`", '',
                '| Method | nDCG@10 all | buyers | n | bought new | n |', '|---|---:|---:|---:|---:|---:|']
        for k, r in s['table'].items():
            out.append(f"| {NAMES.get(k, k)} | {f4(r['all'])} | {f4(r['buyers'])} | {r['n_buyers']} | "
                       f"{f4(r['bought_new'])} | {r['n_bought_new']} |")
        if s['contrasts']:
            out += ['', '| Contrast (all users) | Δ | 95% CI | better/worse | sign p |', '|---|---:|---|---|---:|']
            for c in s['contrasts']:
                x = c['all_users']
                out.append(f"| {c['question']}: {c['a']} − {c['b']} | {x['diff']:+.4f} | "
                           f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}] | {x['better']}/{x['worse']} | {x['sign_p']:.3f} |")
        for k in ('llm_status', 'news_coverage', 'lists_changed_vs_c0', 'prompt_chars', 'seconds'):
            if m.get(k):
                out.append(f"\n{k}: `{json.dumps(m[k])}`")
        for arm, dg in s.get('subset_diagnostics', {}).items():
            x = dg['llm_full_minus_it']
            out.append(f"\nsubset {arm}: users {dg['users']}, identical lists {dg['identical_lists']}, "
                       f"mean Kendall tau {dg['mean_kendall_tau']}, llm_full − it {x['diff']:+.4f} "
                       f"[{x['ci95'][0]:+.4f}, {x['ci95'][1]:+.4f}]")
        out.append('')
    (HERE / 'outputs' / 'summary.md').write_text('\n'.join(out) + '\n', encoding='utf-8')
    print('\n'.join(out))


if __name__ == '__main__':
    main()
