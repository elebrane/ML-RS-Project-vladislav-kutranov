"""News layer: daily per-asset information field (GDELT) and time-safe news cards at a cutoff.

Time boundary. GDELT days are UTC; Athens midnight of the cutoff day is 21:00-22:00 UTC of the day
before. Daily counts therefore use UTC days up to cutoff - 2 (30-day window C-31 ... C-2), and a
headline is used only if GDELT first saw it before 21:00 UTC on cutoff - 1. Both rules are asserted.
"""
from __future__ import annotations

import csv
import json
import re
import unicodedata
from collections import defaultdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEWS = HERE / 'news'
MASK = '[company]'


# ------------------------------------------------------------------ text matching and masking
def norm1(ch):
    base = unicodedata.normalize('NFD', ch)[0].lower()
    if base == 'ς':
        base = 'σ'
    return base if base.isalnum() else ' '


def normalize(text):
    """One output character per input character, so spans map back to the original title."""
    return ''.join(norm1(c) for c in text or '')


def compile_pattern(p):
    prefix = p.endswith('*')
    core = normalize(p.rstrip('*')).strip()
    core = re.sub(r'\s+', ' ', core)
    body = r'\s+'.join(re.escape(w) for w in core.split(' '))
    return re.compile(r'(?<!\S)' + body + (r'\S*' if prefix else r'(?!\S)'))


class Aliases:
    def __init__(self, rows):
        self.rows = {r['isin']: r for r in rows}
        self.pats = {r['isin']: [compile_pattern(p) for p in r['title_patterns']] for r in rows}

    @classmethod
    def load(cls, path=NEWS / 'aliases.json'):
        return cls(json.loads(Path(path).read_text(encoding='utf-8')))

    def matches(self, isin, title):
        t = normalize(title)
        return any(p.search(t) for p in self.pats.get(isin, ()))

    def mask(self, title, isins=None):
        """Replace every alias of the given assets (default: all) by [company] in the original title."""
        t = normalize(title)
        spans = []
        for i in (isins if isins is not None else self.pats):
            for p in self.pats.get(i, ()):
                spans += [m.span() for m in p.finditer(t)]
        if not spans:
            return title
        spans.sort()
        merged = [list(spans[0])]
        for a, b in spans[1:]:
            if a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        out, last = [], 0
        for a, b in merged:
            out.append(title[last:a])
            out.append(MASK)
            last = b
        out.append(title[last:])
        return re.sub(r'\s+', ' ', ''.join(out)).strip()


# ------------------------------------------------------------------ the information field
def boundary(cutoff):
    """Latest moment a headline may have been seen: 21:00 UTC on the day before the cutoff."""
    c = date.fromisoformat(cutoff)
    return datetime(c.year, c.month, c.day, tzinfo=timezone.utc) - timedelta(hours=3)


def seen_time(s):
    return datetime.strptime(s, '%Y%m%dT%H%M%SZ').replace(tzinfo=timezone.utc)


class InfoField:
    def __init__(self, daily, headlines, aliases):
        self.daily = daily                  # isin -> {day iso: (articles, tone)}
        self.headlines = headlines          # isin -> [row], newest first
        self.aliases = aliases
        self._total = {}

    @classmethod
    def load(cls, news_dir=NEWS):
        daily = defaultdict(dict)
        with open(Path(news_dir) / 'daily_infofield.csv', newline='', encoding='utf-8') as fh:
            for r in csv.DictReader(fh):
                tone = float(r['tone']) if r['tone'] not in ('', 'None') else None
                daily[r['isin']][r['day']] = (int(r['articles']), tone)
        heads = defaultdict(list)
        with open(Path(news_dir) / 'headlines.jsonl', encoding='utf-8') as fh:
            for line in fh:
                r = json.loads(line)
                if r['title_match']:
                    heads[r['isin']].append(r)
        for v in heads.values():
            v.sort(key=lambda r: r['seendate'], reverse=True)
        return cls(dict(daily), dict(heads), Aliases.load(Path(news_dir) / 'aliases.json'))

    def tracked(self, isin):
        return isin in self.daily

    def _window(self, isin, cutoff, first, last):
        """Sum of articles and article-weighted tone over UTC days cutoff-first ... cutoff-last."""
        c = date.fromisoformat(cutoff)
        d = self.daily.get(isin, {})
        arts, tone_sum = 0, 0.0
        for k in range(last, first + 1):
            day = (c - timedelta(days=k)).isoformat()
            a, t = d.get(day, (0, None))
            arts += a
            if a and t is not None:
                tone_sum += a * t
        return arts, (tone_sum / arts if arts else None)

    def total30(self, cutoff):
        if cutoff not in self._total:
            self._total[cutoff] = sum(self._window(i, cutoff, 31, 2)[0] for i in self.aliases.rows)
        return self._total[cutoff]

    def card(self, isin, cutoff, n_headlines=2):
        """Numbers and headlines for one tracked stock, all strictly before the cutoff boundary."""
        if not self.tracked(isin):
            return None
        a30, tone30 = self._window(isin, cutoff, 31, 2)
        a7, _ = self._window(isin, cutoff, 8, 2)
        base, _ = self._window(isin, cutoff, 91, 32)
        ratio = (a30 / 30 + 0.2) / (base / 60 + 0.2)
        tot = self.total30(cutoff)
        bnd = boundary(cutoff)
        lo = bnd - timedelta(days=30)
        heads, seen_titles = [], set()
        for h in self.headlines.get(isin, []):
            t = seen_time(h['seendate'])
            if not (lo <= t < bnd):
                continue
            key = re.sub(r'\s+', ' ', normalize(h['title'])).strip()
            if key in seen_titles:
                continue
            seen_titles.add(key)
            heads.append(h)
            if len(heads) == n_headlines:
                break
        for h in heads:
            assert seen_time(h['seendate']) < bnd, 'headline after the cutoff boundary'
        return {'articles30': a30, 'articles7': a7, 'base60': base, 'ratio': ratio, 'tone30': tone30,
                'share30': (a30 / tot) if tot else 0.0, 'headlines': heads}


def card_lines(card, aliases, block_isins, headlines=True, unmask=False):
    """Prompt lines for one candidate's news; [] for assets the news layer does not track."""
    if card is None:
        return []
    if card['articles30'] == 0:
        return ['    news (Greek press, last 30 days before the cutoff): no articles']
    tone = card['tone30']
    tone_txt = 'unknown' if tone is None else f'{tone:+.1f} (below 0 negative, above 0 positive)'
    line = (f"    news (Greek press, last 30 days before the cutoff): {card['articles30']} articles, "
            f"x{card['ratio']:.1f} its usual level, {100 * card['share30']:.0f}% of the coverage of all "
            f"listed candidates; average tone {tone_txt}; "
            f"last 7 days: {card['articles7']} articles")
    out = [line]
    if headlines and card['headlines']:
        parts = []
        for h in card['headlines']:
            title = h['title'] if unmask else aliases.mask(h['title'], block_isins)
            title = title.replace('"', "'")[:160]
            parts.append(f"[{h['seendate'][:4]}-{h['seendate'][4:6]}-{h['seendate'][6:8]}] \"{title}\"")
        out.append('    latest headlines: ' + '; '.join(parts))
    return out
