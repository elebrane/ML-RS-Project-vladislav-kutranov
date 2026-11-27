"""Unit tests (no model, no network): time boundary, masking, parsing, arm isolation, fake run."""
import json
import random
import re
import unittest
from collections import Counter
from datetime import date, timedelta

import block_ranker as br
import common
import news

ISIN_RE = re.compile(r'\b[A-Z]{2}[A-Z0-9]{9}[0-9]\b')


def synthetic_field(cutoff, isins):
    """Every tracked stock gets articles on days C-40..C (C-1 and C must be ignored) and headlines
    around the boundary (the one at 21:30 UTC on C-1 must be ignored)."""
    aliases = news.Aliases.load()
    c = date.fromisoformat(cutoff)
    daily, heads = {}, {}
    for n, i in enumerate(isins):
        if i not in aliases.rows:
            continue
        daily[i] = {(c - timedelta(days=k)).isoformat(): (1 + n % 3, -1.0 + n % 3) for k in range(0, 100)}
        name = aliases.rows[i]['title_patterns'][0].rstrip('*')
        d1 = c - timedelta(days=1)
        heads[i] = [
            {'isin': i, 'seendate': d1.strftime('%Y%m%d') + 'T213000Z', 'title': f'{name} LATE headline', 'title_match': True},
            {'isin': i, 'seendate': d1.strftime('%Y%m%d') + 'T205900Z', 'title': f'{name} raises capital', 'title_match': True},
            {'isin': i, 'seendate': (c - timedelta(days=5)).strftime('%Y%m%d') + 'T100000Z', 'title': f'{name} results', 'title_match': True},
        ]
    return news.InfoField(daily, heads, aliases)


class TestText(unittest.TestCase):
    def test_greek_matching_and_masking(self):
        a = news.Aliases.load()
        t = 'Χρ . Σταϊκούρας : Με 10 δισ . ευρώ ... η τράπεζα Πειραιώς'
        self.assertTrue(a.matches('GRS014003032', t))
        self.assertEqual(a.mask(t), 'Χρ . Σταϊκούρας : Με 10 δισ . ευρώ ... η [company]')
        self.assertTrue(a.matches('GRS393503008', 'ΜΥΤΙΛΗΝΑΙΟΥ: κέρδη'))
        self.assertFalse(a.matches('GRS260333000', 'Ποτέ ξανά'))
        self.assertEqual(a.mask('ΓΕΚ ΤΕΡΝΑ - ΤΕΡΝΑ Ενεργειακή'), '[company] - [company]')

    def test_parse_choice(self):
        allowed = {'A11', 'A12', 'A13', 'A14'}
        self.assertEqual(br.parse_choice('["A12","A11"]', allowed, 2), (['A12', 'A11'], 'ok'))
        self.assertEqual(br.parse_choice('["A12","A12","A99","A13"]', allowed, 2), (['A12', 'A13'], 'repaired'))
        self.assertEqual(br.parse_choice('I pick A14 then A11', allowed, 2), (['A14', 'A11'], 'repaired'))
        self.assertEqual(br.parse_choice('["A99"]', allowed, 1), (None, 'fallback'))
        self.assertEqual(br.parse_choice('', allowed, 1), (None, 'fallback'))


class TestCohort(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cohorts = json.loads((common.HERE / 'configs' / 'cohorts.json').read_text())
        cls.cohort = cohorts['D0_june']
        cls.c = common.load_cohort(cls.cohort, common.HERE / 'outputs' / '_test')
        cls.blocks = common.blocks_for(cls.c)
        block_isins = {i for b in cls.blocks.values() for i in b['block']}
        cls.field = synthetic_field(cls.cohort['cutoff'], block_isins)

    def test_time_boundary(self):
        cutoff = self.cohort['cutoff']
        i = next(iter(self.field.daily))
        card = self.field.card(i, cutoff)
        per_day = self.field.daily[i][(date.fromisoformat(cutoff) - timedelta(days=2)).isoformat()][0]
        self.assertEqual(card['articles30'], 30 * per_day)          # C-31..C-2 only
        self.assertEqual(card['articles7'], 7 * per_day)
        titles = [h['title'] for h in card['headlines']]
        self.assertTrue(all('LATE' not in t for t in titles))
        self.assertEqual(len(titles), 2)
        for h in card['headlines']:
            self.assertLess(news.seen_time(h['seendate']), news.boundary(cutoff))

    def _prompts(self, u):
        b, f = self.blocks[u], self.c['facts'][u]
        starter, assets, *_ = self.c['bundle']
        cutoff = self.cohort['cutoff']
        cat = common.hybrid.category_of(assets)
        states = {i: self.c['market'].state(i, self.c['market'].cutoff, cat, list(assets)) for i in b['block']}
        ids = common.hybrid.assign_ids(u, cutoff, b['block'])
        shown = list(b['block'])
        random.Random(1).shuffle(shown)
        cards = {i: self.field.card(i, cutoff) for i in b['block'] if self.field.tracked(i)}
        tpl = (common.HERE / 'prompts' / 'block_ranker_v1.txt').read_text()
        out = {}
        for arm, acfg in list(br.LLM_ARMS.items()) + list(br.SUBSET_ARMS.items()):
            out[arm] = br.build_prompt(tpl, acfg, u, b, f, ids, shown, assets, states, cards, self.field, cutoff)[0]
        return out, cards

    def test_arms_differ_only_in_news_lines(self):
        users = [u for u, b in self.blocks.items() if b['slots'] > 0 and b['main_category'] == 'Stock'][:10]
        self.assertTrue(users)
        for u in users:
            P, cards = self._prompts(u)
            lines = {a: P[a].split('\n') for a in P}
            is_news = lambda s: s.startswith('    news') or s.startswith('    latest headlines')
            self.assertEqual([s for s in lines['llm_full'] if not is_news(s)], lines['llm_nonews'])
            self.assertEqual([s for s in lines['llm_full'] if not s.startswith('    latest headlines')], lines['llm_numbers'])
            self.assertEqual(Counter(s for s in lines['llm_full'] if is_news(s)),
                             Counter(s for s in lines['llm_shuffled'] if is_news(s)))
            self.assertEqual(len(lines['llm_full']), len(lines['llm_shuffled']))
            if sum(c is not None for c in cards.values()) >= 2:
                self.assertNotEqual(P['llm_full'], P['llm_shuffled'])
            for arm in ('llm_nonews', 'llm_numbers', 'llm_full', 'llm_shuffled', 'llm_full_order2'):
                self.assertIsNone(ISIN_RE.search(P[arm]), arm)
                self.assertNotIn(' | name ', P[arm])
            self.assertIn(' | name ', P['llm_full_names'])

    def test_headlines_masked(self):
        u = next(u for u, b in self.blocks.items() if b['slots'] > 0 and b['main_category'] == 'Stock')
        P, cards = self._prompts(u)
        heads = [s for s in P['llm_full'].split('\n') if s.startswith('    latest headlines')]
        self.assertTrue(heads)
        for s in heads:
            for i in self.blocks[u]['block']:
                self.assertFalse(self.field.aliases.matches(i, s), s)
            self.assertIn('[company]', s)

    def test_fake_run_end_to_end(self):
        cfg = {'cohort': 'D0_june', 'run_name': '_unit_test_fake_model', 'arms': list(br.CODE_ARMS) + list(br.LLM_ARMS),
               'subset_arms': list(br.SUBSET_ARMS), 'subset_users': 20, '_field': self.field}
        s = br.run(cfg, br.FakeGenerator())
        self.assertEqual(s['meta']['users'], 100)
        self.assertEqual(s['table']['c0_popularity']['all'], 0.0262)
        st = s['meta']['llm_status']['llm_full']
        self.assertEqual(st.get('fallback', 0), 0)
        self.assertEqual(sum(st.values()), 100)


if __name__ == '__main__':
    unittest.main()
