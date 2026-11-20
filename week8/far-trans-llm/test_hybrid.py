"""Checks for hybrid.py without a real model: parsers, status rule, leakage and a fake-model run."""
from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

import hybrid

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('FAR_TRANS_DATA', HERE.parent / 'Data FAR-Trans'))


class Parsers(unittest.TestCase):
    rows = ['A11', 'A42']

    def enc(self, **change):
        obj = {'observed_patterns': [{'pattern': 'buys A11 often', 'supporting_assets': ['A11']}],
               'asset_status': {'A11': {'status': 'core', 'note': 'x'},
                                'A42': {'status': 'one_off', 'note': 'y'}},
               'unknowns': ['goals']}
        obj.update(change)
        return obj

    def test_encoder_valid_with_text_around(self):
        obj, err = hybrid.parse_encoder('Here:\n```json\n' + json.dumps(self.enc()) + '\n```', self.rows, 8)
        self.assertIsNone(err)
        self.assertEqual(obj['asset_status']['A11']['status'], 'core')

    def test_encoder_rejects(self):
        cases = {
            'missing row': self.enc(asset_status={'A11': {'status': 'core'}}),
            'bad status': self.enc(asset_status={'A11': {'status': 'core'}, 'A42': {'status': 'hot'}}),
            'foreign id': self.enc(observed_patterns=[{'pattern': 'p', 'supporting_assets': ['A99']}]),
            'no support': self.enc(observed_patterns=[{'pattern': 'p', 'supporting_assets': []}]),
            'too many': self.enc(observed_patterns=[{'pattern': 'p', 'supporting_assets': ['A11']}] * 9),
        }
        for name, obj in cases.items():
            with self.subTest(name):
                self.assertIsNone(hybrid.parse_encoder(json.dumps(obj), self.rows, 8)[0])
        self.assertIsNone(hybrid.parse_encoder('no json here', self.rows, 8)[0])

    def test_ranking(self):
        allowed = {f'A{n}' for n in range(10, 25)}
        good = [f'A{n}' for n in range(10, 20)]
        self.assertEqual(hybrid.parse_ranking('```' + json.dumps(good) + '```', allowed)[0], good)
        self.assertIsNone(hybrid.parse_ranking(json.dumps(good[:9]), allowed)[0])
        self.assertIsNone(hybrid.parse_ranking(json.dumps(good[:9] + ['A10']), allowed)[0])
        self.assertIsNone(hybrid.parse_ranking(json.dumps(good[:9] + ['A99']), allowed)[0])

    def test_fill_keeps_json_braces(self):
        self.assertEqual(hybrid.fill('{"a": 1} {x}', x=5), '{"a": 1} 5')

    def test_kendall(self):
        self.assertEqual(hybrid.kendall_tau(['a', 'b', 'c'], ['a', 'b', 'c']), 1.0)
        self.assertEqual(hybrid.kendall_tau(['a', 'b', 'c'], ['c', 'b', 'a']), -1.0)


class StatusRule(unittest.TestCase):
    def f(self, **kw):
        base = {'position': 'holds', 'last_sale_vs_avg_buy': None, 'buys': 1,
                'last_buy_days_ago': 200, 'first_buy_days_ago': 200}
        base.update(kw)
        return base

    def test_order_of_definitions(self):
        self.assertEqual(hybrid.rule_status(self.f(position='sold all', last_sale_vs_avg_buy=0.2, buys=5,
                                                   last_buy_days_ago=10)), 'core')
        self.assertEqual(hybrid.rule_status(self.f(position='sold all', last_sale_vs_avg_buy=0.2, buys=5,
                                                   last_buy_days_ago=120)), 'exited_gain')
        self.assertEqual(hybrid.rule_status(self.f(position='sold all', last_sale_vs_avg_buy=-0.1)), 'exited_loss')
        self.assertEqual(hybrid.rule_status(self.f(buys=3, last_buy_days_ago=20, first_buy_days_ago=40)), 'core')
        self.assertEqual(hybrid.rule_status(self.f(last_buy_days_ago=30, first_buy_days_ago=30)), 'new')
        self.assertEqual(hybrid.rule_status(self.f(buys=4, last_buy_days_ago=120)), 'lapsed')
        self.assertEqual(hybrid.rule_status(self.f()), 'one_off')


@unittest.skipUnless((DATA / 'transactions.csv').exists(), 'FAR-Trans data not found')
class FakeRun(unittest.TestCase):
    """Full pipeline on a validation cutoff with a fake model: outputs, ids, no leakage."""

    def test_pipeline(self):
        import data_adapter
        with tempfile.TemporaryDirectory() as tmp:
            args = hybrid.parse_args(['--data-dir', str(DATA), '--output-dir', tmp,
                                      '--cutoff', '2021-05-01', '--max-users', '6',
                                      '--permutations', '1,2'])
            bundle = data_adapter.load(args)
            assets = bundle[1]
            fake = hybrid.FakeGenerator()
            meta = hybrid.run(args, bundle, fake)
            metrics = json.loads((Path(tmp) / 'metrics.json').read_text())
            for name in list(hybrid.CELLS.values()) + ['latest_buy', 'popularity', 'cbf',
                                                       'latest_buy_or_sell']:
                self.assertIn(name, metrics)
                self.assertEqual(metrics[name]['users'], 6)
            self.assertEqual(meta['failures_by_stage'], {})
            self.assertTrue((Path(tmp) / 'protocol.json').exists())
            details = json.loads((Path(tmp) / 'details.json').read_text())
            prompts = '\n'.join(fake.prompts)
            self.assertFalse(hybrid.ISIN_RE.search(prompts), 'an ISIN reached the model')
            for info in details.values():
                isins = [x['isin'] for v in info['pool'].values() for x in v]
                self.assertGreaterEqual(len(isins), 10)
                self.assertEqual(len(isins), len(set(isins)))
                for isin in isins:
                    name = (assets[isin].get('assetName') or '').strip()
                    if len(name) > 6:
                        self.assertNotIn(name, prompts, 'an asset name reached the model')
            # every user gets 1 encoder + 2 cells x 2 permutations reranker calls
            self.assertEqual(len(fake.prompts), 6 * 5)

    def test_no_future_rows(self):
        market = hybrid.Market(DATA, '2021-05-01', [])
        market.assert_no_future()
        self.assertLess(market.max_input_day.isoformat(), '2021-05-01')


if __name__ == '__main__':
    unittest.main()
