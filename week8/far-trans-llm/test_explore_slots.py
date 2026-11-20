"""Tests for explore_slots.py (no model, no real data needed)."""
import importlib.util
import json
import random
import tempfile
import unittest
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import explore_slots as es

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('starter_for_tests', HERE.parent / 'far-trans' / 'starter.py')
starter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(starter)


class FakeGenerator:
    """Answers with the shortlist ids in reverse id order (deterministic, model-free)."""
    metadata = {'backend': 'fake'}

    def __init__(self, broken_every=0):
        self.calls, self.prompts, self.broken_every = 0, [], broken_every

    def generate(self, prompt, max_new_tokens=96):
        self.calls += 1
        self.prompts.append(prompt)
        data = json.loads(prompt.split('DATA:\n', 1)[1])
        k = int(prompt.split('Choose the ', 1)[1].split(' ', 1)[0])
        ids = [c['id'] for c in data['SHORTLIST']][::-1][:k]
        if self.broken_every and self.calls % self.broken_every == 0:
            return {'text': 'I think N01 is good', 'seconds': 0.01}
        return {'text': json.dumps(ids), 'seconds': 0.01}


def synthetic_bundle(cutoff='2021-05-01', horizon=30, n_users=40, seed=3):
    rng = random.Random(seed)
    cats = ['Stock', 'Bond', 'MTF']
    assets = {f'A{n:02d}': {'ISIN': f'A{n:02d}', 'assetName': f'Company {n}', 'assetShortName': f'C{n}',
                            'assetCategory': cats[n % 3], 'assetSubCategory': 'Sub' + str(n % 4),
                            'marketID': 'XATH', 'sector': 'S' + str(n % 5), 'industry': 'I' + str(n % 6),
                            'timestamp': '2018-01-01'} for n in range(30)}
    events = defaultdict(list)
    truth = defaultdict(set)
    days = [f'{y}-{m:02d}-{d:02d}' for y in (2018, 2019, 2020, 2021) for m in range(1, 13) for d in (3, 17)]
    end = es.add_days(cutoff, horizon)
    for u in range(n_users):
        uid = f'U{u:03d}'
        favourite = rng.sample(sorted(assets), 3)
        for day in days:
            if day >= end or rng.random() > 0.35:
                continue
            isin = rng.choice(favourite) if rng.random() < 0.7 else rng.choice(sorted(assets))
            kind = 'Buy' if rng.random() < 0.8 else 'Sell'
            if day < cutoff:
                events[uid].append((day, isin, kind))
            elif kind == 'Buy':
                truth[uid].add(isin)
    history = defaultdict(Counter)
    for uid, evs in events.items():
        for _d, isin, kind in evs:
            if kind == 'Buy':
                history[uid][isin] += 1
    users = sorted(u for u in history if history[u])[:30]
    features = {i: starter.tokens(a) for i, a in assets.items()}
    pop = starter.popularity_scores(history, assets)
    profiles = {u: {'customerID': u, 'customerType': 'Mass', 'riskLevel': 'Balanced',
                    'investmentCapacity': 'CAP_LT30K', 'lastQuestionnaireDate': '2019-01-01',
                    'timestamp': '2019-01-01'} for u in users}
    contexts, baseline = starter.contexts_for(users, assets, profiles, history, features, pop, cutoff, 20, False)
    recent = starter.recency_rankings(contexts, pop, events, k=10)
    for uid in users:
        baseline[uid].update(recent[uid])
    manifest = {'cutoff': cutoff, 'end_exclusive': end, 'users': users}
    return (starter, assets, history, truth, contexts, baseline, events, pop, manifest)


def args_for(tmp, **extra):
    base = dict(cutoff='2021-05-01', horizon_days=30, arms=','.join(es.ARMS), history_events=50, own_max=30,
                propensity_start='2019-07-01', shortlist_size=12, popularity_days=30, holdings_shown=10,
                shuffle_seed=0, second_seed=1, max_new_tokens=96, prepare_only=False,
                on_invalid='fallback', run_name='test', output_dir=Path(tmp), candidate_count=20)
    base.update(extra)
    return SimpleNamespace(**base)


class WindowRowsTest(unittest.TestCase):
    def test_features_use_only_earlier_events_and_labels_follow(self):
        evs = [('2020-01-05', 'A', 'Buy'), ('2020-02-10', 'B', 'Buy'), ('2020-02-20', 'A', 'Sell'),
               ('2020-03-05', 'C', 'Buy'), ('2020-03-20', 'A', 'Buy')]
        rows = es.window_rows(evs, ['2020-01-01', '2020-03-01'], 30)
        self.assertEqual(len(rows), 1)                    # no Buy before 2020-01-01
        k, feats, any_buy, new_buy = rows[0]
        self.assertEqual(k, '2020-03-01')
        self.assertEqual(any_buy, 1)
        self.assertEqual(new_buy, 1)                      # C was never traded before 2020-03-01
        self.assertAlmostEqual(feats[0], es.math.log1p(20))   # last buy 2020-02-10
        self.assertAlmostEqual(feats[1], es.math.log1p(10))   # last txn 2020-02-20 (sell)
        self.assertAlmostEqual(feats[3], es.math.log1p(2))    # distinct traded: A, B
        self.assertAlmostEqual(feats[5], es.math.log1p(1))    # one sell in the last 30 days
        with_future = es.window_rows(sorted(evs + [('2020-02-29', 'Z', 'Buy')]), ['2020-03-01'], 30)
        self.assertNotEqual(with_future[0][1], feats)
        later_only = es.window_rows(sorted(evs + [('2020-03-01', 'Z', 'Buy')]), ['2020-03-01'], 30)
        self.assertEqual(later_only[0][1], feats)         # an event on the cutoff day is not a feature

    def test_repeat_purchase_is_not_new(self):
        rows = es.window_rows([('2020-01-05', 'A', 'Buy'), ('2020-03-10', 'A', 'Buy')], ['2020-03-01'], 30)
        self.assertEqual(rows[0][2:], (1, 0))


class PartsTest(unittest.TestCase):
    def test_own_order_count_then_recency(self):
        window = [('2020-01-01', 'A'), ('2020-02-01', 'B'), ('2020-03-01', 'A'), ('2020-04-01', 'C')]
        order, counts = es.own_order(window)
        self.assertEqual(order, ['A', 'C', 'B'])
        self.assertEqual(counts['A'], 2)

    def test_shortlist_excludes_traded_and_puts_main_category_first(self):
        assets = {'A': {'assetCategory': 'Stock'}, 'B': {'assetCategory': 'Bond'},
                  'C': {'assetCategory': 'Stock'}, 'D': {'assetCategory': 'Stock'}, 'E': {'assetCategory': 'Bond'}}
        b30 = {'B': 50, 'C': 3, 'D': 9, 'E': 1}
        b365 = {'A': 5, 'B': 60, 'C': 10, 'D': 20, 'E': 2}
        short = es.shortlist_for(['A', 'B', 'C', 'D', 'E', 'F'], {'A'}, assets, 'Stock', b30, b365, 4)
        self.assertEqual(short, ['D', 'C', 'B', 'E'])     # F never bought by anybody in 365 days

    def test_slots_and_compose(self):
        self.assertEqual(es.slots(2, 0.1), 8)
        self.assertEqual(es.slots(15, 0.34), 3)
        self.assertEqual(es.slots(15, 0.0), 0)
        own = [f'O{i}' for i in range(12)]
        ranking = es.compose(own, ['N1', 'N2', 'N3'], 3, ['X1', 'X2'])
        self.assertEqual(ranking, own[:7] + ['N1', 'N2', 'N3'])
        short = es.compose(['O1'], ['N1'], 1, ['O1', 'X1', 'X2', 'X3', 'X4', 'X5', 'X6', 'X7', 'X8', 'X9'])
        self.assertEqual(len(set(short)), 10)
        self.assertEqual(short[:2], ['O1', 'N1'])


class ParseTest(unittest.TestCase):
    def test_valid_repaired_and_partial_answers(self):
        allowed = {'N01', 'N02', 'N03', 'N04'}
        self.assertEqual(es.parse_ids('["N02", "N01"]', allowed, 2), (['N02', 'N01'], True, False))
        self.assertEqual(es.parse_ids('```json\n["N03","N04"]\n```', allowed, 2), (['N03', 'N04'], True, True))
        ids, ok, _ = es.parse_ids('["N03", "N99"]', allowed, 2)
        self.assertEqual((ids, ok), (['N03'], False))
        ids, ok, _ = es.parse_ids('Best: N04 then N02', allowed, 2)
        self.assertEqual((ids, ok), (['N04', 'N02'], True))
        self.assertEqual(es.parse_ids('["N01", "N01"]', allowed, 2)[1], False)
        self.assertEqual(es.parse_ids('nothing', allowed, 2), ([], False, True))


class RunTest(unittest.TestCase):
    def test_end_to_end_with_fake_model(self):
        bundle = synthetic_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            args = args_for(tmp)
            es.set_prompts({})
            gen = FakeGenerator(broken_every=7)
            meta = es.run(args, bundle, generator=gen)
            metrics = json.loads((Path(tmp) / 'metrics.json').read_text())
            for name in ('freq_recency_cbf', 'explore_control_popularity', 'explore_llm',
                         'explore_llm_masked_names', 'explore_llm_shuffled_shortlist', 'popularity'):
                self.assertIn(name, metrics)
            self.assertGreater(gen.calls, 0)
            self.assertGreater(meta['invalid_by_arm']['llm'] + meta['invalid_by_arm']['llm_masked_names']
                               + meta['invalid_by_arm']['llm_shuffled_shortlist'], 0)
            preds = defaultdict(lambda: defaultdict(list))
            for line in (Path(tmp) / 'predictions.csv').read_text().splitlines()[1:]:
                method, uid, _rank, isin = line.split(',')
                preds[uid][method].append(isin)
            truth = bundle[3]
            for prompt in gen.prompts:
                data = json.loads(prompt.split('DATA:\n', 1)[1])
                for h in data['HOLDINGS']:
                    self.assertLess(h['last_trade_date'], '2021-05-01')
            masked = [p for p in gen.prompts if 'Held asset 1' in p]
            self.assertTrue(masked)
            self.assertTrue(all('Company' not in p.split('DATA:\n', 1)[1] for p in masked))
            for uid, methods in preds.items():
                own = bundle[5][uid]  # baseline rankings exist for every user
                self.assertTrue(own)
                for method, ranking in methods.items():
                    self.assertEqual(len(set(ranking)), 10, (uid, method))
                # the repeat part is identical across the exploration arms
                k_users = json.loads((Path(tmp) / 'diagnostics.json').read_text())
                self.assertIn('k_distribution', k_users)
            self.assertNotIn('truth', ' '.join(gen.prompts).lower().replace('truthful', ''))
            self.assertTrue(truth)

    def test_prepare_only_writes_prompts_without_a_model(self):
        bundle = synthetic_bundle()
        with tempfile.TemporaryDirectory() as tmp:
            summary = es.run(args_for(tmp, prepare_only=True), bundle)
            self.assertEqual(summary['stage'], 'prepare_only_no_inference')
            self.assertTrue((Path(tmp) / 'prompts.jsonl').exists())
            self.assertGreater(summary['llm_calls'], 0)

    def test_config_parsing(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = Path(tmp) / 'c.json'
            cfg.write_text(json.dumps({
                'run_name': 'x', 'data_dir': 'data', 'output_dir': 'out',
                'protocol': {'cutoff': '2021-05-01', 'horizon_days': 30, 'max_users': 5},
                'pipeline': {'shortlist_size': 15},
                'arms': ['control_popularity', 'llm'],
                'llm_input': {'holdings_shown': 6, 'max_new_tokens': 80, 'on_invalid': 'fallback'},
                'generation': {'model': 'Qwen/Qwen3-4B-Instruct-2507', 'revision': 'main', 'device': 'mps',
                               'temperature': 0.0, 'context_limit': 4096, 'allow_download': True},
                'prompts': {}}))
            args = es.parse_args(['--config', str(cfg), '--max-users', '3'])
            self.assertEqual((args.max_users, args.shortlist_size, args.arms, args.device, args.model),
                             (3, 15, 'control_popularity,llm', 'mps', 'Qwen/Qwen3-4B-Instruct-2507'))
            cfg.write_text(json.dumps({'data_dir': 'd', 'output_dir': 'o', 'generation': {'model': 'm', 'bogus': 1}}))
            with self.assertRaises(ValueError):
                es.parse_args(['--config', str(cfg)])


if __name__ == '__main__':
    unittest.main()
