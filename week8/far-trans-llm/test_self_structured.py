"""CPU-only tests with fake generators for self_structured_ranker; not model experiments."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import data_adapter
import self_structured_ranker as ssr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'far-trans'))
import test_starter  # noqa: E402  (provided synthetic FAR-Trans fixture)

CUTOFF = '2021-06-01'


def data_json(prompt):
    return json.loads(prompt.split('DATA:\n', 1)[1])


class FakeGenerator:
    """Stage 1: returns ``stage1`` (callable or text). Ranking: reversed candidate ids."""
    metadata = {'backend': 'fake_test_not_model'}

    def __init__(self, stage1=None, rank=None):
        self.stage1, self.rank, self.prompts = stage1, rank, []

    def generate(self, prompt, max_new_tokens):
        self.prompts.append(prompt)
        if prompt.startswith(ssr.STAGE1_INSTRUCTION):
            text = self.stage1(prompt) if callable(self.stage1) else (self.stage1 or '{"assets": []}')
        elif self.rank is not None:
            text = self.rank
        else:
            ids = [c['id'] for c in data_json(prompt)['candidates']]
            text = json.dumps(list(reversed(ids))[:10])
        return {'text': text, 'input_tokens': 1, 'output_tokens': 1, 'seconds': 0.0}


def perfect_stage1(prompt):
    """Answer stage 1 by applying the documented rules to the prompt's own data."""
    data = data_json(prompt)
    window = [(e['date'], e['id']) for e in data['buy_events_oldest_to_newest']]
    cards = {c['id']: c for c in data['history_assets']}
    ids = {i: i for i in cards}
    return json.dumps(ssr.code_profile(window, ids, cards, data['cutoff_date_exclusive']))


class RuleTests(unittest.TestCase):
    def test_buying_pattern(self):
        self.assertEqual(ssr.buying_pattern(['2021-01-01']), 'one_off')
        self.assertEqual(ssr.buying_pattern(['2021-01-01', '2021-01-01']), 'occasional')
        self.assertEqual(ssr.buying_pattern(['2021-01-01', '2021-02-01', '2021-03-03']), 'regular')
        self.assertEqual(ssr.buying_pattern(['2021-01-01', '2021-01-05', '2021-05-01']),
                         'occasional')

    def test_typical_interval_buckets(self):
        self.assertIsNone(ssr.typical_interval(['2021-01-01', '2021-01-01']))
        self.assertEqual(ssr.typical_interval(['2021-01-01', '2021-01-08']), 'weekly')
        self.assertEqual(ssr.typical_interval(['2021-01-01', '2021-01-31']), 'monthly')
        self.assertEqual(ssr.typical_interval(['2021-01-01', '2021-04-01']), 'quarterly')
        self.assertEqual(ssr.typical_interval(['2020-01-01', '2021-01-01']), 'longer')

    def test_recency_edges(self):
        self.assertEqual(ssr.recency_bucket('2021-05-02', CUTOFF), 'last_30_days')   # 30 days
        self.assertEqual(ssr.recency_bucket('2021-05-01', CUTOFF), '1_3_months')     # 31 days
        self.assertEqual(ssr.recency_bucket('2021-03-03', CUTOFF), '1_3_months')     # 90 days
        self.assertEqual(ssr.recency_bucket('2021-03-02', CUTOFF), '3_12_months')    # 91 days
        self.assertEqual(ssr.recency_bucket('2020-06-01', CUTOFF), '3_12_months')    # 365 days
        self.assertEqual(ssr.recency_bucket('2020-05-31', CUTOFF), 'over_1_year')

    def test_window_drops_sells_future_ineligible_and_keeps_latest(self):
        events = [('2021-05-01', 'A', 'Buy'), ('2021-05-02', 'B', 'Sell'),
                  ('2021-06-01', 'A', 'Buy'), ('2021-04-01', 'X', 'Buy'),
                  ('2021-03-01', 'A', 'Buy'), ('2021-04-15', 'C', 'Buy')]
        assets = {'A': {}, 'B': {}, 'C': {}}
        self.assertEqual(ssr.buy_window(events, CUTOFF, assets, 2),
                         [('2021-04-15', 'C'), ('2021-05-01', 'A')])

    def test_candidates_own_by_recency_then_never_bought_cbf(self):
        window = [('2021-01-01', 'A'), ('2021-03-01', 'B'), ('2021-02-01', 'A')]
        cands, n_own = ssr.build_candidates(window, {'A', 'B', 'Z'}, list('ZABCDEFGHIJKLM'),
                                            total=12, own_max=30)
        self.assertEqual(cands[:2], ['B', 'A'])
        self.assertEqual(n_own, 2)
        self.assertNotIn('Z', cands)          # bought before the window: neither own nor new
        self.assertEqual(len(cands), 12)
        cands, n_own = ssr.build_candidates(window, {'A', 'B'}, list('ABCDEFGHIJKLM'),
                                            total=10, own_max=1)
        self.assertEqual((cands[0], n_own, len(cands)), ('B', 1, 10))

    def test_ids_hide_candidate_order_and_mark_history_only(self):
        window = [('2021-01-01', 'H1'), ('2021-02-01', 'A')]
        ids, shown = ssr.assign_ids(list('ABCDEFGHIJ'), window, seed=0)
        self.assertEqual(sorted(ids[i] for i in 'ABCDEFGHIJ'),
                         [f'C{n:02d}' for n in range(1, 11)])
        self.assertEqual(ids['H1'], 'H01')
        self.assertNotEqual(shown, list('ABCDEFGHIJ'))
        self.assertEqual(ssr.assign_ids(list('ABCDEFGHIJ'), window, seed=0), (ids, shown))

    def test_code_profile(self):
        window = [('2021-01-04', 'A'), ('2021-02-03', 'A'), ('2021-03-05', 'A'),
                  ('2021-05-20', 'B')]
        ids = {'A': 'C01', 'B': 'C02'}
        cards = {'C01': {'category': 'Stock', 'sector': 'Banks', 'market': 'M1'},
                 'C02': {'category': 'unknown', 'sector': 'Energy', 'market': 'M1'}}
        prof = ssr.code_profile(window, ids, cards, CUTOFF)
        a = next(e for e in prof['assets'] if e['id'] == 'C01')
        self.assertEqual((a['times_bought'], a['buying_pattern'], a['typical_interval'],
                          a['recency']), (3, 'regular', 'monthly', '1_3_months'))
        self.assertEqual(prof['assets'][0]['id'], 'C02')   # most recent first
        self.assertEqual(prof['preferences']['main_categories'], ['Stock'])
        self.assertEqual(prof['preferences']['main_sectors'], ['Banks', 'Energy'])
        self.assertEqual(prof['overall_activity'],
                         {'total_buys': 4, 'distinct_assets': 2, 'bought_in_last_30_days': True})

    def test_permuted_window_keeps_dates_and_assets(self):
        window = [('2021-01-01', 'A'), ('2021-02-01', 'B'), ('2021-03-01', 'A')]
        perm = ssr.permuted_window(window, 0, 'U1')
        self.assertEqual(sorted(d for d, _ in perm), sorted(d for d, _ in window))
        self.assertEqual(sorted(i for _, i in perm), sorted(i for _, i in window))
        self.assertEqual(perm, ssr.permuted_window(window, 0, 'U1'))


class ParsingTests(unittest.TestCase):
    def test_parse_object_repairs_fence_and_prose(self):
        self.assertEqual(ssr.parse_object('{"a": 1}'), ({'a': 1}, False))
        self.assertEqual(ssr.parse_object('```json\n{"a": 1}\n```'), ({'a': 1}, True))
        self.assertEqual(ssr.parse_object('Here: {"a": 1} done'), ({'a': 1}, True))
        for bad in ('no json', '{"a": ', '[1, 2]', None):
            with self.assertRaises(ValueError):
                ssr.parse_object(bad)

    def test_clean_profile_keeps_only_valid_facts(self):
        value = {'assets': [
            {'id': 'C01', 'times_bought': 2, 'first_buy_date': '2021-01-01',
             'last_buy_date': '2021-07-01', 'buying_pattern': 'often',
             'typical_interval': None, 'recency': 'last_30_days'},
            {'id': 'C01', 'times_bought': 9},
            {'id': 'C99', 'times_bought': 1}, 'junk'],
            'preferences': {'main_categories': ['Stock', 3, 'Bond', 'Fund']},
            'overall_activity': {'total_buys': True}}
        prof, issues = ssr.clean_profile(value, {'C01', 'C02'}, CUTOFF)
        entry = prof['assets'][0]
        self.assertEqual(len(prof['assets']), 1)
        self.assertIsNone(entry['last_buy_date'])          # not before the cutoff
        self.assertIsNone(entry['buying_pattern'])         # outside the allowed values
        self.assertEqual(prof['preferences']['main_categories'], ['Stock', 'Bond'])
        self.assertIsNone(prof['overall_activity']['total_buys'])
        self.assertEqual(issues['duplicate_asset_id'], 1)
        self.assertEqual(issues['unknown_or_bad_asset_id'], 2)
        self.assertEqual(issues['omitted_history_assets'], 1)
        with self.assertRaises(ValueError):
            ssr.clean_profile({'preferences': {}}, {'C01'}, CUTOFF)

    def test_field_accuracy_perfect_and_partial(self):
        truth = {'assets': [{'id': 'C01', 'times_bought': 2, 'first_buy_date': '2021-01-01',
                             'last_buy_date': '2021-02-01', 'buying_pattern': 'occasional',
                             'typical_interval': 'monthly', 'recency': '3_12_months'}],
                 'preferences': {'main_categories': ['Stock'], 'main_sectors': [],
                                 'main_markets': ['M']},
                 'overall_activity': {'total_buys': 2, 'distinct_assets': 1,
                                      'bought_in_last_30_days': False}}
        acc = ssr.summarise_accuracy([ssr.field_accuracy(truth, truth)])
        self.assertEqual((acc['asset_recall'], acc['times_bought_exact'],
                          acc['jaccard_main_categories']), (1.0, 1.0, 1.0))
        empty = {'assets': [], 'preferences': {k: [] for k, _ in ssr.PREF_KEYS},
                 'overall_activity': {}}
        acc = ssr.summarise_accuracy([ssr.field_accuracy(empty, truth)])
        self.assertEqual((acc['asset_recall'], acc['times_bought_exact']), (0.0, None))


class EndToEndTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        test_starter.write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()

    def args(self, *extra):
        return ssr.build_parser().parse_args([
            '--data-dir', str(self.root), '--output-dir', str(self.root / 'out'),
            '--model', 'fake', '--candidate-count', '10', '--on-invalid', 'fallback',
            *extra])

    def test_prepare_only_writes_prompts_without_future_facts(self):
        args = self.args('--prepare-only', '--arms', 'R,Rp,S,C')
        summary = ssr.run(args, data_adapter.load(args))
        self.assertEqual(summary['stage'], 'prepare_only_no_inference')
        self.assertFalse((self.root / 'out' / 'metrics.json').exists())
        rows = [json.loads(line) for line in
                (self.root / 'out' / 'prompts.jsonl').read_text().splitlines()]
        self.assertEqual({r['arm'] for r in rows}, {'R', 'Rp', 'S_stage1', 'C'})
        text = ' '.join(r['prompt'] for r in rows)
        for future in ('2021-06-05', '2021-06-10', '2021-06-20', '2021-06-15', '2021-06-02'):
            self.assertNotIn(future, text)
        self.assertNotIn('"ISIN"', text)                 # short ids only
        self.assertNotIn('Professional', text)           # profile row after the cutoff

    def test_full_run_with_perfect_stage1_matches_code(self):
        args = self.args('--arms', 'R,Rp,S,C')
        gen = FakeGenerator(stage1=perfect_stage1)
        meta = ssr.run(args, data_adapter.load(args), generator=gen)
        out = self.root / 'out'
        metrics = json.loads((out / 'metrics.json').read_text())
        for name in ('llm_R_raw', 'llm_Rp_permuted_dates', 'llm_S_self_structured',
                     'llm_C_code_structured', 'cand_order_first10', 'freq_recency',
                     'popularity', 'cbf', 'latest_buy', 'latest_buy_or_sell'):
            self.assertEqual(metrics[name]['users'], 2, name)
        self.assertEqual(meta['invalid_by_arm'], {'R': 0, 'Rp': 0, 'S': 0, 'C': 0})
        acc = json.loads((out / 'stage1_accuracy.json').read_text())
        self.assertEqual(acc['asset_recall'], 1.0)
        self.assertEqual(acc['recency_exact'], 1.0)
        # Same profile and same candidates -> identical stage-2 prompts for S and C.
        stage2 = [p for p in gen.prompts if p.startswith(ssr.RANK_INSTRUCTION[:40])
                  and '"history"' in p]
        self.assertEqual(len(stage2), 4)
        self.assertEqual(stage2[0], stage2[1])
        self.assertEqual(stage2[2], stage2[3])

    def test_stage1_garbage_falls_back_and_is_counted(self):
        args = self.args('--arms', 'S')
        meta = ssr.run(args, data_adapter.load(args), generator=FakeGenerator(stage1='nope'))
        self.assertEqual(meta['invalid_by_arm'], {'S': 2})
        records = [json.loads(line) for line in
                   (self.root / 'out' / 'arm_responses.jsonl').read_text().splitlines()]
        self.assertTrue(all('stage1' in r['error'] and r['fallback'] for r in records))
        valid = json.loads((self.root / 'out' / 'metrics_valid_only.json').read_text())
        self.assertEqual(valid['llm_S_self_structured']['valid_users'], 0)

    def test_invalid_answer_text_is_kept_for_inspection(self):
        args = self.args('--arms', 'R')
        ssr.run(args, data_adapter.load(args), generator=FakeGenerator(rank='I like C01 most.'))
        records = [json.loads(line) for line in
                   (self.root / 'out' / 'arm_responses.jsonl').read_text().splitlines()]
        self.assertTrue(all(r['raw_response'] == 'I like C01 most.' and r['error'] for r in records))

    def test_invalid_ranking_raises_under_error_policy(self):
        args = ssr.build_parser().parse_args([
            '--data-dir', str(self.root), '--output-dir', str(self.root / 'out'),
            '--model', 'fake', '--candidate-count', '10', '--arms', 'C'])
        with self.assertRaises(ValueError):
            ssr.run(args, data_adapter.load(args), generator=FakeGenerator(rank='["C01"]'))

    def test_controls_only_needs_no_generator(self):
        args = self.args('--controls-only')
        meta = ssr.run(args, data_adapter.load(args))
        metrics = json.loads((self.root / 'out' / 'metrics.json').read_text())
        self.assertIn('freq_recency', metrics)
        self.assertFalse(any(name.startswith('llm_') for name in metrics))
        self.assertEqual(meta['arms'], [])

    def test_omitted_assets_become_not_bought_in_stage2(self):
        args = self.args('--arms', 'S')
        gen = FakeGenerator(stage1='{"assets": []}')
        ssr.run(args, data_adapter.load(args), generator=gen)
        stage2 = [data_json(p) for p in gen.prompts if not p.startswith(ssr.STAGE1_INSTRUCTION)]
        self.assertTrue(all(c['history'] == 'not_bought'
                            for d in stage2 for c in d['candidates']))
        acc = json.loads((self.root / 'out' / 'stage1_accuracy.json').read_text())
        self.assertEqual(acc['asset_recall'], 0.0)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        test_starter.write_fixture(self.root)

    def tearDown(self):
        self.tmp.cleanup()
        ssr.set_prompts({})

    def write(self, config):
        path = self.root / 'run.json'
        path.write_text(json.dumps(config), encoding='utf-8')
        return path

    def test_config_supplies_required_flags_and_cli_wins(self):
        path = self.write({'run_name': 'exp1', 'data_dir': str(self.root),
                           'output_dir': str(self.root / 'o'),
                           'settings': {'arms': 'R,C', 'candidate_count': 10, 'mask_names': True},
                           'generation': {'model': 'Qwen/Qwen3-0.6B', 'device': 'mps',
                                          'temperature': 0.7, 'thinking': True}})
        args = ssr.parse_args(['--config', str(path), '--arms', 'S'])
        self.assertEqual((args.run_name, args.arms, args.candidate_count, args.mask_names),
                         ('exp1', 'S', 10, True))
        self.assertEqual((args.model, args.device, args.temperature, args.thinking),
                         ('Qwen/Qwen3-0.6B', 'mps', 0.7, True))
        self.assertEqual(args.output_dir, self.root / 'o')

    def test_config_rejects_unknown_keys_and_bad_choices(self):
        with self.assertRaises(ValueError):
            ssr.parse_args(['--config', str(self.write({'settings': {'armz': 'R'}}))])
        with self.assertRaises(ValueError):
            ssr.parse_args(['--config', str(self.write({'generation': {'device': 'tpu'}}))])
        with self.assertRaises(ValueError):
            ssr.parse_args(['--config', str(self.write({'prompts': {'rank_instructions': 'x'}}))])

    def test_edited_prompts_reach_the_model_and_the_manifest(self):
        path = self.write({'data_dir': str(self.root), 'output_dir': str(self.root / 'o'),
                           'settings': {'arms': 'R,S', 'candidate_count': 10,
                                        'on_invalid': 'fallback'},
                           'generation': {'model': 'fake'},
                           'prompts': {'rank_instruction': 'RANK IN {horizon} DAYS {horizon}',
                                       'stage1_instruction': 'MAKE A PROFILE'}})
        args = ssr.parse_args(['--config', str(path)])
        gen = FakeGenerator()
        meta = ssr.run(args, data_adapter.load(args), generator=gen)
        self.assertTrue(any(p.startswith('RANK IN 30 DAYS 30\n') for p in gen.prompts))
        self.assertTrue(any(p.startswith('MAKE A PROFILE\n') for p in gen.prompts))
        self.assertEqual(meta['prompts_edited'], ['rank_instruction', 'stage1_instruction'])
        manifest = json.loads((self.root / 'o' / 'run_manifest.json').read_text())
        self.assertEqual(manifest['runtime']['prompts']['stage1_instruction'], 'MAKE A PROFILE')

    def test_without_config_prompts_are_reset_to_defaults(self):
        ssr.set_prompts({'raw_note': 'changed'})
        ssr.parse_args(['--data-dir', 'x', '--output-dir', 'y', '--model', 'm'])
        self.assertEqual(ssr.PROMPTS, ssr.DEFAULT_PROMPTS)


class FakeTensor:
    def __init__(self, n):
        self.shape = (1, n)


class FakeEncoding(dict):
    """Like transformers' BatchEncoding: a dict that can be moved to a device."""

    def to(self, device):
        return self


class FakeTorchModule:
    """Just enough of torch to exercise ModelGenerator's plumbing (no model runs)."""

    def __init__(self, cuda=False, mps=True):
        self.calls = []
        self.cuda = SimpleNamespace(is_available=lambda: cuda, synchronize=lambda: None,
                                    is_bf16_supported=lambda: True)
        self.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: mps))
        self.mps = SimpleNamespace(synchronize=lambda: None)
        self.bfloat16, self.float16, self.float32 = 'bf16', 'fp16', 'fp32'
        self.__version__ = 'fake'

    def set_num_threads(self, n):
        pass

    def manual_seed(self, seed):
        self.calls.append(('seed', seed))

    def inference_mode(self):
        import contextlib
        return contextlib.nullcontext()


def fake_modules(torch, answer='<think>plan [1]</think>["C01"]'):
    class Tokenizer:
        eos_token_id = 0

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            torch.calls.append(('tokenizer', model, kwargs))
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            torch.calls.append(('template', kwargs))
            return messages[0]['content']

        def __call__(self, text, return_tensors):
            return FakeEncoding(input_ids=FakeTensor(len(text.split())))

        def decode(self, tokens, skip_special_tokens):
            return answer

    class Outputs:
        def __getitem__(self, key):
            return [1, 2, 3]

    class Model:
        config = SimpleNamespace(_commit_hash='abc')

        @classmethod
        def from_pretrained(cls, model, **kwargs):
            torch.calls.append(('model', model, kwargs))
            return cls()

        def to(self, device):
            torch.calls.append(('to', device))
            return self

        def eval(self):
            return self

        def generate(self, **kwargs):
            torch.calls.append(('generate', {k: v for k, v in kwargs.items()
                                             if k != 'input_ids'}))
            return Outputs()

    transformers = SimpleNamespace(AutoTokenizer=Tokenizer, AutoModelForCausalLM=Model,
                                   __version__='fake')
    return {'torch': torch, 'transformers': transformers}


class ModelGeneratorTests(unittest.TestCase):
    def make(self, torch, **kwargs):
        with patch.dict(sys.modules, fake_modules(torch)):
            return ssr.ModelGenerator('Qwen/Qwen3-1.7B', **kwargs)

    def test_auto_picks_mac_gpu_with_float16_and_greedy(self):
        torch = FakeTorchModule(cuda=False, mps=True)
        gen = self.make(torch)
        self.assertEqual((gen.device, gen.metadata['dtype'], gen.metadata['do_sample']),
                         ('mps', 'fp16', False))
        self.assertIn(('to', 'mps'), torch.calls)
        self.assertTrue(torch.calls[0][2]['local_files_only'])
        out = gen.generate('a b c', max_new_tokens=5)
        settings = [c for c in torch.calls if c[0] == 'generate'][0][1]
        self.assertEqual((settings['do_sample'], settings['max_new_tokens']), (False, 5))
        self.assertNotIn('temperature', settings)
        self.assertEqual(out['input_tokens'], 3)

    def test_sampling_thinking_and_download_flags(self):
        torch = FakeTorchModule(cuda=True, mps=False)
        gen = self.make(torch, temperature=0.8, top_p=0.9, thinking=True, seed=7,
                        allow_download=True, revision='r1')
        out = gen.generate('x y', max_new_tokens=4)
        settings = [c for c in torch.calls if c[0] == 'generate'][0][1]
        self.assertEqual((settings['do_sample'], settings['temperature'], settings['top_p']),
                         (True, 0.8, 0.9))
        self.assertIn(('seed', 7), torch.calls)
        self.assertEqual(out['text'], '["C01"]')
        self.assertIn('<think>', out['raw_text'])
        template = [c for c in torch.calls if c[0] == 'template'][0][1]
        self.assertTrue(template['enable_thinking'])
        loaded = [c for c in torch.calls if c[0] == 'model'][0][2]
        self.assertEqual((loaded['local_files_only'], loaded['revision'],
                          loaded['torch_dtype']), (False, 'r1', 'bf16'))

    def test_refusals(self):
        with self.assertRaises(ValueError):
            self.make(FakeTorchModule(cuda=False, mps=False), device='mps')
        with self.assertRaises(ValueError):
            self.make(FakeTorchModule(cuda=False, mps=False), device='cuda')
        with self.assertRaises(ValueError):
            self.make(FakeTorchModule(), temperature=-1)
        gen = self.make(FakeTorchModule(), context_limit=4)
        with self.assertRaises(ValueError):
            gen.generate('a b c', max_new_tokens=2)


if __name__ == '__main__':
    unittest.main()
