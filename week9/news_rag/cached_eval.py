"""Re-evaluate a run from the generation cache only (no model): used when a long LLM run is stopped early.

The backend answers only prompts that are already in outputs/<cohort>/llm_cache; its metadata is copied
from the cache log, so cache keys match the real model's. A prompt that is not cached raises, and
block_ranker counts that client as a fallback; --max-users is chosen so that no such client is included.
Users are the cohort's own order, so "first N users" is a rule on past data only.
"""
import argparse
import json
from pathlib import Path

import block_ranker as br
import common


class CacheOnly:
    """Same cache key as runtime.LoggedGenerator, but read-only (never writes the shared log)."""

    def __init__(self, metadata, cache_dir):
        self.metadata, self.cache = metadata, Path(cache_dir)
        self.hits = self.misses = 0

    def generate(self, prompt, max_new_tokens=90):
        import hashlib
        d = {'prompt': prompt, 'settings': self.metadata, 'max_new_tokens': max_new_tokens}
        key = hashlib.sha256(json.dumps(d, sort_keys=True).encode()).hexdigest()
        f = self.cache / (key + '.json')
        if not f.exists():
            self.misses += 1
            raise RuntimeError('not in cache')
        self.hits += 1
        return json.loads(f.read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', required=True)
    p.add_argument('--max-users', type=int, required=True)
    p.add_argument('--run-name', required=True)
    a = p.parse_args()
    cfg = json.loads(Path(a.config).read_text())
    cfg['run_name'] = a.run_name
    log = common.HERE / 'outputs' / cfg['cohort'] / 'llm_cache' / 'generations.jsonl'
    meta = None
    with log.open() as fh:
        for line in fh:
            meta = json.loads(line)['settings']
            break
    gen = CacheOnly(meta, common.HERE / 'outputs' / cfg['cohort'] / 'llm_cache' / 'generation_cache')
    br.run(cfg, gen, a.max_users)
    print('cache hits', gen.hits, 'misses', gen.misses)


if __name__ == '__main__':
    main()
