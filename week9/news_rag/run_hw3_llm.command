#!/bin/bash
# HW3, ночь 1 (обязательный минимум): тесты -> C-news на апреле (без модели) -> smoke 60 клиентов ->
# май: M и A1 -> июнь: M и A1 -> сводка outputs/summary.md и решение о ночи 2. Модель — только для «акционных» клиентов.
# Около 2.5–3 часов (A1 уже в кэше). Запускать ПОСЛЕ STATUS: NEWS_DONE и STATUS: A1_DONE — две копии модели не влезут в память. Прерывание безопасно (кэш ответов).
set -u
cd "$(dirname "$0")" || exit 1
LOG="run_hw3_llm.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== hw3 night1 start $(date '+%Y-%m-%d %H:%M:%S') ==="
PY="$PWD/../../week8/far-trans-llm/.venv/bin/python"
[ -x "$PY" ] || { echo "STATUS: NO_VENV"; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -c "import torch, transformers" || { echo "STATUS: VENV_BROKEN"; exit 2; }
[ -f news/daily_infofield.csv ] || "$PY" build_infofield.py || { echo "Сначала run_news_collect.command"; echo "STATUS: NO_NEWS"; exit 3; }
"$PY" -m unittest test_news_rag || { echo "STATUS: TESTS_FAILED"; exit 4; }
step () {
  if [ -f "outputs/$2/$3/hw3_summary.json" ]; then echo "--- $1: уже готов"; return 0; fi
  echo "--- $1: start $(date '+%H:%M:%S')"
  caffeinate -i "$PY" block_ranker.py --config "configs/$1" "${@:4}" || { echo "STATUS: FAILED_$1"; exit 5; }
  echo "--- $1: done $(date '+%H:%M:%S')"
}
step gonogo_TUNE_apr.json TUNE_apr gonogo_code_news
step smoke_TUNE_apr.json TUNE_apr smoke_llm --max-users 60
"$PY" - <<'PYEOF' || { echo "STATUS: SMOKE_FAILED"; exit 7; }
import json, sys
m = json.load(open('outputs/TUNE_apr/smoke_llm/hw3_summary.json'))['meta']
s = m['llm_status']
bad = {a: v.get('fallback', 0) + v.get('repaired', 0) for a, v in s.items()}
calls = {a: v.get('ok', 0) + bad[a] for a, v in s.items()}
print('smoke: invalid (repaired+fallback)', bad, 'of calls', calls, '| generation seconds', m['seconds']['generation_by_arm_incl_cached'])
sys.exit(1 if any(calls[a] and bad[a] > 0.25 * calls[a] for a in s) else 0)
PYEOF
step night1_D1_may.json D1_may night1_main_and_A1
step night1_D0_june.json D0_june night1_main_and_A1
"$PY" summarize.py > /dev/null && echo "сводка: outputs/summary.md"
"$PY" - <<'PYEOF'
import json
s = json.load(open('outputs/D1_may/night1_main_and_A1/hw3_summary.json'))
c = next(x for x in s['contrasts'] if x['a'] == 'llm_full' and x['b'] == 'llm_nonews')['all_users']
print(f"M - A1 on May: {c['diff']:+.4f} [{c['ci95'][0]:+.4f}, {c['ci95'][1]:+.4f}], better/worse {c['better']}/{c['worse']}")
print('A2/A3 (run_hw3_night2.command) — только если останется время до дедлайна; в отчёте указать как не выполненные')
PYEOF
echo "=== hw3 night1 finish $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "STATUS: NIGHT1_DONE"
