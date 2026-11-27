#!/bin/bash
# HW3, запускать СРАЗУ, параллельно со сбором новостей: ветке A1 (модель без новостей) новости не нужны.
# Тесты -> smoke A1 на 60 апрельских клиентах -> A1 на мае (669 «акционных» клиентов). Около 1–1.5 ч.
# Ответы кэшируются; основной прогон run_hw3_llm.command потом возьмёт их из кэша.
set -u
cd "$(dirname "$0")" || exit 1
exec > >(tee -a "run_hw3_A1_now.log") 2>&1
echo "=== A1 start $(date '+%Y-%m-%d %H:%M:%S') ==="
PY="$PWD/../../week8/far-trans-llm/.venv/bin/python"
[ -x "$PY" ] || { echo "STATUS: NO_VENV"; exit 2; }
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" -m unittest test_news_rag || { echo "STATUS: TESTS_FAILED"; exit 4; }
[ -f outputs/TUNE_apr/smoke_a1/hw3_summary.json ] || caffeinate -i "$PY" block_ranker.py --config configs/smoke_a1_TUNE_apr.json --max-users 60 || { echo "STATUS: SMOKE_FAILED"; exit 5; }
"$PY" -c "import json; m=json.load(open('outputs/TUNE_apr/smoke_a1/hw3_summary.json'))['meta']; print('smoke A1:', m['llm_status'], m['seconds']['generation_by_arm_incl_cached'])"
[ -f outputs/D1_may/a1_now/hw3_summary.json ] || caffeinate -i "$PY" block_ranker.py --config configs/a1_D1_may.json || { echo "STATUS: FAILED_A1"; exit 5; }
echo "=== A1 finish $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "STATUS: A1_DONE"
