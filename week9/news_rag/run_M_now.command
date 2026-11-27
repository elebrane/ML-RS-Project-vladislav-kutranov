#!/bin/bash
# Срочный прогон к дедлайну: инфополе из уже собранных новостей -> тесты -> C-news (без модели) -> M на мае (A1 из кэша) -> июнь -> сводка.
set -u
cd "$(dirname "$0")" || exit 1
exec > >(tee -a "run_M_now.log") 2>&1
echo "=== M start $(date '+%Y-%m-%d %H:%M:%S') ==="
PY="$PWD/../../week8/far-trans-llm/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"$PY" build_infofield.py > news/build.log 2>&1; tail -3 news/build.log
"$PY" -m unittest test_news_rag 2>&1 | tail -2
caffeinate -i "$PY" block_ranker.py --config configs/gonogo_TUNE_apr.json | tail -20
caffeinate -i "$PY" block_ranker.py --config configs/night1_D1_may.json || { echo "STATUS: FAILED_M"; exit 5; }
"$PY" summarize.py > /dev/null
echo "STATUS: M_DONE $(date '+%H:%M:%S')"
caffeinate -i "$PY" block_ranker.py --config configs/night1_D0_june.json && "$PY" summarize.py > /dev/null
echo "STATUS: ALL_DONE $(date '+%H:%M:%S')"
