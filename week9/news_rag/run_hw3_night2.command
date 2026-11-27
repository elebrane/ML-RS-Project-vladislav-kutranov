#!/bin/bash
# HW3, ночь 2 (только если ночь 1 показала, что новости меняют результат): A2 только числа, A3 плацебо,
# проверки порядка подачи и названий на 200 клиентах. M и A1 берутся из кэша. Около 5 часов.
set -u
cd "$(dirname "$0")" || exit 1
exec > >(tee -a "run_hw3_night2.log") 2>&1
PY="$PWD/../../week8/far-trans-llm/.venv/bin/python"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
[ -f outputs/D1_may/night1_main_and_A1/hw3_summary.json ] || { echo "Сначала ночь 1"; echo "STATUS: NO_NIGHT1"; exit 3; }
caffeinate -i "$PY" block_ranker.py --config configs/night2_D1_may.json || { echo "STATUS: FAILED"; exit 5; }
"$PY" summarize.py > /dev/null && echo "сводка: outputs/summary.md"
echo "STATUS: NIGHT2_DONE"
