#!/bin/bash
# Гибрид v2: смоук с prompts/encoder_v3.txt на тех же 3 покупателях мая (около 8 минут),
# затем тест HW2 на июне (100 клиентов, около 3-4 часов). Сбои модели не останавливают июнь:
# они обрабатываются запасными порядками и считаются в отчёте. Июнь не стартует, только если
# смоук упал с ошибкой программы. Двойной щелчок в Finder.
# Держи зарядку подключённой и крышку открытой; окно Terminal не закрывай.
# Журнал: week8/run_hybrid.log; последняя строка STATUS: ... — итог.
set -u
cd "$(dirname "$0")/far-trans-llm" || exit 1
LOG="../run_hybrid.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== smoke v3 start $(date '+%Y-%m-%d %H:%M:%S') ==="
if [ ! -x .venv/bin/python ]; then
  echo "Нет окружения far-trans-llm/.venv"; echo "STATUS: NO_VENV"; exit 2
fi
source .venv/bin/activate
if pgrep -f "explore_slots.py" >/dev/null; then
  echo "Останавливаю прогон explore_slots (кэш ответов сохраняется)..."
  pkill -INT -f "explore_slots.py"
  for _ in $(seq 1 30); do pgrep -f "explore_slots.py" >/dev/null || break; sleep 2; done
  pgrep -f "explore_slots.py" >/dev/null && pkill -TERM -f "explore_slots.py"
  sleep 3
fi
python -m unittest test_hybrid || { echo "STATUS: TESTS_FAILED"; exit 4; }
caffeinate -i python hybrid.py --config configs/hybrid_smoke_may.json \
  || { echo "STATUS: FAILED_SMOKE (июнь не запускаю)"; exit 5; }
echo "=== smoke v3 done $(date '+%Y-%m-%d %H:%M:%S'); STATUS: SMOKE_V3_DONE ==="
echo "=== test start $(date '+%Y-%m-%d %H:%M:%S') ==="
caffeinate -i python hybrid.py --config configs/hybrid_test_june.json \
  || { echo "STATUS: FAILED_TEST"; exit 7; }
echo "=== test finish $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "STATUS: TEST_DONE"
