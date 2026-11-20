#!/bin/bash
# Гибрид v2: тест HW2 на дефолтах стартера (100 клиентов, срез 2021-06-01, 30 дней).
# Конфигурация заморожена: configs/hybrid_test_june.json, prompts/*_v1.txt. Один запуск, около 4-8 часов.
# Двойной щелчок в Finder. Журнал: week8/run_hybrid.log; последняя строка STATUS: ... — итог.
set -u
cd "$(dirname "$0")/far-trans-llm" || exit 1
LOG="../run_hybrid.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== test start $(date '+%Y-%m-%d %H:%M:%S') ==="
if [ ! -x .venv/bin/python ]; then
  echo "Нет окружения far-trans-llm/.venv"; echo "STATUS: NO_VENV"; exit 2
fi
source .venv/bin/activate
if pgrep -f "explore_slots.py" >/dev/null; then
  echo "Идёт прогон explore_slots — останавливаю (кэш ответов сохраняется)..."
  pkill -INT -f "explore_slots.py"
  for _ in $(seq 1 30); do pgrep -f "explore_slots.py" >/dev/null || break; sleep 2; done
  pgrep -f "explore_slots.py" >/dev/null && pkill -TERM -f "explore_slots.py"
  sleep 3
fi
python -m unittest test_hybrid || { echo "STATUS: TESTS_FAILED"; exit 4; }
caffeinate -i python hybrid.py --config configs/hybrid_test_june.json \
  || { echo "STATUS: FAILED_TEST"; exit 5; }
echo "=== test finish $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "STATUS: TEST_DONE"
