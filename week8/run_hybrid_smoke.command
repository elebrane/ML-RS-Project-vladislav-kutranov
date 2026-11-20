#!/bin/bash
# Гибрид v2 (encoder + reranker, абляция 2x2): смоук-тест на 3 покупателях проверочного среза
# 2021-05-01 (окно до 31 мая, июнь не трогаем). 15 запросов к Qwen3-4B, около 15-25 минут.
# Двойной щелчок в Finder. Сначала останавливает прогон run_explore (его ответы в кэше,
# повторный запуск run_explore.command продолжит с того же места), потом тесты, потом смоук.
# Журнал: week8/run_hybrid.log; последняя строка STATUS: ... — итог.
set -u
cd "$(dirname "$0")/far-trans-llm" || exit 1
LOG="../run_hybrid.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== smoke start $(date '+%Y-%m-%d %H:%M:%S') ==="
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
  echo "explore_slots остановлен: $(date '+%H:%M:%S')"
fi
python -c "import torch, transformers; print('torch', torch.__version__, '| transformers', transformers.__version__, '| mps:', torch.backends.mps.is_available())" \
  || { echo "STATUS: IMPORT_FAILED"; exit 3; }
python -m unittest test_hybrid || { echo "STATUS: TESTS_FAILED"; exit 4; }
caffeinate -i python hybrid.py --config configs/hybrid_smoke_may.json \
  || { echo "STATUS: FAILED_SMOKE"; exit 5; }
echo "=== smoke finish $(date '+%Y-%m-%d %H:%M:%S') ==="
echo "STATUS: SMOKE_DONE"
