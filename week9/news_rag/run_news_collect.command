#!/bin/bash
# HW3, шаг 1: сбор новостного слоя из GDELT (только на Mac: из облака и VM GDELT недоступен).
# Двойной щелчок в Finder. Mac не уснёт (caffeinate). Можно прервать и запустить снова — скачанное не повторяется.
# Только 30 самых значимых акций (остальные почти не встречаются в блоках). Около 30–90 минут. Итог — строка STATUS.
set -u
cd "$(dirname "$0")" || exit 1
LOG="run_news_collect.log"
exec > >(tee -a "$LOG") 2>&1
echo "=== news collect start $(date '+%Y-%m-%d %H:%M:%S') ==="
PY="$PWD/../../week8/far-trans-llm/.venv/bin/python"
[ -x "$PY" ] || PY="$(command -v python3)"
echo "python: $PY"
caffeinate -i "$PY" collect_gdelt.py --limit 30
RC=$?
"$PY" build_infofield.py || { echo "STATUS: BUILD_FAILED"; exit 6; }
echo "=== news collect finish $(date '+%Y-%m-%d %H:%M:%S') ==="
[ $RC -eq 0 ] && echo "STATUS: NEWS_DONE" || echo "STATUS: NEWS_PARTIAL (запустите ещё раз, чтобы докачать)"
