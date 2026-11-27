#!/bin/bash
# Докачать только счётчики статей (timelinevolraw) по самым важным пропущенным акциям, не дольше чем до 15:00.
set -u
cd "$(dirname "$0")" || exit 1
exec > >(tee -a "run_news_collect.log") 2>&1
PY="$PWD/../../week8/far-trans-llm/.venv/bin/python"
caffeinate -i "$PY" collect_gdelt.py --modes timelinevolraw --pause 20 --until 15:00 \
  --isins GRS014003032,GRS015003007,GRS003003035,GRS145003000,GRS191213008,GRS213213002
"$PY" build_infofield.py
echo "STATUS: BANKS_DONE"
