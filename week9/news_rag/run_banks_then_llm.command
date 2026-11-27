#!/bin/bash
# Докачка счётчиков по главным акциям (до 15:00), сборка инфополя, затем все прогоны M (run_hw3_llm.command).
cd "$(dirname "$0")" || exit 1
bash run_news_banks.command
bash run_hw3_llm.command
