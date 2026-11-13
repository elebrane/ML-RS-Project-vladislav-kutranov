# ML-RS Project — Vladislav Kutranov

LLM4Rec course (HSE University, GSB/AVITO cohort): team project on FAR-Trans financial asset recommendation.

## Week 7 — Session 1: research question and study plan

- **Report draft:** [week7/session1_work/report/vladislav_kutranov_session1_report_DRAFT.pdf](week7/session1_work/report/vladislav_kutranov_session1_report_DRAFT.pdf) (HTML version and build script in the same folder).
- **Data overview (interactive):** [week7/fartrans_overview.html](https://elebrane.github.io/ML-RS-Project-vladislav-kutranov/week7/fartrans_overview.html) — table keys, dashboard with date / customer / asset filters, figures of the FAR-Trans paper.
- **Analysis and notes:** [week7/session1_work/](week7/session1_work/)
  - data audit: `explore_fartrans.py`, `explore_details.py` → `data_stats.txt`, `data_details.txt`, `step1_data_notes.md`
  - dashboard data: `build_dashboard_data.py`, `build_dashboard_cube.py`
  - data example: `pick_record.py` → `session1_data_example.md`
  - related work, research questions, hypotheses: `session1_related_work.md`, `research_questions_scored.md`, `project_hypotheses.md`
  - `baseline_popularity.py`, `verify_example.py`: written and run as checks; their results are not used in the Session 1 report.
- **Course task descriptions:** [week7/project_GSB_AVITO.md](week7/project_GSB_AVITO.md), [week7/session1_research_question_instructions.md](week7/session1_research_question_instructions.md)
- **Raw data:** [week7/Data FAR-Trans/](week7/Data%20FAR-Trans/)

## Reproduce

```bash
cd week7/session1_work
python3 explore_fartrans.py "../Data FAR-Trans"
python3 explore_details.py "../Data FAR-Trans"
python3 build_dashboard_data.py "../Data FAR-Trans"
python3 build_dashboard_cube.py "../Data FAR-Trans"
python3 pick_record.py "../Data FAR-Trans"
```

Python 3.10, pandas 2.3.3, numpy 2.2.6.

## Data licence

FAR-Trans is redistributed unchanged with its README under CC BY 4.0. Cite: J. Sanz-Cruzado, N. Droukas, R. McCreadie, "FAR-Trans: An Investment Dataset for Financial Asset Recommendation," IJCAI-2024 Workshop on Recommender Systems in Finance (Fin-RecSys), 2024. Source: https://researchdata.gla.ac.uk/1658/
