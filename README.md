# Lecture 6 Assignment — Titanic Classification

**Repo:** https://github.com/MhmdSaker/lec6-titanic-classification

Evaluation, experimentation, and AI software-design patterns applied to the Titanic survival classification task.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

Outputs are written deterministically to `outputs/figures/` and `outputs/logs/`.

## Project structure

```
.
├── config.yaml              # all experiment settings (one source of truth)
├── requirements.txt         # pinned package versions
├── main.py                  # end-to-end runner: baseline + experiment + logs
├── REPORT.md                # full written report covering all 6 tasks
├── src/
│   ├── data.py              # load + preprocess Titanic (ColumnTransformer)
│   ├── models.py            # ModelFactory (Factory pattern)
│   ├── evaluation.py        # Metric strategies + confusion-matrix plotting
│   └── error_analysis.py    # build & summarise error table (Task 4)
└── outputs/
    ├── figures/             # confusion matrices, metric comparison bar chart
    └── logs/                # run_log.json, baseline_errors.csv, predictions
```

## What each task maps to

| Task | Location |
|---|---|
| 1. Dataset selection | `REPORT.md` § Task 1 · `src/data.py:describe_dataset` |
| 2. Baseline model | `REPORT.md` § Task 2 · `main.py:run_one_configuration` · `src/models.py` |
| 3. Evaluation metrics | `REPORT.md` § Task 3 · `src/evaluation.py` · `outputs/figures/confusion_matrix_baseline.png` |
| 4. Error analysis | `REPORT.md` § Task 4 · `src/error_analysis.py` · `outputs/logs/baseline_errors.csv` |
| 5. Experimentation (single factor) | `REPORT.md` § Task 5 · `outputs/figures/metric_comparison.png` |
| 6. Reproducibility | `REPORT.md` § Task 6 · `config.yaml` · `outputs/logs/run_log.json` |
