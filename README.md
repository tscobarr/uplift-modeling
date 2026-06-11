# Uplift Modeling: Email Marketing Campaign

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://uplift-modeling-d8kksx5aquge9anu8rbcac.streamlit.app)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)]()

End-to-end uplift modeling project for targeted email marketing. Uses A/B testing and causal inference to identify which customers should receive promotional emails -- and which should not.

## Quick Start

```bash
git clone https://github.com/tscobarr/uplift-modeling.git
cd uplift-modeling
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run dashboard/app.py
```

## Results

| Metric | Value |
|--------|-------|
| A/B test significance | p < 0.001 |
| Best model | T-Learner (AUUC 35.81 vs 32.15 baseline) |
| Persuadables | 35.9% of customers |
| Sure Things | 28.4% -- wasted cost |
| Sleeping Dogs | 21.6% -- negative reaction |
| Lost Causes | 14.1% -- never convert |
| Email reduction | 64% targeting only Persuadables |

## Dataset

Kevin Hillstrom MineThatData Email Analytics (64,000 customers, 2008).

Three randomly assigned groups:
- **No E-Mail** (control, n=21,306): 0.57% conversion
- **Mens E-Mail** (n=21,307): 1.25% conversion (+119% lift)
- **Womens E-Mail** (n=21,387): 0.88% conversion (+54% lift)

Features: recency, historical spend, purchase categories, location, channel, customer age.

## Project Structure

```
├── data/hillstrom.csv              Dataset
├── notebooks/
│   ├── 01_eda.ipynb                Exploratory data analysis
│   ├── 02_ab_testing.ipynb         A/B testing: lift, significance, power
│   └── 03_uplift_modeling.ipynb    Uplift models: S/T/X-Learners, Qini curves
├── dashboard/app.py                Streamlit dashboard (4 tabs)
├── requirements.txt
└── PLAN.md                         Full project plan
```

## Tech Stack

Python, pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, plotly, Streamlit.

Uplift methods: S-Learner, T-Learner, X-Learner implemented with scikit-learn. Qini curves and AUUC for model evaluation.

## Dashboard

[Live demo](https://uplift-modeling-d8kksx5aquge9anu8rbcac.streamlit.app) — four tabs: A/B Test results, Uplift model metrics, Customer segmentation (4 quadrants), Business simulator.

## Notebooks

- `01_eda.ipynb` — Exploratory data analysis
- `02_ab_testing.ipynb` — A/B testing: lift, statistical significance, power analysis
- `03_uplift_modeling.ipynb` — Uplift models: S-Learner, T-Learner, X-Learner, Qini curves

## References

- Hillstrom, K. (2008). MineThatData Email Analytics Challenge
- Radcliffe, N. (2007). Using control groups to target on predicted lift
- Gutierrez, P. & Gerardy, J-Y. (2017). Causal Inference and Uplift Modeling
