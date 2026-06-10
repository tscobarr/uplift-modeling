# Uplift Modeling: Email Marketing Campaign

End-to-end uplift modeling project for targeted email marketing. Uses A/B testing and causal inference to identify which customers should receive promotional emails -- and which should not.

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

## Key Findings

### A/B Test
- Email produces statistically significant lift (p < 0.001) on both conversion and visit outcomes
- Effect varies by customer segment: higher recency and purchase history segments respond more

### Uplift Modeling
- **T-Learner** (two Random Forest models) achieved best AUUC: 35.81 vs 32.15 random baseline
- Customer classification into 4 quadrants:
  - **Persuadables** (35.9%): respond positively to email -- target these
  - **Sure Things** (28.4%): convert regardless -- wasted email cost
  - **Sleeping Dogs** (21.6%): react negatively -- do not target
  - **Lost Causes** (14.1%): never convert

### Business Impact
Targeting only Persuadables reduces email volume by 64% while capturing most conversions. The Streamlit dashboard includes a business simulator (cost per email, revenue per conversion) to quantify expected profit.

## Tech Stack

Python, pandas, numpy, scipy, scikit-learn, statsmodels, matplotlib, seaborn, plotly, Streamlit.

Uplift methods: S-Learner, T-Learner, X-Learner implemented with scikit-learn. Qini curves and AUUC for model evaluation.

## Dashboard

```
cd uplift-modeling
source .venv/bin/activate
streamlit run dashboard/app.py
```

Four tabs: A/B Test results, Uplift model metrics, Customer segmentation (4 quadrants), Business simulator.

## References

- Hillstrom, K. (2008). MineThatData Email Analytics Challenge
- Radcliffe, N. (2007). Using control groups to target on predicted lift
- Gutierrez, P. & Gerardy, J-Y. (2017). Causal Inference and Uplift Modeling
