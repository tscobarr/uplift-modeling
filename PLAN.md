# Uplift Modeling Project Plan

## Objetivo
Proyecto end-to-end de uplift modeling con A/B testing e inferencia causal.
Dataset: Kevin Hillstrom MineThatData Email Analytics.

## Dataset
- **Origen**: `data/hillstrom.csv` (64,000 clientes, 12 columnas)
- **Tratamiento**: 3 brazos aleatorizados — No E-Mail (control), Mens E-Mail, Womens E-Mail
- **Outcomes**: `visit` (visitó el sitio), `conversion` (compró)
- **Features**: recency, history_segment, history, mens, womens, zip_code, newbie, channel

## Estructura del proyecto
```
uplift-modeling/
├── data/
│   └── hillstrom.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_ab_testing.ipynb
│   └── 03_uplift_modeling.ipynb
├── src/
│   ├── preprocessing.py
│   ├── ab_test.py
│   ├── uplift_models.py
│   └── plots.py
├── dashboard/
│   └── app.py (Streamlit)
├── requirements.txt
├── README.md
└── PLAN.md
```

## Requisitos técnicos
```
pandas, numpy, scipy, scikit-learn, statsmodels
econml, dowhy
matplotlib, seaborn, plotly
streamlit
```

## Fases

### Fase 1 — EDA (notebook 01)
- Carga y limpieza
- Estadísticas descriptivas por segmento
- Visualización de distribuciones de features
- Análisis de balance entre grupos
- Mapas de correlación

### Fase 2 — A/B Testing (notebook 02)
Para cada par (Mens vs Control, Womens vs Control):

- **Overall lift**: diferencia en conversion rate, intervalo de confianza
- **Significancia**: chi-cuadrado para proporciones, p-value, tamaño de efecto (Cohen's h)
- **Power analysis**: ¿era la muestra suficiente? potencia post-hoc
- **Por feature**: lift desagregado por recency, history_segment, channel, zip_code
- **Visualización**: barras con IC, funnel de conversión, forest plot por segmento

### Fase 3 — Uplift Modeling (notebook 03)
Agrupar Mens + Womens como "treatment" vs control:

- **Meta-learners con EconML**:
  - S-Learner (un modelo con treatment como feature)
  - T-Learner (un modelo por grupo)
  - X-Learner (modelos cruzados, mejor para desbalance)
- **Métrica de evaluación**: Qini curve, uplift curve, AUUC (Area Under Uplift Curve)
- **Clasificación de clientes** en 4 cuadrantes:
  - Persuadables: uplift > 0 (compran solo con email)
  - Sure Things: compran sin email, uplift ≈ 0
  - Sleeping Dogs: uplift < 0 (reaccionan negativamente)
  - Lost Causes: no compran nunca

### Fase 4 — Dashboard (Streamlit)
- Tab 1: A/B Test Results — lifts, significancia, power
- Tab 2: Uplift Model — curvas Qini, distribución de uplift
- Tab 3: Customer Segmentation — los 4 cuadrantes con filtros
- Tab 4: Business Simulator — "si targeteás top X% por uplift, ganancia esperada = $Y"

## Milestones

| Semana | Entregable |
|--------|-----------|
| Día 1-2 | EDA completo, datos preparados |
| Día 3-4 | A/B testing completo con todos los análisis |
| Día 5-7 | Uplift models entrenados, evaluados, documentados |
| Día 8-10 | Dashboard completo, README, repo público |

## Stack
- **Datos**: pandas, numpy, scipy
- **A/B testing**: statsmodels, scipy.stats
- **Causal inference**: EconML (Microsoft), DoWhy (Microsoft Research)
- **Visualización**: matplotlib, seaborn, plotly
- **Dashboard**: Streamlit

## Referencias
- Hillstrom, K. (2008). MineThatData Email Analytics Challenge
- Radcliffe, N. (2007). Using control groups to target on predicted lift
- Gutierrez, P. & Gérardy, J-Y. (2017). Causal Inference and Uplift Modeling

## Entregable final
- Repo público en GitHub con README completo
- 3 notebooks ejecutados con outputs visibles
- Dashboard deployado en Streamlit Cloud
- Bullet de proyecto para CV
