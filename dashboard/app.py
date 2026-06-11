"""
Dashboard de Uplift Modeling — Hillstrom dataset
Ejecutar: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc
from statsmodels.stats.proportion import proportion_confint

st.set_page_config(page_title="Uplift Dashboard", layout="wide")

DATA_PATH = "data/hillstrom.csv"

# ---------------------------------------------------------------------------
# Cache: data loading & model training (run once)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Cargando datos...")
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["treatment"] = (df["segment"] != "No E-Mail").astype(int)
    return df


@st.cache_data(show_spinner="Entrenando modelo T-Learner...")
def train_models(_df):
    """Train T-Learner: two RandomForest models (control, treatment)."""
    cat_cols = ["history_segment", "zip_code", "channel"]
    df = pd.get_dummies(_df, columns=cat_cols, drop_first=True)

    treat = df[df["treatment"] == 1]
    ctrl = df[df["treatment"] == 0]

    drop_cols = [
        "segment", "treatment", "conversion", "visit", "spend",
        "history_segment", "zip_code", "channel",
    ]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    rf_ctrl = RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=1
    )
    rf_treat = RandomForestClassifier(
        n_estimators=50, max_depth=8, random_state=42, n_jobs=1
    )

    rf_ctrl.fit(ctrl[feature_cols], ctrl["conversion"])
    rf_treat.fit(treat[feature_cols], treat["conversion"])

    df["p_ctrl"] = rf_ctrl.predict_proba(df[feature_cols])[:, 1]
    df["p_treat"] = rf_treat.predict_proba(df[feature_cols])[:, 1]
    df["uplift"] = df["p_treat"] - df["p_ctrl"]
    return df, feature_cols


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def wilson_ci(n, k, z=1.96):
    """Wilson confidence interval for a proportion."""
    if n == 0:
        return (0.0, 0.0)
    return proportion_confint(k, n, alpha=0.05, method="wilson")


def qini_curve(df, uplift_col="uplift", outcome="conversion", n_bins=100):
    df_sorted = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
    total_treat = df_sorted["treatment"].sum()
    total_ctrl = len(df_sorted) - total_treat
    ratio = total_treat / total_ctrl if total_ctrl > 0 else 0

    n = len(df_sorted)
    step = max(1, n // n_bins)
    results = []
    cum_inc = 0.0

    for i in range(0, n, step):
        chunk = df_sorted.iloc[i : i + step]
        inc = chunk[outcome].sum() - ratio * chunk.loc[~chunk["treatment"].astype(bool), outcome].sum()
        cum_inc += inc
        results.append({"pct": (i + len(chunk)) / n * 100, "cum_inc": cum_inc})

    qdf = pd.DataFrame(results)
    total_inc = qdf["cum_inc"].iloc[-1]
    qdf["random"] = np.linspace(0, total_inc, len(qdf))

    sorted_treat = df_sorted[df_sorted["treatment"] == 1].sort_values(outcome, ascending=False)
    n_treat = len(sorted_treat)
    perfect = []
    cum_perf = 0.0
    for i in range(0, n, step):
        chunk = sorted_treat.iloc[i : i + step] if i < n_treat else pd.DataFrame()
        cum_perf += chunk[outcome].sum() if len(chunk) else 0
        perfect.append(cum_perf)
    qdf["perfect"] = perfect[: len(qdf)]
    return qdf


def compute_auuc(df, outcome="conversion"):
    qdf = qini_curve(df, outcome=outcome)
    return auc(qdf["pct"] / 100, qdf["cum_inc"]) - auc(qdf["pct"] / 100, qdf["random"])


# ---------------------------------------------------------------------------
# Load data once
# ---------------------------------------------------------------------------
df_raw = load_data()

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.header("Configuracion")
outcome = st.sidebar.selectbox(
    "Variable objetivo", ["conversion", "visit"],
    format_func=lambda x: "Conversion" if x == "conversion" else "Visita"
)

# ---------------------------------------------------------------------------
# Tab 1: A/B Test
# ---------------------------------------------------------------------------
def tab_ab_test():
    st.header("A/B Test")
    kpi_cols, _ = st.columns([0.3, 0.01])

    ctrl = df_raw[df_raw["treatment"] == 0]
    treat = df_raw[df_raw["treatment"] == 1]
    n_c, n_t = len(ctrl), len(treat)
    k_c, k_t = ctrl[outcome].sum(), treat[outcome].sum()
    rate_c, rate_t = k_c / n_c, k_t / n_t
    abs_lift = rate_t - rate_c
    rel_lift = abs_lift / rate_c * 100 if rate_c > 0 else 0

    ci_c = wilson_ci(n_c, k_c)
    ci_t = wilson_ci(n_t, k_t)

    with kpi_cols:
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Control (No Email)", f"{rate_c:.2%}")
        r2.metric("Tratamiento (Email)", f"{rate_t:.2%}")
        r3.metric("Lift Absoluto", f"{abs_lift:+.2%}")
        r4.metric("Lift Relativo", f"{rel_lift:+.2f}%")

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Control", "Tratamiento"],
        y=[rate_c, rate_t],
        error_y=dict(
            type="data", symmetric=False,
            array=[[ci_t[1] - rate_t], [rate_c - ci_c[0]]],
            arrayminus=[[rate_t - ci_t[0]], [rate_c - ci_c[0]]],
            visible=True,
        ),
        marker_color=["#1f77b4", "#ff7f0e"],
        text=[f"{rate_c:.2%}", f"{rate_t:.2%}"],
        textposition="outside",
    ))
    fig.update_layout(
        title=f"Tasa de {'Conversion' if outcome == 'conversion' else 'Visita'}",
        yaxis=dict(tickformat=".1%"), height=400,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Analisis por subgrupo"):
        sub = st.selectbox("Variable", ["history_segment", "zip_code", "channel"])
        sub_df = df_raw.groupby(sub).apply(
            lambda g: pd.Series({
                "n": len(g),
                "ctrl_rate": g[g["treatment"] == 0][outcome].mean(),
                "treat_rate": g[g["treatment"] == 1][outcome].mean(),
                "lift": g[g["treatment"] == 1][outcome].mean() - g[g["treatment"] == 0][outcome].mean(),
            })
        ).reset_index()
        sub_df.columns = [sub, "Muestras", "Control", "Tratamiento", "Lift"]
        sub_df["Lift %"] = sub_df["Lift"] / sub_df["Control"] * 100
        sub_df = sub_df.sort_values("Lift", ascending=False)
        st.dataframe(sub_df.style.format({
            "Muestras": "{:,.0f}", "Control": "{:.2%}",
            "Tratamiento": "{:.2%}", "Lift": "{:+.4f}", "Lift %": "{:+.2f}%",
        }), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 2: Model
# ---------------------------------------------------------------------------
def tab_modelo():
    st.header("Modelo de Uplift")
    df, _ = train_models(df_raw)
    met1, met2, met3 = st.columns(3)
    met1.metric("AUUC", f"{compute_auuc(df, outcome):.4f}")
    met2.metric("Uplift Promedio", f"{df['uplift'].mean():+.4f}")
    met3.metric("% Uplift Positivo", f"{(df['uplift'] > 0).mean():.1%}")

    qdf = qini_curve(df, outcome=outcome)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=qdf["pct"], y=qdf["perfect"], name="Perfecto", line=dict(dash="dot", color="green")))
    fig.add_trace(go.Scatter(x=qdf["pct"], y=qdf["random"], name="Aleatorio", line=dict(dash="dot", color="gray")))
    fig.add_trace(go.Scatter(x=qdf["pct"], y=qdf["cum_inc"], name="T-Learner", line=dict(color="#ff7f0e", width=3)))
    fig.update_layout(
        title="Curva Qini", xaxis_title="% Clientes", yaxis_title="Ganancia Acumulada", height=450
    )
    st.plotly_chart(fig, use_container_width=True)

    fig2 = px.histogram(df, x="uplift", nbins=60, labels={"uplift": "Uplift"})
    fig2.add_vline(x=0, line_dash="dash", line_color="red", opacity=0.6)
    fig2.update_layout(title="Distribucion del Uplift", height=400)
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 3: Segmentation
# ---------------------------------------------------------------------------
def tab_segmentacion():
    st.header("Segmentacion")
    df, _ = train_models(df_raw)
    df["p_conv"] = (df["p_ctrl"] + df["p_treat"]) / 2
    thr_uplift = st.slider("Umbral de Uplift", -0.05, 0.15, 0.0, 0.001, format="%+.3f")
    thr_conv = df["p_conv"].median()

    def quadrant(row):
        if row["uplift"] >= thr_uplift and row["p_conv"] >= thr_conv:
            return "Sure Things"
        if row["uplift"] >= thr_uplift and row["p_conv"] < thr_conv:
            return "Persuadables"
        if row["uplift"] < thr_uplift and row["p_conv"] >= thr_conv:
            return "Sleeping Dogs"
        return "Lost Causes"

    df["quadrant"] = df.apply(quadrant, axis=1)
    order = ["Sure Things", "Persuadables", "Sleeping Dogs", "Lost Causes"]
    colors = {"Sure Things": "#2ecc71", "Persuadables": "#3498db", "Sleeping Dogs": "#f1c40f", "Lost Causes": "#e74c3c"}

    counts = df["quadrant"].value_counts().reindex(order, fill_value=0)
    fig = px.bar(x=counts.index, y=counts.values, color=counts.index,
                 color_discrete_map=colors, labels={"x": "Cuadrante", "y": "Clientes"},
                 text=counts.values, height=400)
    fig.update_layout(title="Distribucion de Cuadrantes", showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

    tbl_data = []
    for q in order:
        sub = df[df["quadrant"] == q]
        if len(sub) == 0:
            tbl_data.append([q, 0, 0, None, None, None])
            continue
        top_ch = sub["channel"].mode().iloc[0]
        tbl_data.append([q, len(sub), len(sub) / len(df) * 100,
                         sub["recency"].mean(), sub["history"].mean(), top_ch])
    tbl = pd.DataFrame(tbl_data, columns=["Cuadrante", "Clientes", "%", "Recency", "History", "Canal"])
    st.dataframe(tbl.style.format({
        "Clientes": "{:,.0f}", "%": "{:.1f}%",
        "Recency": "{:.1f}", "History": "${:,.0f}",
    }), hide_index=True, use_container_width=True)


# ---------------------------------------------------------------------------
# Tab 4: Business Simulator
# ---------------------------------------------------------------------------
def tab_simulador():
    st.header("Simulador de Negocio")
    col1, col2, col3 = st.columns(3)
    cost_email = col1.number_input("Coste por email ($)", 0.0, 1.0, 0.05, 0.01)
    rev_conv = col2.number_input("Ingreso por conversion ($)", 0.0, 200.0, 50.0, 1.0)

    df, _ = train_models(df_raw)
    df = df.sort_values("uplift", ascending=False).reset_index(drop=True)

    pct = col3.slider("% Clientes objetivo", 0, 100, 20) / 100
    n_target = max(1, int(len(df) * pct))
    targeted = df.iloc[:n_target]

    base_conv = targeted.loc[targeted["treatment"] == 0, "p_ctrl"].sum()
    uplift_conv = targeted.loc[targeted["treatment"] == 1, "uplift"].sum()
    expected_conv = base_conv + uplift_conv

    emails = n_target
    cost = emails * cost_email
    revenue = expected_conv * rev_conv
    profit = revenue - cost

    r1, r2, r3, r4, r5 = st.columns(5)
    r1.metric("Emails enviados", f"{emails:,.0f}")
    r2.metric("Conversiones esperadas", f"{expected_conv:,.1f}")
    r3.metric("Coste", f"${cost:,.2f}")
    r4.metric("Ingreso", f"${revenue:,.2f}")
    r5.metric("Ganancia", f"${profit:,.2f}")

    fig = go.Figure(go.Waterfall(
        name="Flujo", orientation="v",
        measure=["relative", "relative", "total"],
        x=["Ingreso", "Coste emails", "Ganancia neta"],
        y=[revenue, -cost, profit],
        decreasing={"marker": {"color": "#e74c3c"}},
        increasing={"marker": {"color": "#2ecc71"}},
        totals={"marker": {"color": "#3498db"}},
    ))
    fig.update_layout(title="Waterfall: Ingreso a Ganancia", height=400, waterfallgap=0.3)
    st.plotly_chart(fig, use_container_width=True)

    pcts = np.linspace(0.01, 1.0, 50)
    curve = []
    for p in pcts:
        sub = df.iloc[:max(1, int(len(df) * p))]
        base = sub.loc[sub["treatment"] == 0, "p_ctrl"].sum()
        up = sub.loc[sub["treatment"] == 1, "uplift"].sum()
        exp = base + up
        curve.append({"pct": p * 100, "profit": exp * rev_conv - len(sub) * cost_email,
                      "customers": len(sub)})
    cdf = pd.DataFrame(curve)
    fig2 = px.line(cdf, x="customers", y="profit", markers=True,
                   labels={"customers": "Clientes", "profit": "Ganancia ($)"},
                   title="Clientes vs Ganancia")
    fig2.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig2.update_layout(height=400)
    st.plotly_chart(fig2, use_container_width=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tabs = st.tabs(["A/B Test", "Modelo de Uplift", "Segmentacion", "Simulador"])

with tabs[0]:
    tab_ab_test()
with tabs[1]:
    tab_modelo()
with tabs[2]:
    tab_segmentacion()
with tabs[3]:
    tab_simulador()
