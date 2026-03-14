import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Regresie Liniară", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.page-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1DB954; font-weight: 700; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #1DB954; margin: 1.5rem 0 0.5rem 0; }
.metric-green { color: #1DB954; font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight:700; }
.metric-label { color: #888; font-size: 0.8rem; }
.info-box { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:1rem; margin-bottom:0.8rem; }
.coef-pos { color: #1DB954; font-family: 'Space Mono', monospace; }
.coef-neg { color: #FF6B6B; font-family: 'Space Mono', monospace; }
hr.green { border: none; border-top: 1px solid #1DB95433; margin: 1.5rem 0; }
.formula-box { background:#111; border:1px solid #1DB95444; border-radius:8px; padding:1rem 1.5rem;
               font-family:'Space Mono',monospace; font-size:0.9rem; color:#f0f0f0; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=["track_name", "artists", "track_genre"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

st.markdown('<div class="page-title"> Regresie Liniară — Model OLS</div>', unsafe_allow_html=True)
st.markdown("Predicția popularității pe baza variabilelor audio. Interpretare econometrică a coeficienților.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error(" `dataset.csv` nu a fost găsit.")
    st.stop()

# ── Configurare model ─────────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Configurare Model</div>', unsafe_allow_html=True)

available_features = ["energy", "danceability", "valence", "loudness", "tempo",
                       "acousticness", "speechiness", "instrumentalness", "liveness",
                       "explicit", "duration_ms", "key", "mode", "time_signature"]
available_features = [f for f in available_features if f in df.columns]

col_cfg1, col_cfg2 = st.columns(2)
with col_cfg1:
    selected_features = st.multiselect(
        "Variabile independente (X)",
        options=available_features,
        default=["energy", "danceability", "loudness", "valence", "acousticness", "tempo"]
    )
with col_cfg2:
    test_size = st.slider("Proporție test set", 0.1, 0.4, 0.2, step=0.05)
    normalize = st.checkbox("Standardizează variabilele (StandardScaler)", value=True)
    filter_genre = st.selectbox("Filtrează pe gen (opțional)", ["Toate genurile"] + sorted(df["track_genre"].unique().tolist()))

if not selected_features:
    st.warning("Selectează cel puțin o variabilă independentă.")
    st.stop()

# ── Pregătire date ────────────────────────────────────────────────────────────
df_model = df.copy()
if filter_genre != "Toate genurile":
    df_model = df_model[df_model["track_genre"] == filter_genre]

if "explicit" in selected_features:
    df_model["explicit"] = df_model["explicit"].astype(int)

df_model = df_model[selected_features + ["popularity"]].dropna()

X = df_model[selected_features].values
y = df_model["popularity"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

if normalize:
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
else:
    X_train_sc = X_train
    X_test_sc = X_test

model = LinearRegression()
model.fit(X_train_sc, y_train)
y_pred = model.predict(X_test_sc)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2_train = model.score(X_train_sc, y_train)

# ── Metrici model ─────────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Performanța Modelului</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (f"{r2:.4f}", "R² (test)"),
    (f"{r2_train:.4f}", "R² (train)"),
    (f"{rmse:.2f}", "RMSE"),
    (f"{mae:.2f}", "MAE"),
    (f"{len(df_model):,}", "Observații"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(f"""<div class="info-box" style="text-align:center">
        <div class="metric-green">{val}</div>
        <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

# Interpretare R²
r2_pct = r2 * 100
if r2 < 0.1:
    r2_interp = f"Modelul explică doar **{r2_pct:.1f}%** din variația popularității — putere explicativă redusă. Popularitatea depinde de factori non-audio (marketing, viralitate etc.)."
elif r2 < 0.3:
    r2_interp = f"Modelul explică **{r2_pct:.1f}%** din variația popularității — putere explicativă moderată."
else:
    r2_interp = f"Modelul explică **{r2_pct:.1f}%** din variația popularității — putere explicativă bună."

st.info(f" **Interpretare R²**: {r2_interp}")

# ── Formula model ─────────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Ecuația estimată</div>', unsafe_allow_html=True)

intercept = model.intercept_
coefs = model.coef_

formula_parts = [f"{c:+.4f}·{f}" for c, f in zip(coefs, selected_features)]
formula_str = f"popularity = {intercept:.3f}  " + "  ".join(formula_parts)
st.markdown(f'<div class="formula-box">{formula_str}</div>', unsafe_allow_html=True)

if normalize:
    st.caption("️ Coeficienții sunt pe variabile standardizate (β standardizați) — comparabili ca magnitudine între ei.")
else:
    st.caption(" Coeficienții sunt nestandarizați — magnitudinile depind de scala fiecărei variabile.")

# ── Tabel coeficienți ─────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Tabel Coeficienți — Interpretare Econometrică</div>', unsafe_allow_html=True)

coef_df = pd.DataFrame({
    "Variabilă": selected_features,
    "Coeficient (β)": coefs.round(4),
    "Efect": ["↑ Pozitiv" if c > 0 else "↓ Negativ" for c in coefs],
    "|β| (magnitudine)": np.abs(coefs).round(4),
    "Interpretare": [
        f"O creștere cu 1 {'unitate' if not normalize else 'std'} în {f} este asociată cu o {'creștere' if c > 0 else 'scădere'} de {abs(c):.3f} puncte în popularitate, ceteris paribus."
        for f, c in zip(selected_features, coefs)
    ]
}).sort_values("|β| (magnitudine)", ascending=False).reset_index(drop=True)

st.dataframe(coef_df[["Variabilă", "Coeficient (β)", "Efect", "|β| (magnitudine)", "Interpretare"]],
             use_container_width=True, hide_index=True)

# ── Grafic coeficienți ────────────────────────────────────────────────────────
fig_coef = go.Figure()
colors_coef = ["#1DB954" if c > 0 else "#FF6B6B" for c in coefs]

fig_coef.add_trace(go.Bar(
    x=coefs, y=selected_features, orientation="h",
    marker_color=colors_coef, marker_line_width=0,
    hovertemplate="%{y}: β = %{x:.4f}<extra></extra>"
))
fig_coef.add_vline(x=0, line_color="white", line_dash="dash", line_width=1)
fig_coef.update_layout(
    title="Coeficienți β — Verde = efect pozitiv, Roșu = efect negativ",
    xaxis_title="Valoare coeficient",
    template="plotly_dark", height=400, paper_bgcolor="#0d0d0d",
    font=dict(color="white"), xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
)
st.plotly_chart(fig_coef, use_container_width=True)

# ── Actual vs Predicted ───────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Actual vs Predicted</div>', unsafe_allow_html=True)

sample_idx = np.random.choice(len(y_test), min(1000, len(y_test)), replace=False)

fig_avp = go.Figure()
fig_avp.add_trace(go.Scatter(
    x=y_test[sample_idx], y=y_pred[sample_idx],
    mode="markers", marker=dict(color="#1DB954", opacity=0.4, size=5),
    name="Observații", hovertemplate="Actual: %{x}<br>Predicted: %{y:.1f}<extra></extra>"
))
min_val = min(y_test.min(), y_pred.min())
max_val = max(y_test.max(), y_pred.max())
fig_avp.add_trace(go.Scatter(
    x=[min_val, max_val], y=[min_val, max_val],
    mode="lines", line=dict(color="white", dash="dash", width=1.5),
    name="Linie perfectă (y=x)"
))
fig_avp.update_layout(
    title=f"Actual vs Predicted Popularity (R² = {r2:.4f})",
    xaxis_title="Valori reale", yaxis_title="Valori prezise",
    template="plotly_dark", height=450, paper_bgcolor="#0d0d0d",
    font=dict(color="white"), xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
)
st.plotly_chart(fig_avp, use_container_width=True)

# ── Reziduuri ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📉 Analiza Reziduurilor</div>', unsafe_allow_html=True)

residuals = y_test - y_pred

col_r1, col_r2 = st.columns(2)

with col_r1:
    fig_res1 = px.histogram(residuals[sample_idx], nbins=60,
                             title="Distribuția Reziduurilor",
                             template="plotly_dark",
                             color_discrete_sequence=["#1DB954"])
    fig_res1.update_layout(paper_bgcolor="#0d0d0d", font=dict(color="white"),
                            xaxis_title="Reziduu (actual - predicted)",
                            showlegend=False, xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"))
    st.plotly_chart(fig_res1, use_container_width=True)

with col_r2:
    fig_res2 = go.Figure()
    fig_res2.add_trace(go.Scatter(
        x=y_pred[sample_idx], y=residuals[sample_idx],
        mode="markers", marker=dict(color="#4D96FF", opacity=0.4, size=4),
        name="Reziduuri"
    ))
    fig_res2.add_hline(y=0, line_color="white", line_dash="dash", line_width=1.5)
    fig_res2.update_layout(
        title="Reziduuri vs Fitted Values",
        xaxis_title="Valori prezise", yaxis_title="Reziduuri",
        template="plotly_dark", paper_bgcolor="#0d0d0d", font=dict(color="white"),
        xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
    )
    st.plotly_chart(fig_res2, use_container_width=True)

res_mean = residuals.mean()
res_std = residuals.std()
st.markdown(f"""
>  **Ipotezele OLS**: Medie reziduuri ≈ {res_mean:.3f} (ideal: 0). 
> Std dev reziduuri = {res_std:.2f}. 
> Dacă distribuția e aproximativ normală și dispersia e constantă (homoskedasticity), ipotezele Gauss-Markov sunt satisfăcute.
""")

# ── Predictor interactiv ──────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Predictor Interactiv</div>', unsafe_allow_html=True)
st.markdown("Introdu valori custom și estimează popularitatea predicției.")

input_cols = st.columns(len(selected_features))
input_vals = []

defaults = {
    "energy": 0.7, "danceability": 0.65, "valence": 0.5, "loudness": -6.0,
    "tempo": 120.0, "acousticness": 0.2, "speechiness": 0.05,
    "instrumentalness": 0.0, "liveness": 0.15, "explicit": 0,
    "duration_ms": 200000, "key": 5, "mode": 1, "time_signature": 4
}

for col, feat in zip(input_cols, selected_features):
    with col:
        default_val = defaults.get(feat, 0.5)
        val = st.number_input(feat, value=float(default_val), format="%.3f", key=f"pred_{feat}")
        input_vals.append(val)

if st.button(" Estimează popularitatea", type="primary"):
    input_arr = np.array(input_vals).reshape(1, -1)
    if normalize:
        input_arr = scaler.transform(input_arr)
    prediction = model.predict(input_arr)[0]
    prediction = np.clip(prediction, 0, 100)
    
    st.success(f"###  Popularitate estimată: **{prediction:.1f} / 100**")
    
    if prediction >= 70:
        st.markdown(" **Hit potențial!** — Caracteristicile introduse sugerează o piesă foarte populară.")
    elif prediction >= 40:
        st.markdown(" **Popularitate medie** — Piesă cu potențial moderat.")
    else:
        st.markdown(" **Nișă sau underground** — Piesă cu popularitate redusă estimată.")
