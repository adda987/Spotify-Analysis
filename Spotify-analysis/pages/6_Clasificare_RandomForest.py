import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Clasificare ML", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.page-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1DB954; font-weight: 700; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #1DB954; margin: 1.5rem 0 0.5rem 0; }
.metric-green { color: #1DB954; font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight:700; }
.metric-label { color: #888; font-size: 0.8rem; }
.info-box { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:1rem; }
hr.green { border: none; border-top: 1px solid #1DB95433; margin: 1.5rem 0; }
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

st.markdown('<div class="page-title"> Clasificare — Random Forest</div>', unsafe_allow_html=True)
st.markdown("Predicția genului muzical pe baza variabilelor audio. Feature importance, confusion matrix, cross-validation.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error(" `dataset.csv` nu a fost găsit.")
    st.stop()

# ── Configurare ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">️ Configurare Model</div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    n_genres = st.slider("Număr genuri (top N)", 3, 15, 8)
    n_trees = st.slider("Număr arbori (n_estimators)", 50, 300, 100, step=50)
with col2:
    test_size = st.slider("Proporție test set", 0.1, 0.4, 0.2, step=0.05)
    max_depth = st.slider("Adâncime maximă arbori", 3, 20, 10)

top_genres = df["track_genre"].value_counts().head(n_genres).index.tolist()
df_clf = df[df["track_genre"].isin(top_genres)].copy()

# Sample pentru viteză
MAX_SAMPLES = 20000
if len(df_clf) > MAX_SAMPLES:
    df_clf = df_clf.groupby("track_genre", group_keys=False).apply(
        lambda x: x.sample(min(len(x), MAX_SAMPLES // n_genres), random_state=42)
    )

features_clf = ["energy", "danceability", "valence", "loudness", "tempo",
                "acousticness", "speechiness", "instrumentalness", "liveness", "explicit"]
features_clf = [f for f in features_clf if f in df_clf.columns]

df_clf["explicit"] = df_clf["explicit"].astype(int)
df_clf = df_clf[features_clf + ["track_genre"]].dropna()

le = LabelEncoder()
y = le.fit_transform(df_clf["track_genre"])
X = df_clf[features_clf].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)

# Antrenare
with st.spinner(" Random Forest..."):
    rf = RandomForestClassifier(n_estimators=n_trees, max_depth=max_depth,
                                 random_state=42, n_jobs=-1, class_weight="balanced")
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)

acc = accuracy_score(y_test, y_pred)

# ── Metrici ───────────────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Performanța Modelului</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
metrics = [
    (f"{acc*100:.1f}%", "Accuracy (test)"),
    (f"{len(df_clf):,}", "Observații"),
    (f"{n_genres}", "Clase (genuri)"),
    (f"{n_trees}", "Arbori"),
]
for col, (val, label) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""<div class="info-box" style="text-align:center">
        <div class="metric-green">{val}</div>
        <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

baseline = 1 / n_genres * 100
st.info(f" **Comparație cu baseline**: Clasificare aleatoare = {baseline:.1f}% | Modelul nostru = {acc*100:.1f}% — de **{acc*100/baseline:.1f}x** mai bun.")

# ── Feature Importance ────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Feature Importance</div>', unsafe_allow_html=True)

importance_df = pd.DataFrame({
    "Feature": features_clf,
    "Importanță": rf.feature_importances_
}).sort_values("Importanță", ascending=True)

fig_imp = px.bar(importance_df, x="Importanță", y="Feature", orientation="h",
                  color="Importanță",
                  color_continuous_scale=[[0, "#0d4d1f"], [0.5, "#1DB954"], [1, "#7fff9a"]],
                  template="plotly_dark",
                  title="Feature Importance — cât contribuie fiecare variabilă la clasificare")
fig_imp.update_layout(
    height=400, paper_bgcolor="#0d0d0d", showlegend=False,
    font=dict(color="white"), xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
)
st.plotly_chart(fig_imp, use_container_width=True)

top_feat = importance_df.iloc[-1]["Feature"]
st.markdown(f">  **Cel mai important feature** pentru clasificarea genului este **`{top_feat}`** — are cel mai mare putere discriminantă între genuri.")

# ── Confusion Matrix ──────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Confusion Matrix</div>', unsafe_allow_html=True)

cm = confusion_matrix(y_test, y_pred)
genre_labels = le.classes_

# Normalizare
cm_norm = cm.astype(float) / cm.sum(axis=1)[:, np.newaxis]

fig_cm = px.imshow(
    cm_norm,
    x=genre_labels, y=genre_labels,
    color_continuous_scale=[[0, "#0d0d0d"], [0.3, "#0d4d1f"], [1, "#1DB954"]],
    template="plotly_dark",
    title="Confusion Matrix Normalizată (pe rând = recall per clasă)",
    text_auto=".2f"
)
fig_cm.update_layout(
    height=550, paper_bgcolor="#0d0d0d",
    font=dict(color="white", size=11),
    xaxis=dict(title="Predicție", tickangle=30),
    yaxis=dict(title="Actual")
)
st.plotly_chart(fig_cm, use_container_width=True)

st.markdown("> **Diagonala** = recall per clasă (cât din piesele unui gen au fost corect identificate). Valori mari pe diagonală = model bun per clasă.")

# ── Classification Report ─────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Raport Clasificare per Gen</div>', unsafe_allow_html=True)

report = classification_report(y_test, y_pred, target_names=genre_labels, output_dict=True)
report_df = pd.DataFrame(report).T.round(3)
report_df = report_df[report_df.index.isin(genre_labels)]
report_df.columns = ["Precision", "Recall", "F1-Score", "Support"]
report_df = report_df.sort_values("F1-Score", ascending=False)
st.dataframe(report_df, use_container_width=True)

st.markdown("""
> 💡 **Interpretare**:
> - **Precision** = din toate piesele prezise ca gen X, câte chiar sunt X
> - **Recall** = din toate piesele reale de gen X, câte au fost identificate corect
> - **F1-Score** = media armonică a celor două — metrica principală de evaluat
""")

# ── Cross-Validation ──────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Cross-Validation (5-fold)</div>', unsafe_allow_html=True)

with st.spinner("Rulează 5-fold cross-validation..."):
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="accuracy", n_jobs=-1)

fig_cv = go.Figure()
fig_cv.add_trace(go.Bar(
    x=[f"Fold {i+1}" for i in range(5)],
    y=cv_scores * 100,
    marker_color=["#1DB954" if s >= cv_scores.mean() else "#FF6B6B" for s in cv_scores],
    text=[f"{s*100:.1f}%" for s in cv_scores],
    textposition="outside"
))
fig_cv.add_hline(y=cv_scores.mean() * 100, line_dash="dash", line_color="white",
                  annotation_text=f"Medie: {cv_scores.mean()*100:.1f}%",
                  annotation_position="right")
fig_cv.update_layout(
    title=f"Accuracy per Fold | Medie: {cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%",
    yaxis_title="Accuracy (%)", template="plotly_dark", height=350,
    paper_bgcolor="#0d0d0d", font=dict(color="white"),
    yaxis=dict(range=[0, 100], gridcolor="#222")
)
st.plotly_chart(fig_cv, use_container_width=True)

st.success(f" Cross-validation : **{cv_scores.mean()*100:.1f}% ± {cv_scores.std()*100:.1f}%** — modelul este {'stabil' if cv_scores.std() < 0.03 else 'cu variabilitate moderată'} între folduri.")
