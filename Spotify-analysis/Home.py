import streamlit as st

st.set_page_config(
    page_title="Spotify ML Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #f0f0f0;
}

.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    color: #1DB954;
    letter-spacing: -1px;
    line-height: 1.1;
}

.hero-sub {
    font-size: 1.15rem;
    color: #a0a0a0;
    margin-top: 0.5rem;
    font-weight: 300;
}

.card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}

.card:hover { border-color: #1DB954; }

.card-icon { font-size: 2rem; margin-bottom: 0.5rem; }

.card-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    color: #1DB954;
    margin-bottom: 0.3rem;
}

.card-desc { font-size: 0.88rem; color: #888; line-height: 1.5; }

.badge {
    display: inline-block;
    background: #1DB95422;
    color: #1DB954;
    border: 1px solid #1DB95455;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    margin: 3px;
}

.stat-box {
    background: #111;
    border: 1px solid #222;
    border-radius: 10px;
    padding: 1.2rem;
    text-align: center;
}

.stat-num {
    font-family: 'Space Mono', monospace;
    font-size: 2rem;
    color: #1DB954;
    font-weight: 700;
}

.stat-label { font-size: 0.8rem; color: #666; margin-top: 4px; }

hr.green { border: none; border-top: 1px solid #1DB95433; margin: 2rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
col_hero, col_img = st.columns([2, 1])
with col_hero:
    st.markdown('<div class="hero-title">Spotify ML<br>Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Explorare · Statistică · Machine Learning pe 114.000 piese Spotify</div>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)


st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Stats rapide ──────────────────────────────────────────────────────────────
st.markdown("###  Dataset la o privire")
c1, c2, c3, c4 = st.columns(4)
stats = [
    ("114.000", "Piese Spotify"),
    ("125", "Genuri muzicale"),
    ("20", "Coloane / Features"),
    ("6", "Modele ML aplicate"),
]
for col, (num, label) in zip([c1, c2, c3, c4], stats):
    with col:
        st.markdown(f"""
        <div class="stat-box">
            <div class="stat-num">{num}</div>
            <div class="stat-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Pagini ────────────────────────────────────────────────────────────────────
st.markdown("###  Structura dashboard-ului")

pages = [
    ("1 · Date & Statistici", "Încărcare dataset, statistici descriptive, missing values, tipuri de date, distribuții de bază."),
    ("2 · Filtrare & Explorare", "Filtrare interactivă după gen, artist, popularitate. Tabele sortabile și export CSV."),
    ("3 · Vizualizări Matplotlib", "Histograme, boxplots, scatter matrix. Grafice statice de calitate pentru analiză."),
    ("4 · Vizualizări Plotly", "Grafice interactive: scatter 3D, radar chart per gen, heatmap corelații, violin plots."),
    ("5 · Regresie Liniară", "Model OLS: popularity ~ features. Coeficienți, R², p-values, interpretare econometrică, residuals."),
    ("6 · Clasificare ML", "Random Forest pentru predicția genului. Confusion matrix, feature importance, cross-validation."),
    ("7 · Clustering K-Means", "Segmentare piese în clustere. Elbow method, PCA 2D, profilul fiecărui cluster."),
]

col1, col2 = st.columns(2)
for i, (title, desc) in enumerate(pages):
    col = col1 if i % 2 == 0 else col2
    with col:
        st.markdown(f"""
        <div class="card">
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; color:#444; font-size:0.8rem; font-family: Space Mono, monospace;'>
    Dataset: <a href='https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset' 
    style='color:#1DB954;'>Kaggle · Spotify Tracks Dataset</a> &nbsp;|&nbsp; 
    Built with Streamlit · Scikit-learn · Plotly
</div>
""", unsafe_allow_html=True)
