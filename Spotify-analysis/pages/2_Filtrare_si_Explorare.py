import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Filtrare & Explorare", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.page-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1DB954; font-weight: 700; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #1DB954; margin: 1.5rem 0 0.5rem 0; }
.info-box { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:1rem; margin-bottom:0.8rem; }
.metric-green { color: #1DB954; font-family: 'Space Mono', monospace; font-size: 1.6rem; font-weight:700; }
.metric-label { color: #888; font-size: 0.8rem; }
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
    df["duration_min"] = (df["duration_ms"] / 60000).round(2)
    return df

st.markdown('<div class="page-title"> Filtrare & Explorare</div>', unsafe_allow_html=True)
st.markdown("Explorează interactiv dataset-ul. Filtrează după gen, popularitate, energie și alți parametri.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error(" `dataset.csv` nu a fost găsit în folderul proiectului.")
    st.stop()

# ── Sidebar Filtre ────────────────────────────────────────────────────────────
st.sidebar.markdown("##  Filtre")

# Filtrare gen
all_genres = sorted(df["track_genre"].unique().tolist())
selected_genres = st.sidebar.multiselect(
    "Genuri muzicale",
    options=all_genres,
    default=["pop", "rock", "jazz", "classical", "hip-hop"][:min(5, len(all_genres))]
)

# Filtrare popularitate
pop_range = st.sidebar.slider("Popularitate (0–100)", 0, 100, (30, 100))

# Filtrare energy
energy_range = st.sidebar.slider("Energy (0–1)", 0.0, 1.0, (0.0, 1.0), step=0.05)

# Filtrare danceability
dance_range = st.sidebar.slider("Danceability (0–1)", 0.0, 1.0, (0.0, 1.0), step=0.05)

# Filtrare explicit
explicit_opt = st.sidebar.radio("Conținut explicit", ["Toate", "Doar explicit", "Fără explicit"])

# Căutare artist
search_artist = st.sidebar.text_input(" Caută artist")

# ── Aplicare filtre ───────────────────────────────────────────────────────────
filtered = df.copy()

if selected_genres:
    filtered = filtered[filtered["track_genre"].isin(selected_genres)]

filtered = filtered[
    (filtered["popularity"] >= pop_range[0]) &
    (filtered["popularity"] <= pop_range[1])
]
filtered = filtered[
    (filtered["energy"] >= energy_range[0]) &
    (filtered["energy"] <= energy_range[1])
]
filtered = filtered[
    (filtered["danceability"] >= dance_range[0]) &
    (filtered["danceability"] <= dance_range[1])
]
if explicit_opt == "Doar explicit":
    filtered = filtered[filtered["explicit"] == True]
elif explicit_opt == "Fără explicit":
    filtered = filtered[filtered["explicit"] == False]

if search_artist:
    filtered = filtered[filtered["artists"].str.contains(search_artist, case=False, na=False)]

# ── Metrici rezultat ──────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
metrics = [
    (f"{len(filtered):,}", "Piese filtrate"),
    (f"{filtered['track_genre'].nunique()}", "Genuri selectate"),
    (f"{filtered['popularity'].mean():.1f}" if len(filtered) > 0 else "—", "Popularitate medie"),
    (f"{filtered['energy'].mean():.2f}" if len(filtered) > 0 else "—", "Energy medie"),
]
for col, (val, label) in zip([c1, c2, c3, c4], metrics):
    with col:
        st.markdown(f"""<div class="info-box" style="text-align:center">
        <div class="metric-green">{val}</div>
        <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

if len(filtered) == 0:
    st.warning("️ Niciun rezultat pentru filtrele selectate.")
    st.stop()

# ── Tabel rezultate ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Rezultate filtrate</div>', unsafe_allow_html=True)

sort_col = st.selectbox("Sortează după", ["popularity", "energy", "danceability", "tempo", "valence", "duration_min"])
sort_dir = st.radio("Direcție", ["Descrescător", "Crescător"], horizontal=True)
ascending = sort_dir == "Crescător"

display_cols = ["track_name", "artists", "track_genre", "popularity", "energy",
                "danceability", "valence", "tempo", "duration_min", "explicit"]
display_cols = [c for c in display_cols if c in filtered.columns]

result_sorted = filtered[display_cols].sort_values(sort_col, ascending=ascending).reset_index(drop=True)
result_sorted.index += 1

st.dataframe(result_sorted, use_container_width=True)

# ── Export CSV ────────────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Export date filtrate</div>', unsafe_allow_html=True)

csv_export = filtered.to_csv(index=False).encode("utf-8")
st.download_button(
    label=f" Descarcă {len(filtered):,} piese ca CSV",
    data=csv_export,
    file_name="spotify_filtrat.csv",
    mime="text/csv"
)

# ── Statistici per gen selectat ───────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Statistici medii per gen (din selecție)</div>', unsafe_allow_html=True)

numeric_feats = ["popularity", "energy", "danceability", "valence", "acousticness", "tempo", "loudness"]
numeric_feats = [f for f in numeric_feats if f in filtered.columns]

genre_stats = filtered.groupby("track_genre")[numeric_feats].mean().round(3).reset_index()
genre_stats = genre_stats.sort_values("popularity", ascending=False)
st.dataframe(genre_stats, use_container_width=True, hide_index=True)

# ── Piesa cea mai reprezentativă per gen ─────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title">Cea mai populară piesă per gen </div>', unsafe_allow_html=True)

top_per_genre = (
    filtered.sort_values("popularity", ascending=False)
    .groupby("track_genre")
    .first()
    .reset_index()
    [["track_genre", "track_name", "artists", "popularity", "energy", "danceability"]]
    .sort_values("popularity", ascending=False)
)
st.dataframe(top_per_genre, use_container_width=True, hide_index=True)
