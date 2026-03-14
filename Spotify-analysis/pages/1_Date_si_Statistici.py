import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title="Date & Statistici", layout="wide")

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

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    # Curățare de bază
    df = df.drop_duplicates()
    df = df.dropna(subset=["track_name", "artists", "track_genre"])
    # Eliminăm coloana unnamed dacă există
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

st.markdown('<div class="page-title"> Date & Statistici Descriptive</div>', unsafe_allow_html=True)
st.markdown("Explorare inițială a dataset-ului: structură, valori lipsă, distribuții și statistici.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error(" Fișierul `dataset.csv` nu a fost găsit.")
    st.info("Link dataset: https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset")
    st.stop()

# ── Metrici rapide ────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
metrics = [
    (f"{len(df):,}", "Piese totale"),
    (f"{df['track_genre'].nunique()}", "Genuri"),
    (f"{df['artists'].nunique():,}", "Artiști unici"),
    (f"{df.isnull().sum().sum()}", "Valori lipsă"),
    (f"{df.shape[1]}", "Coloane"),
]
for col, (val, label) in zip([c1, c2, c3, c4, c5], metrics):
    with col:
        st.markdown(f"""<div class="info-box" style="text-align:center">
        <div class="metric-green">{val}</div>
        <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Preview dataset ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Preview dataset</div>', unsafe_allow_html=True)
n_rows = st.slider("Număr rânduri afișate", 5, 50, 10)
st.dataframe(df.head(n_rows), use_container_width=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Tipuri de date ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Tipuri de date per coloană</div>', unsafe_allow_html=True)
col_left, col_right = st.columns(2)

dtype_df = pd.DataFrame({
    "Coloană": df.dtypes.index,
    "Tip": df.dtypes.values.astype(str),
    "Non-null": df.count().values,
    "Null": df.isnull().sum().values,
    "Unice": [df[c].nunique() for c in df.columns]
})
with col_left:
    st.dataframe(dtype_df, use_container_width=True, hide_index=True)

with col_right:
    st.markdown("**Descrierea coloanelor numerice cheie:**")
    col_descriptions = {
        "popularity": "Scor 0–100. Cu cât mai mare, cu atât mai populară piesa pe Spotify.",
        "danceability": "0–1. Cât de potrivită e piesa pentru dans.",
        "energy": "0–1. Intensitate și activitate perceptuală.",
        "loudness": "dB. Volumul mediu. De obicei între -60 și 0.",
        "valence": "0–1. Pozitivitate muzicală (1 = veselă, 0 = tristă).",
        "tempo": "BPM — bătăi pe minut.",
        "acousticness": "0–1. Probabilitate că piesa e acustică.",
        "speechiness": "0–1. Prezența cuvintelor vorbite.",
        "instrumentalness": "0–1. Probabilitate că nu are voce.",
        "liveness": "0–1. Probabilitate că e înregistrată live.",
        "duration_ms": "Durata piesei în milisecunde.",
        "explicit": "1 dacă are conținut explicit, 0 dacă nu.",
        "key": "Tonalitatea piesei (0=C, 1=C#, ..., 11=B).",
        "mode": "Mod: 1=major, 0=minor.",
        "time_signature": "Metru: câte bătăi per măsură."
    }
    for col_name, desc in col_descriptions.items():
        if col_name in df.columns:
            st.markdown(f"- **`{col_name}`**: {desc}")

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Statistici descriptive ────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Statistici descriptive (coloane numerice)</div>', unsafe_allow_html=True)

num_cols = df.select_dtypes(include=np.number).columns.tolist()
desc = df[num_cols].describe().T.round(3)
desc.columns = ["Count", "Medie", "Std Dev", "Min", "Q1 (25%)", "Mediană (50%)", "Q3 (75%)", "Max"]
st.dataframe(desc, use_container_width=True)

st.markdown("""
>  **Interpretare econometrică**: Std Dev mare la `loudness` (-60 → 0 dB) și `duration_ms` 
> indică variabilitate ridicată — aceste variabile vor necesita normalizare înainte de ML.
> `popularity` cu medie ~33/100 sugerează o distribuție asimetrică spre dreapta.
""")

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Top 10 genuri ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title"> Top 10 genuri după număr de piese</div>', unsafe_allow_html=True)
genre_counts = df["track_genre"].value_counts().head(10).reset_index()
genre_counts.columns = ["Gen", "Număr piese"]
genre_counts["% din total"] = (genre_counts["Număr piese"] / len(df) * 100).round(2).astype(str) + "%"
st.dataframe(genre_counts, use_container_width=True, hide_index=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── Cele mai populare piese ───────────────────────────────────────────────────
st.markdown('<div class="section-title"> Top 15 cele mai populare piese</div>', unsafe_allow_html=True)
top_songs = df.nlargest(15, "popularity")[["track_name", "artists", "track_genre", "popularity"]].reset_index(drop=True)
top_songs.index += 1
st.dataframe(top_songs, use_container_width=True)
