import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm

st.set_page_config(page_title="Vizualizări Plotly", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.page-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1DB954; font-weight: 700; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #1DB954; margin: 1.5rem 0 0.5rem 0; }
hr.green { border: none; border-top: 1px solid #1DB95433; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

PLOTLY_TEMPLATE = "plotly_dark"

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=["track_name", "artists", "track_genre"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

st.markdown('<div class="page-title"> Vizualizări Interactive Plotly</div>', unsafe_allow_html=True)
st.markdown("Grafice interactive. Relațiile dintre variabilele audio.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error(" `dataset.csv` nu a fost găsit.")
    st.stop()

# ── 1. Scatter 3D ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">1. Scatter 3D: Energy × Danceability × Valence</div>', unsafe_allow_html=True)
st.markdown("Hover pe puncte pentru a vedea titlul piesei, artistul și popularitatea.")

genres_3d = st.multiselect("Genuri pentru 3D scatter", sorted(df["track_genre"].unique()),
                            default=["pop", "rock", "classical", "hip-hop", "jazz", "electronic"])
n_sample = st.slider("Număr puncte (sample)", 500, 5000, 2000, step=500)

if genres_3d:
    df3d = df[df["track_genre"].isin(genres_3d)].sample(min(n_sample, len(df[df["track_genre"].isin(genres_3d)])), random_state=42)
    
    fig1 = px.scatter_3d(
        df3d, x="energy", y="danceability", z="valence",
        color="track_genre", size="popularity",
        hover_data={"track_name": True, "artists": True, "popularity": True,
                    "energy": ":.2f", "danceability": ":.2f", "valence": ":.2f"},
        opacity=0.7, template=PLOTLY_TEMPLATE,
        title="Spațiu 3D: Energy × Danceability × Valence (mărimea = popularitate)",
        color_discrete_sequence=px.colors.qualitative.Set1
    )
    fig1.update_layout(
        height=600, paper_bgcolor="#0d0d0d", plot_bgcolor="#0d0d0d",
        font=dict(color="white"),
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333")
    )
    st.plotly_chart(fig1, use_container_width=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 2. Radar Chart per gen ────────────────────────────────────────────────────
st.markdown('<div class="section-title">️ 2. Radar Chart: Profilul audio per gen</div>', unsafe_allow_html=True)

radar_genres = st.multiselect("Selectează maxim 6 genuri",
                               sorted(df["track_genre"].unique()),
                               default=["pop", "rock", "classical", "hip-hop", "jazz"])

radar_feats = ["energy", "danceability", "valence", "acousticness", "speechiness", "liveness"]

if radar_genres:
    fig2 = go.Figure()
    colors_radar = ["#1DB954", "#FF6B6B", "#FFD93D", "#4D96FF", "#CC5DE8", "#FF922B"]
    
    for i, genre in enumerate(radar_genres[:6]):
        vals = df[df["track_genre"] == genre][radar_feats].mean().tolist()
        vals += [vals[0]]  # Închide poligonul
        
        fig2.add_trace(go.Scatterpolar(
            r=vals,
            theta=radar_feats + [radar_feats[0]],
            fill="toself",
            name=genre,
            line_color=colors_radar[i % len(colors_radar)],
            fillcolor=colors_radar[i % len(colors_radar)],
            opacity=0.35
        ))
    
    fig2.update_layout(
        polar=dict(
            bgcolor="#1a1a1a",
            radialaxis=dict(visible=True, range=[0, 1], color="#666", gridcolor="#333"),
            angularaxis=dict(color="white", gridcolor="#333")
        ),
        showlegend=True,
        template=PLOTLY_TEMPLATE,
        height=500,
        paper_bgcolor="#0d0d0d",
        font=dict(color="white"),
        title="Profil Audio Mediu per Gen — Radar Chart",
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333")
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 3. Violin plots ───────────────────────────────────────────────────────────
st.markdown('<div class="section-title"> 3. Violin Plot: distribuție per gen</div>', unsafe_allow_html=True)

violin_feat = st.selectbox("Variabilă pentru violin plot",
                            ["popularity", "energy", "danceability", "valence", "tempo", "loudness"])
top10_genres = df["track_genre"].value_counts().head(10).index.tolist()
df_violin = df[df["track_genre"].isin(top10_genres)]

fig3 = px.violin(df_violin, x="track_genre", y=violin_feat, color="track_genre",
                  box=True, points=False,
                  template=PLOTLY_TEMPLATE,
                  title=f"Distribuția '{violin_feat}' per Gen (Top 10 genuri)",
                  color_discrete_sequence=px.colors.qualitative.Set2)
fig3.update_layout(
    height=500, paper_bgcolor="#0d0d0d", showlegend=False,
    font=dict(color="white"),
    xaxis=dict(tickangle=30, gridcolor="#222"),
    yaxis=dict(gridcolor="#222")
)
st.plotly_chart(fig3, use_container_width=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 4. Scatter interactiv cu linie trend ──────────────────────────────────────
st.markdown('<div class="section-title"> 4. Scatter interactiv cu trend OLS</div>', unsafe_allow_html=True)

col_x = st.selectbox("Axa X", ["energy", "danceability", "loudness", "tempo", "valence", "acousticness"], index=0)
col_y = st.selectbox("Axa Y", ["popularity", "energy", "danceability", "valence"], index=0)
color_by = st.selectbox("Colorează după", ["track_genre", "explicit"])

df_sc = df.sample(min(3000, len(df)), random_state=42)

fig4 = px.scatter(df_sc, x=col_x, y=col_y, color=color_by,
                   hover_data=["track_name", "artists", "popularity"],
                   trendline="ols",
                   template=PLOTLY_TEMPLATE,
                   title=f"{col_x.capitalize()} vs {col_y.capitalize()} (cu linie OLS)",
                   opacity=0.6,
                   color_discrete_sequence=px.colors.qualitative.Vivid)
fig4.update_layout(
    height=500, paper_bgcolor="#0d0d0d",
    font=dict(color="white"),
    xaxis=dict(gridcolor="#222"),
    yaxis=dict(gridcolor="#222"),
    legend=dict(bgcolor="#1a1a1a", bordercolor="#333")
)
st.plotly_chart(fig4, use_container_width=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 5. Sunburst: Gen → Explicit ───────────────────────────────────────────────
st.markdown('<div class="section-title"> 5. Sunburst: Gen → Conținut Explicit</div>', unsafe_allow_html=True)

top15 = df["track_genre"].value_counts().head(15).index.tolist()
df_sun = df[df["track_genre"].isin(top15)].copy()
df_sun["explicit_label"] = df_sun["explicit"].map({True: "Explicit", False: "Clean"})

sun_data = df_sun.groupby(["track_genre", "explicit_label"]).size().reset_index(name="count")

fig5 = px.sunburst(sun_data, path=["track_genre", "explicit_label"], values="count",
                    template=PLOTLY_TEMPLATE,
                    title="Distribuție Gen → Conținut Explicit",
                    color_discrete_sequence=px.colors.qualitative.Set3)
fig5.update_layout(height=550, paper_bgcolor="#0d0d0d", font=dict(color="white"))
st.plotly_chart(fig5, use_container_width=True)

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 6. Top artiști ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title"> 6. Top artiști după popularitate medie</div>', unsafe_allow_html=True)

n_artists = st.slider("Număr artiști afișați", 10, 30, 15)
top_artists = (df.groupby("artists")["popularity"]
               .agg(["mean", "count"])
               .reset_index()
               .rename(columns={"mean": "Popularitate medie", "count": "Nr. piese"})
               .query("`Nr. piese` >= 3")  # Cel puțin 3 piese
               .sort_values("Popularitate medie", ascending=False)
               .head(n_artists))

fig6 = px.bar(top_artists, x="Popularitate medie", y="artists",
              orientation="h", color="Popularitate medie",
              color_continuous_scale=[[0, "#0d4d1f"], [0.5, "#1DB954"], [1, "#7fff9a"]],
              hover_data=["Nr. piese"],
              template=PLOTLY_TEMPLATE,
              title=f"Top {n_artists} Artiști după Popularitate Medie (min. 3 piese)")
fig6.update_layout(
    height=500, paper_bgcolor="#0d0d0d", showlegend=False,
    font=dict(color="white"),
    yaxis=dict(autorange="reversed", gridcolor="#222"),
    xaxis=dict(gridcolor="#222")
)
st.plotly_chart(fig6, use_container_width=True)
