import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import rcParams
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Vizualizări Matplotlib", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.page-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1DB954; font-weight: 700; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #1DB954; margin: 1.5rem 0 0.5rem 0; }
hr.green { border: none; border-top: 1px solid #1DB95433; margin: 1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Matplotlib dark theme ─────────────────────────────────────────────────────
plt.style.use("dark_background")
rcParams["font.family"] = "monospace"
SPOTIFY_GREEN = "#1DB954"
ACCENT = "#FF6B6B"
BG = "#0d0d0d"
PANEL = "#1a1a1a"

@st.cache_data
def load_data():
    df = pd.read_csv("dataset.csv")
    df = df.drop_duplicates()
    df = df.dropna(subset=["track_name", "artists", "track_genre"])
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

st.markdown('<div class="page-title">Vizualizări Matplotlib</div>', unsafe_allow_html=True)
st.markdown("Grafice statice de analiză: distribuții, corelații, comparații între genuri.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("`dataset.csv` nu a fost găsit.")
    st.stop()

# ── 1. Histograme distribuții ─────────────────────────────────────────────────
st.markdown('<div class="section-title"> 1. Distribuția variabilelor audio</div>', unsafe_allow_html=True)

features = ["popularity", "energy", "danceability", "valence", "acousticness", "loudness", "tempo", "speechiness"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7), facecolor=BG)
fig.suptitle("Distribuția Variabilelor Audio — Spotify 114k Tracks", color="white", fontsize=13, y=1.01)

colors = [SPOTIFY_GREEN, "#FF6B6B", "#FFD93D", "#6BCB77", "#4D96FF", "#FF922B", "#CC5DE8", "#20C997"]

for ax, feat, color in zip(axes.flat, features, colors):
    ax.set_facecolor(PANEL)
    ax.hist(df[feat].dropna(), bins=50, color=color, alpha=0.85, edgecolor="none")
    mean_val = df[feat].mean()
    ax.axvline(mean_val, color="white", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.set_title(feat, color="white", fontsize=10, pad=6)
    ax.set_xlabel("")
    ax.tick_params(colors="#888", labelsize=8)
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")
    ax.text(0.97, 0.95, f"μ={mean_val:.2f}", transform=ax.transAxes,
            ha="right", va="top", color="white", fontsize=8, alpha=0.8)

plt.tight_layout()
st.pyplot(fig, use_container_width=True)
plt.close()

st.markdown("""
>  **Observații**: `popularity` are o distribuție bimodală — multe piese cu popularitate 0 (obscure) și un vârf în zona 40-60.
> `energy` și `danceability` sunt aproximativ normale. `loudness` este asimetric stânga (skewed left).
""")

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 2. Boxplots per gen ───────────────────────────────────────────────────────
st.markdown('<div class="section-title"> 2. Boxplot: Popularitate per gen (Top 12)</div>', unsafe_allow_html=True)

top_genres = df["track_genre"].value_counts().head(12).index.tolist()
df_top = df[df["track_genre"].isin(top_genres)]

fig2, ax2 = plt.subplots(figsize=(16, 6), facecolor=BG)
ax2.set_facecolor(PANEL)

data_by_genre = [df_top[df_top["track_genre"] == g]["popularity"].dropna().values for g in top_genres]
bp = ax2.boxplot(data_by_genre, patch_artist=True, notch=False,
                  medianprops=dict(color="white", linewidth=2),
                  whiskerprops=dict(color="#888"),
                  capprops=dict(color="#888"),
                  flierprops=dict(marker="o", color=SPOTIFY_GREEN, alpha=0.3, markersize=3))

for patch, color in zip(bp["boxes"], plt.cm.Set2(np.linspace(0, 1, len(top_genres)))):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)

ax2.set_xticks(range(1, len(top_genres) + 1))
ax2.set_xticklabels(top_genres, rotation=35, ha="right", color="white", fontsize=10)
ax2.set_ylabel("Popularitate", color="white")
ax2.set_title("Distribuția Popularității per Gen Muzical", color="white", fontsize=12)
ax2.tick_params(colors="#888")
for spine in ax2.spines.values():
    spine.set_edgecolor("#333")
ax2.yaxis.label.set_color("white")

plt.tight_layout()
st.pyplot(fig2, use_container_width=True)
plt.close()

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 3. Scatter: Energy vs Popularity ─────────────────────────────────────────
st.markdown('<div class="section-title"> 3. Scatter: Energy vs Popularity (sample 3000)</div>', unsafe_allow_html=True)

selected_for_scatter = st.multiselect(
    "Selectează genuri pentru scatter",
    options=sorted(df["track_genre"].unique()),
    default=["pop", "rock", "classical", "hip-hop", "jazz"]
)

if selected_for_scatter:
    df_scatter = df[df["track_genre"].isin(selected_for_scatter)].sample(min(3000, len(df)), random_state=42)
    
    fig3, ax3 = plt.subplots(figsize=(10, 6), facecolor=BG)
    ax3.set_facecolor(PANEL)
    
    cmap = plt.cm.Set1
    genres_unique = df_scatter["track_genre"].unique()
    colors_map = {g: cmap(i / len(genres_unique)) for i, g in enumerate(genres_unique)}
    
    for genre in genres_unique:
        mask = df_scatter["track_genre"] == genre
        ax3.scatter(df_scatter[mask]["energy"], df_scatter[mask]["popularity"],
                    c=[colors_map[genre]], alpha=0.5, s=15, label=genre)
    
    # Linie de trend globală
    x = df_scatter["energy"].values
    y = df_scatter["popularity"].values
    mask_valid = ~(np.isnan(x) | np.isnan(y))
    z = np.polyfit(x[mask_valid], y[mask_valid], 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax3.plot(x_line, p(x_line), "--", color="white", linewidth=1.5, alpha=0.7, label=f"Trend (slope={z[0]:.2f})")
    
    ax3.set_xlabel("Energy", color="white")
    ax3.set_ylabel("Popularity", color="white")
    ax3.set_title("Energy vs Popularity", color="white", fontsize=12)
    ax3.legend(fontsize=8, framealpha=0.2, labelcolor="white")
    ax3.tick_params(colors="#888")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#333")
    
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 4. Heatmap corelații ──────────────────────────────────────────────────────
st.markdown('<div class="section-title">️ 4. Heatmap corelații (Pearson)</div>', unsafe_allow_html=True)

num_feats = ["popularity", "energy", "danceability", "valence", "acousticness",
             "loudness", "tempo", "speechiness", "instrumentalness", "liveness"]
num_feats = [f for f in num_feats if f in df.columns]
corr_matrix = df[num_feats].corr()

fig4, ax4 = plt.subplots(figsize=(11, 8), facecolor=BG)
ax4.set_facecolor(PANEL)

from matplotlib.colors import LinearSegmentedColormap
cmap_div = LinearSegmentedColormap.from_list(
    "spotify_corr",
    ["#FF6B6B", "#1a1a1a", "#1DB954"],
    N=256
)

im = ax4.imshow(corr_matrix.values, cmap=cmap_div, vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04).ax.yaxis.set_tick_params(color="white", labelcolor="white")

ax4.set_xticks(range(len(num_feats)))
ax4.set_yticks(range(len(num_feats)))
ax4.set_xticklabels(num_feats, rotation=45, ha="right", color="white", fontsize=9)
ax4.set_yticklabels(num_feats, color="white", fontsize=9)

for i in range(len(num_feats)):
    for j in range(len(num_feats)):
        val = corr_matrix.values[i, j]
        text_color = "black" if abs(val) > 0.4 else "white"
        ax4.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=text_color)

ax4.set_title("Matricea de Corelații — Variabile Audio Spotify", color="white", fontsize=12, pad=12)
plt.tight_layout()
st.pyplot(fig4, use_container_width=True)
plt.close()

st.markdown("""
>  **Interpretare**: `energy` și `loudness` sunt puternic corelate pozitiv (≈0.76) — piesele energice tind să fie mai tari.
> `energy` și `acousticness` sunt corelate negativ (≈-0.72) — piesele acustice sunt mai liniștite.
> `popularity` are corelații slabe cu toate variabilele audio — deci popularitatea e greu de explicat doar prin audio features.
""")

st.markdown('<hr class="green">', unsafe_allow_html=True)

# ── 5. Bar chart medii per gen ────────────────────────────────────────────────
st.markdown('<div class="section-title"> 5. Profilul audio mediu per gen (Top 8)</div>', unsafe_allow_html=True)

feat_sel = st.multiselect("Features pentru comparație", 
                           ["energy", "danceability", "valence", "acousticness", "speechiness"],
                           default=["energy", "danceability", "valence", "acousticness"])

top8 = df["track_genre"].value_counts().head(8).index.tolist()
df8 = df[df["track_genre"].isin(top8)]

if feat_sel:
    genre_means = df8.groupby("track_genre")[feat_sel].mean()
    
    fig5, ax5 = plt.subplots(figsize=(14, 6), facecolor=BG)
    ax5.set_facecolor(PANEL)
    
    x = np.arange(len(top8))
    width = 0.8 / len(feat_sel)
    colors5 = [SPOTIFY_GREEN, "#FF6B6B", "#FFD93D", "#4D96FF", "#CC5DE8"]
    
    for i, (feat, color) in enumerate(zip(feat_sel, colors5)):
        offset = (i - len(feat_sel)/2 + 0.5) * width
        bars = ax5.bar(x + offset, genre_means[feat], width * 0.9, label=feat, color=color, alpha=0.85)
    
    ax5.set_xticks(x)
    ax5.set_xticklabels(top8, rotation=30, ha="right", color="white", fontsize=10)
    ax5.set_ylabel("Valoare medie (0–1)", color="white")
    ax5.set_title("Profilul Audio Mediu per Gen Muzical", color="white", fontsize=12)
    ax5.legend(fontsize=9, framealpha=0.2, labelcolor="white")
    ax5.tick_params(colors="#888")
    ax5.set_ylim(0, 1)
    for spine in ax5.spines.values():
        spine.set_edgecolor("#333")
    
    plt.tight_layout()
    st.pyplot(fig5, use_container_width=True)
    plt.close()
