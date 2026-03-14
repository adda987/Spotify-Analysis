import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Clustering K-Means", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; background-color: #0d0d0d; color: #f0f0f0; }
.page-title { font-family: 'Space Mono', monospace; font-size: 2rem; color: #1DB954; font-weight: 700; }
.section-title { font-family: 'Space Mono', monospace; font-size: 1rem; color: #1DB954; margin: 1.5rem 0 0.5rem 0; }
.metric-green { color: #1DB954; font-family: 'Space Mono', monospace; font-size: 1.8rem; font-weight:700; }
.metric-label { color: #888; font-size: 0.8rem; }
.info-box { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; padding:1rem; }
.cluster-card { background:#1a1a1a; border:1px solid #1DB95433; border-radius:10px; padding:1.2rem; margin-bottom:0.8rem; }
.cluster-title { font-family:'Space Mono',monospace; color:#1DB954; font-size:1.1rem; font-weight:700; }
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

st.markdown('<div class="page-title"> Clustering K-Means</div>', unsafe_allow_html=True)
st.markdown("Elbow method, Silhouette Score, vizualizare PCA 2D.")
st.markdown('<hr class="green">', unsafe_allow_html=True)

try:
    df = load_data()
except FileNotFoundError:
    st.error("`dataset.csv` nu a fost găsit.")
    st.stop()

# ── Configurare ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">️ Configurare</div>', unsafe_allow_html=True)

cluster_features = ["energy", "danceability", "valence", "loudness", "tempo",
                     "acousticness", "speechiness", "instrumentalness", "liveness"]
cluster_features = [f for f in cluster_features if f in df.columns]

col1, col2 = st.columns(2)
with col1:
    selected_feats = st.multiselect("Variabile pentru clustering", cluster_features,
                                     default=["energy", "danceability", "valence", "acousticness", "tempo"])
    n_sample = st.slider("Sample size", 5000, 30000, 15000, step=5000)
with col2:
    k_clusters = st.slider("Număr clustere (K)", 2, 10, 5)
    show_elbow = st.checkbox("Calculează Elbow + Silhouette (K=2..10)", value=True)

if not selected_feats:
    st.warning("Selectează cel puțin 2 variabile.")
    st.stop()

df_sample = df[selected_feats + ["track_name", "artists", "track_genre", "popularity"]].dropna()
if len(df_sample) > n_sample:
    df_sample = df_sample.sample(n_sample, random_state=42)

X = df_sample[selected_feats].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── Elbow + Silhouette ────────────────────────────────────────────────────────
if show_elbow:
    st.markdown('<hr class="green">', unsafe_allow_html=True)
    st.markdown('<div class="section-title"> Elbow Method & Silhouette Score</div>', unsafe_allow_html=True)
    
    with st.spinner("Calculez WCSS și Silhouette pentru K=2..10..."):
        wcss = []
        sil_scores = []
        k_range = range(2, 11)
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=200)
            labels = km.fit_predict(X_scaled)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X_scaled, labels, sample_size=min(3000, len(X_scaled))))
    
    col_e1, col_e2 = st.columns(2)
    
    with col_e1:
        fig_elbow = go.Figure()
        fig_elbow.add_trace(go.Scatter(
            x=list(k_range), y=wcss, mode="lines+markers",
            line=dict(color="#1DB954", width=2), marker=dict(size=8, color="#1DB954"),
            name="WCSS"
        ))
        fig_elbow.add_vline(x=k_clusters, line_dash="dash", line_color="#FF6B6B",
                             annotation_text=f"K={k_clusters} selectat",
                             annotation_font_color="#FF6B6B")
        fig_elbow.update_layout(
            title="Elbow Method (WCSS vs K)",
            xaxis_title="K (număr clustere)", yaxis_title="WCSS (inerție)",
            template="plotly_dark", height=350, paper_bgcolor="#0d0d0d",
            font=dict(color="white"), xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222")
        )
        st.plotly_chart(fig_elbow, use_container_width=True)
    
    with col_e2:
        best_k = list(k_range)[np.argmax(sil_scores)]
        colors_sil = ["#1DB954" if k == best_k else "#4D96FF" for k in k_range]
        
        fig_sil = go.Figure()
        fig_sil.add_trace(go.Bar(
            x=list(k_range), y=sil_scores,
            marker_color=colors_sil,
            text=[f"{s:.3f}" for s in sil_scores],
            textposition="outside"
        ))
        fig_sil.update_layout(
            title=f"Silhouette Score (cel mai bun K={best_k} ↑)",
            xaxis_title="K", yaxis_title="Silhouette Score",
            template="plotly_dark", height=350, paper_bgcolor="#0d0d0d",
            font=dict(color="white"), xaxis=dict(gridcolor="#222"),
            yaxis=dict(range=[0, max(sil_scores) + 0.05], gridcolor="#222")
        )
        st.plotly_chart(fig_sil, use_container_width=True)
    
    st.info(f" **Elbow**: Caută genunchiul curbei WCSS. **Silhouette**: Valori mai mari = clustere mai bine separate. Optim sugerată: K={best_k}.")

# ── Antrenare KMeans ──────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown(f'<div class="section-title"> Rezultate Clustering cu K={k_clusters}</div>', unsafe_allow_html=True)

with st.spinner(f"Rulează K-Means cu K={k_clusters}..."):
    km_final = KMeans(n_clusters=k_clusters, random_state=42, n_init=10, max_iter=300)
    cluster_labels = km_final.fit_predict(X_scaled)
    df_sample = df_sample.copy()
    df_sample["cluster"] = cluster_labels.astype(str)
    sil_final = silhouette_score(X_scaled, cluster_labels, sample_size=min(3000, len(X_scaled)))

st.success(f"Silhouette Score final: **{sil_final:.4f}**")

# ── PCA 2D Scatter ────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">Vizualizare PCA 2D</div>', unsafe_allow_html=True)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
df_sample["PCA_1"] = X_pca[:, 0]
df_sample["PCA_2"] = X_pca[:, 1]

var_exp = pca.explained_variance_ratio_

fig_pca = px.scatter(
    df_sample, x="PCA_1", y="PCA_2", color="cluster",
    hover_data={"track_name": True, "artists": True, "track_genre": True,
                "popularity": True, "PCA_1": False, "PCA_2": False},
    opacity=0.6,
    template="plotly_dark",
    title=f"Clustere K-Means în spațiu PCA 2D (varianta explicată: PC1={var_exp[0]*100:.1f}%, PC2={var_exp[1]*100:.1f}%)",
    color_discrete_sequence=px.colors.qualitative.Set1
)
fig_pca.update_layout(
    height=550, paper_bgcolor="#0d0d0d",
    font=dict(color="white"), xaxis=dict(gridcolor="#222"), yaxis=dict(gridcolor="#222"),
    legend=dict(bgcolor="#1a1a1a", bordercolor="#333")
)
st.plotly_chart(fig_pca, use_container_width=True)

# ── Profilul clusterelor ──────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Profilul fiecărui cluster</div>', unsafe_allow_html=True)

cluster_profiles = df_sample.groupby("cluster")[selected_feats + ["popularity"]].mean().round(3)
cluster_sizes = df_sample["cluster"].value_counts().sort_index()
cluster_profiles["Dimensiune"] = cluster_sizes.values
cluster_profiles["% din total"] = (cluster_sizes.values / len(df_sample) * 100).round(1)

st.dataframe(cluster_profiles, use_container_width=True)

# ── Interpretare automată clustere ───────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Interpretarea clusterelor</div>', unsafe_allow_html=True)

cluster_means = df_sample.groupby("cluster")[selected_feats].mean()

cluster_names = {}
for cid in cluster_means.index:
    row = cluster_means.loc[cid]
    tags = []
    
    if "energy" in row and row["energy"] > 0.7: tags.append(" Energetic")
    elif "energy" in row and row["energy"] < 0.35: tags.append(" Calm")
    
    if "danceability" in row and row["danceability"] > 0.7: tags.append(" Dansabil")
    
    if "acousticness" in row and row["acousticness"] > 0.6: tags.append(" Acustic")
    
    if "valence" in row and row["valence"] > 0.65: tags.append(" Pozitiv")
    elif "valence" in row and row["valence"] < 0.35: tags.append(" Melancolic")
    
    if "instrumentalness" in row and row["instrumentalness"] > 0.4: tags.append(" Instrumental")
    
    if "speechiness" in row and row["speechiness"] > 0.15: tags.append(" Spoken word")
    
    if not tags: tags.append(" Mixtă")
    
    cluster_names[cid] = " · ".join(tags)

top_genre_per_cluster = df_sample.groupby("cluster")["track_genre"].agg(lambda x: x.value_counts().index[0])

for cid in sorted(df_sample["cluster"].unique(), key=lambda x: int(x)):
    size = cluster_sizes[cid]
    pct = size / len(df_sample) * 100
    name = cluster_names.get(cid, "Mixtă")
    top_genre = top_genre_per_cluster.get(cid, "—")
    pop_mean = df_sample[df_sample["cluster"] == cid]["popularity"].mean()
    
    st.markdown(f"""
    <div class="cluster-card">
        <div class="cluster-title">Cluster {cid} — {name}</div>
        <div style='color:#888; font-size:0.85rem; margin-top:4px'>
             {size:,} piese ({pct:.1f}%) &nbsp;|&nbsp; 
             Gen dominant: <b style='color:white'>{top_genre}</b> &nbsp;|&nbsp; 
             Popularitate medie: <b style='color:#1DB954'>{pop_mean:.1f}/100</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Radar clusters ────────────────────────────────────────────────────────────
st.markdown('<hr class="green">', unsafe_allow_html=True)
st.markdown('<div class="section-title"> Radar Chart: Comparație Clustere</div>', unsafe_allow_html=True)

radar_feats_cl = [f for f in ["energy", "danceability", "valence", "acousticness", "speechiness"] if f in selected_feats]

if len(radar_feats_cl) >= 3:
    fig_radar = go.Figure()
    colors_r = px.colors.qualitative.Set1
    
    for i, cid in enumerate(sorted(df_sample["cluster"].unique(), key=lambda x: int(x))):
        vals = cluster_means.loc[cid, radar_feats_cl].tolist()
        vals += [vals[0]]
        
        fig_radar.add_trace(go.Scatterpolar(
            r=vals, theta=radar_feats_cl + [radar_feats_cl[0]],
            fill="toself", name=f"Cluster {cid}",
            line_color=colors_r[i % len(colors_r)],
            fillcolor=colors_r[i % len(colors_r)],
            opacity=0.3
        ))
    
    fig_radar.update_layout(
        polar=dict(
            bgcolor="#1a1a1a",
            radialaxis=dict(visible=True, range=[0, 1], color="#666", gridcolor="#333"),
            angularaxis=dict(color="white", gridcolor="#333")
        ),
        template="plotly_dark", height=500, paper_bgcolor="#0d0d0d",
        font=dict(color="white"), showlegend=True,
        title="Profilul Audio Mediu per Cluster",
        legend=dict(bgcolor="#1a1a1a", bordercolor="#333")
    )
    st.plotly_chart(fig_radar, use_container_width=True)
