# 🎵 Spotify Analysis


Un dashboard interactiv construit cu **Streamlit** care aplică tehnici de analiză a datelor și machine learning pe un dataset de piese Spotify — de la statistici descriptive până la regresie, clasificare și clustering.

---

## 📁 Structura proiectului

```
files/
├── Home.py                          # Pagina principală
├── dataset.csv                      # Setul de date
├── requirements.txt                 # Dependențe Python
└── pages/
    ├── 1_Date_si_Statistici.py      # Statistici descriptive
    ├── 2_Filtrare_si_Explorare.py   # Filtrare interactivă
    ├── 3_Vizualizari_Matplotlib.py  # Grafice Matplotlib
    ├── 4_Vizualizari_Plotly.py      # Grafice Plotly interactive
    ├── 5_Regresie_Liniara.py        # Model de regresie liniară
    ├── 6_Clasificare_RandomForest.py# Model de clasificare
    └── 7_Clustering_KMeans.py       # Clustering K-Means
```

---

##  Instructiuni de rulare:

### 1. Clonează repo
```bash
git clone https://github.com/numetau/spotify-analysis.git
cd spotify-analysis
```

### 3. Instalează dependențele
```bash
pip install -r requirements.txt
```

### 4. Rulează aplicația
```bash
streamlit run Home.py
```

---

##  Despre date

Setul de date conține **114.000 de piese Spotify**, distribuite uniform în **125 de genuri muzicale** (câte 1.000 de piese per gen), extras de pe [Kaggle — Spotify Tracks Dataset](https://www.kaggle.com/datasets/maharshipandya/-spotify-tracks-dataset). Fiecare piesă este descrisă prin **20 de coloane**:

| Tip | Coloane |
|-----|---------|
| **Caracteristici audio** | `energy`, `danceability`, `valence`, `tempo`, `loudness`, `acousticness`, `speechiness`, `instrumentalness`, `liveness` |
| **Metadate** | `track_name`, `artists`, `album_name`, `track_genre`, `duration_ms`, `explicit` |
| **Variabila țintă** | `popularity` (scor 0–100, medie ~33/100) |

---

## Rezultate

###  1. Date & Statistici
Încărcare dataset, statistici descriptive (medie, mediană, deviație standard), analiza valorilor lipsă, tipuri de date și distribuții de bază pentru fiecare variabilă numerică.

###  2. Filtrare & Explorare
Filtrare interactivă după gen muzical, artist și interval de popularitate. Tabele sortabile cu opțiune de export CSV — util pentru explorarea rapidă a subseturilor de date.

###  3. Vizualizări Matplotlib
Histograme, boxplots și scatter matrix pentru analiza distribuțiilor și a relațiilor dintre variabile. Grafice statice de calitate pentru raportare.

###  4. Vizualizări Plotly
Grafice interactive: scatter 3D, radar chart comparativ per gen muzical, heatmap de corelații și violin plots — cu hover, zoom și filtrare dinamică.

###  5. Regresie Liniară (OLS)
**Obiectiv:** predicția popularității pe baza caracteristicilor audio.

**Rezultate cheie:**
- R² scăzut (~0.05–0.10) — un rezultat valoros în sine: demonstrează că popularitatea nu poate fi prezisă doar din caracteristicile audio. Factorii de marketing, viralitate și context social au un rol dominant.
- Coeficienți pozitivi asociați cu `danceability` și `energy`
- Coeficienți negativi pentru `acousticness` și `instrumentalness`
- Analiza reziduurilor confirmă ipotezele Gauss-Markov (medie ≈ 0, distribuție aproximativ normală)
- Predictor interactiv: introdu valori custom și estimează popularitatea unei piese

###  6. Clasificare — Random Forest
**Obiectiv:** predicția genului muzical pe baza caracteristicilor audio.

**Rezultate cheie:**
- Acuratețe ~60–70% pe top 8 genuri (de ~6–8x mai bun față de clasificarea aleatoare)
- Cel mai important feature: `acousticness`, urmat de `instrumentalness` și `energy`
- Cross-validation 5-fold stabil, fără overfitting semnificativ
- Confusion matrix normalizată și raport complet per gen (Precision, Recall, F1-Score)

###  7. Clustering K-Means
**Obiectiv:** segmentarea nesupervizată a pieselor în grupuri cu profil audio similar.

**Rezultate cheie:**
- Elbow method + Silhouette Score sugerează un optim la **K=4–5 clustere**
- Vizualizare PCA 2D cu separare clară între clustere
- Profiluri identificate automat:

| Cluster | Profil | Genuri dominante |
|---------|--------|-----------------|
| Energetic | Energy și tempo ridicate, acousticness scăzută | Rock, Metal |
| Acustic & Calm | Acousticness ridicată, energy scăzută | Folk, Classical |
| Dansabil & Pozitiv | Danceability și valence ridicate | Pop, Latin |
| Spoken Word | Speechiness ridicată | Comedy, Spoken |

---


Andreea S.
