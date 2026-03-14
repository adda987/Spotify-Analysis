##  Spotify analysis Project

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
    ├── 5_Regresie_Liniara.py        # Model de regresie
    ├── 6_Clasificare_RandomForest.py# Model de clasificare
    └── 7_Clustering_KMeans.py       # Clustering K-Means
```

---

##  Cum rulezi proiectul

### 1. Clonează repository-ul
```bash
git clone https://github.com/numetau/spotify-analysis.git
cd spotify-analysis
```

### 2. Creează un virtual environment (opțional, recomandat)
```bash
python -m venv .venv
source .venv/bin/activate        # Mac/Linux
.venv\Scripts\activate           # Windows
```

### 3. Instalează dependențele
```bash
pip install -r requirements.txt
```

### 4. Rulează aplicația
```bash
streamlit run Home.py
```

Aplicația se deschide automat la `http://localhost:8501` 



Andreea S.
