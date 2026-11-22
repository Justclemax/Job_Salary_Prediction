# Job Salary Prediction

This dataset contains **250,000 job records** designed for salary prediction, data analysis, and machine learning projects.

It includes information about job titles, experience level, education background, skills, industry type, company size, and work location. The dataset helps researchers and data scientists explore how different factors influence salary.

## Use Cases

- Salary prediction models
- Exploratory data analysis (EDA)
- Machine learning practice
- Data visualization projects
- Career and industry salary insights

## Dataset Columns

| Column | Description |
|---|---|
| `job_title` | The job role or position (e.g., Data Analyst, AI Engineer) |
| `experience_years` | Number of years of professional experience |
| `education_level` | Highest level of education completed |
| `skills_count` | Number of technical or professional skills |
| `industry` | Industry sector where the job belongs |
| `company_size` | Size of the company (small, medium, large) |
| `location` | Job location or region |
| `remote_work` | Whether the job allows remote work |
| `certifications` | Number of professional certifications |
| `salary` | Annual salary of the employee |

## Source

- **Auteur :** Aleesha Nadeem
- **Licence :** CC0 Public Domain
- **Plateforme :** [Kaggle](https://www.kaggle.com/datasets/nalisha/job-salary-prediction-dataset)

## Projet

Analyse complète et application de prédiction :

- **Analyse statistique** (`src/salary/stats.py`) : tests avec tailles d'effet, indépendance des variables, modèle log-linéaire, décomposition ANOVA.
- **Analyse causale** (`src/salary/causal.py`) : DAG, DoWhy (régression ajustée, IPW, réfutations), EconML Double ML, effets hétérogènes (CATE).
- **Machine learning** (`src/salary/train.py`) : Ridge, Random Forest, XGBoost, LightGBM, CatBoost et un ensemble. Réglage Optuna, validation croisée à 5 plis, intervalles de prédiction conformalisés, SHAP.
- **Notebook** (`notebook/EDA.ipynb`) : le récit de l'analyse.
- **Données réelles** (`src/salary/world.py`, `src/salary/sources/`) : la même étude refaite sur 22 143 développeurs salariés de 29 pays (Stack Overflow Developer Survey 2025 et 2026, licence ODbL 1.0), avec la parité de pouvoir d'achat de l'API de la Banque mondiale. Comparaison synthétique vs réel, effets causaux avec un DAG explicite, modèles en validation croisée, simulateur mondial.
- **Inférence en Rust** (`serving/`) : serveur HTTP qui sert le modèle LightGBM et l'intervalle à 80 %, avec des prédictions identiques au bit près à Python.
- **Application Streamlit** (`app/`) : 8 pages, thèmes clair et sombre.

```bash
uv sync                                   # macOS : brew install libomp
uv run python -m salary.stats             # rapports statistiques
uv run python -m salary.causal            # rapports causaux
uv run python -m salary.train             # entraînement des modèles
uv run python -W ignore -m salary.world   # données réelles (télécharge ~260 Mo la 1re fois)
uv run streamlit run app/Home.py          # application
uv run pytest                             # tests
```

### Inférence en Rust

`salary.export` convertit les forêts LightGBM en un format JSON à plat. Le serveur Rust (`serving/`, axum) les charge sans Python ni bibliothèque de ML native. `cargo test` vérifie sur 500 profils de test que Rust reproduit Python à l'identique (écart mesuré : 0).

```bash
uv run python -m salary.export                         # écrit models/rust/
cd serving && cargo run --release                      # http://127.0.0.1:8000
curl -s localhost:8000/predict -H 'content-type: application/json' -d '{"job_title":"Data Scientist",
  "experience_years":5,"education_level":"Master","skills_count":8,"industry":"Technology",
  "company_size":"Medium","location":"Germany","remote_work":"Hybrid","certifications":1}'
# {"salary":139755.8,"low":132746.3,"high":146850.0}
SALARY_API_URL=http://127.0.0.1:8000 uv run streamlit run app/Home.py   # le simulateur interroge Rust
docker compose up --build                               # app + API (limite mémoire de l'API : 64 Mo)
```

Mesures sur un Mac Apple Silicon : modèle LightGBM et deux modèles quantiles, soit 2 300 arbres.

| | Python (scikit-learn + LightGBM) | Rust (`salary-serve`) |
|---|---|---|
| Mémoire du processus | 283 Mo | 7 Mo au démarrage, 28 Mo en charge |
| Chargement des modèles | 0,95 s | 12 ms |
| 1 profil | 6,3 ms (appel en mémoire) | 0,14 ms (aller-retour HTTP compris) |
| Lot de 10 000 profils | 327 ms (en mémoire) | 140 ms (HTTP), 86 ms (en mémoire) |
| Binaire / image | environnement Python complet | binaire de 1,1 Mo |

`GET /health`, `GET /model` (schéma, arbres, mémoire) et `POST /predict` (un profil, ou une liste d'au plus 10 000 profils). Les profils invalides renvoient une erreur 422 en français, avec l'indice du profil fautif.

Le serveur sert LightGBM plutôt que CatBoost, le modèle retenu en validation croisée (RMSE 5 103 $ contre 5 032 $, soit 1,4 % d'écart). CatBoost encode les catégories avec des statistiques internes et un hachage propres à sa bibliothèque C++, qu'on ne peut pas reproduire fidèlement en Rust pur.
