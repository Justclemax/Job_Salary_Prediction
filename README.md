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
- **Application Streamlit** (`app/`) : 7 pages, thèmes clair et sombre.

```bash
uv sync                                   # macOS : brew install libomp
uv run python -m salary.stats             # rapports statistiques
uv run python -m salary.causal            # rapports causaux
uv run python -m salary.train             # entraînement des modèles
uv run streamlit run app/Home.py          # application
uv run pytest                             # tests
```
