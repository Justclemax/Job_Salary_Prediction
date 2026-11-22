"""Entry point: `uv run streamlit run app/Home.py`."""

import streamlit as st

import ui

st.set_page_config(page_title="Salary Lab — Prédiction des salaires tech",
                   page_icon=":material/payments:", layout="wide",
                   initial_sidebar_state="collapsed")
ui.apply_style()

pages = [
    st.Page("views/accueil.py", title="Synthèse", icon=":material/flag:", default=True),
    st.Page("views/exploration.py", title="Données", icon=":material/table_chart:"),
    st.Page("views/statistiques.py", title="Statistiques", icon=":material/functions:"),
    st.Page("views/causalite.py", title="Causalité", icon=":material/account_tree:"),
    st.Page("views/modeles.py", title="Modèles", icon=":material/model_training:"),
    st.Page("views/explicabilite.py", title="SHAP", icon=":material/psychology:"),
    st.Page("views/prediction.py", title="Simulateur", icon=":material/calculate:"),
    st.Page("views/monde.py", title="Monde réel", icon=":material/public:"),
]
st.navigation(pages, position="top").run()
ui.footer()
