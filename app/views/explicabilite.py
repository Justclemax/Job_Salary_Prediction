import numpy as np
import plotly.express as px

import ui
from salary import config as cfg
from ui import st

m = ui.metrics()
ui.header("Explicabilité (SHAP)",
          f"Valeurs SHAP du modèle {ui.model_label(m['shap_model'])} sur 3 000 profils du jeu de test. "
          "Le modèle prédit log(salaire) : chaque valeur SHAP est un **effet multiplicatif** "
          "(exp(φ) − 1 = % d'écart au salaire de référence).", art="explain", kicker="Machine learning")
col = ui.colors()
L = ui.feature_label
imp = ui.load_csv("models", "shap_importance.csv")
sv = ui.load_parquet("models", "shap_sample.parquet")
base = float(np.exp(m["shap_base_value_log"]))
st.metric("Salaire de référence (moyenne géométrique des prédictions)", ui.money(base))

left, right = st.columns([2, 3])
with left:
    d = imp.sort_values("mean_pct_effect")
    fig = px.bar(d, x="mean_pct_effect", y=d.feature.map(L), orientation="h",
                 color_discrete_sequence=[col["series"][0]], text=d.mean_pct_effect.map(lambda v: f"{v:.1f} %"),
                 labels={"mean_pct_effect": "Impact moyen |SHAP| (en %)", "y": ""})
    fig.update_traces(textposition="outside", hovertemplate="%{y} : %{x:.2f} %<extra></extra>")
    fig.update_layout(title="Importance globale")
    ui.show(ui.style(fig, height=440, legend=False))
with right:
    f = st.selectbox("Effet d'une variable (dépendance)", imp.feature, format_func=L)
    d = sv[[f, f"shap_{f}"]].copy()
    d["pct"] = (np.exp(d[f"shap_{f}"]) - 1) * 100
    if f in cfg.NUMERIC:
        g = d.groupby(f)["pct"].agg(["mean", "std"]).reset_index()
        fig = px.strip(d, x=f, y="pct", color_discrete_sequence=[col["series"][0]],
                       labels={f: L(f), "pct": "Effet sur le salaire (%)"})
        fig.update_traces(marker=dict(size=4, opacity=0.35), hoverinfo="skip")
        fig.add_scatter(x=g[f], y=g["mean"], mode="lines+markers", name="Moyenne",
                        line=dict(width=2, color=col["series"][1]), marker=dict(size=8),
                        hovertemplate="%{x} : %{y:+.2f} %<extra></extra>")
    else:
        order = d.groupby(f)["pct"].median().sort_values().index.tolist()
        fig = px.box(d, x=f, y="pct", category_orders={f: order}, points=False,
                     color_discrete_sequence=[col["series"][0]],
                     labels={f: L(f), "pct": "Effet sur le salaire (%)"})
    fig.add_hline(y=0, line_width=1, line_color=col["muted"])
    fig.update_layout(title=f"Contribution SHAP de « {L(f)} »")
    ui.show(ui.style(fig, height=440))

st.subheader("Vue d'ensemble : contributions par variable")
top = imp.feature.tolist()
long = ui.pd.concat([ui.pd.DataFrame({"Variable": L(f), "Effet (%)": (np.exp(sv[f"shap_{f}"]) - 1) * 100})
                     for f in top])
fig = px.strip(ui.sample(long, 15000), x="Effet (%)", y="Variable",
               category_orders={"Variable": [L(f) for f in top]},
               color_discrete_sequence=[col["series"][0]])
fig.update_traces(marker=dict(size=4, opacity=0.3), hovertemplate="%{y} : %{x:+.1f} %<extra></extra>")
fig.add_vline(x=0, line_width=1, line_color=col["muted"])
fig.update_layout(title="Chaque point = un profil (contribution SHAP en %)")
ui.show(ui.style(fig, height=460, legend=False))
st.caption("SHAP explique **le modèle**, pas la causalité. Ici les deux concordent parce que les "
           "variables sont indépendantes. Les effets causaux sont dans *Analyse causale*.")
