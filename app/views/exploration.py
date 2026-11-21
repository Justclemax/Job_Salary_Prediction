import plotly.express as px

import ui
from salary import config as cfg
from ui import st

ui.header("Exploration des données", "Distribution des salaires et comparaison par variable.", art="exploration", kicker="Analyse")
df = ui.load_data()

f1, f2, f3 = st.columns(3)
jobs = f1.multiselect("Poste", cfg.CATEGORY_LEVELS["job_title"], placeholder="Tous les postes")
locs = f2.multiselect("Localisation", cfg.CATEGORY_LEVELS["location"], placeholder="Toutes")
edus = f3.multiselect("Niveau d'études", cfg.CATEGORY_LEVELS["education_level"], placeholder="Tous")
d = df
if jobs:
    d = d[d.job_title.isin(jobs)]
if locs:
    d = d[d.location.isin(locs)]
if edus:
    d = d[d.education_level.isin(edus)]
if d.empty:
    st.warning("Aucun profil ne correspond à ces filtres.")
    st.stop()

c = st.columns(4)
c[0].metric("Profils", f"{len(d):,}".replace(",", " "))
c[1].metric("Moyenne", ui.money(d.salary.mean()))
c[2].metric("Médiane", ui.money(d.salary.median()))
c[3].metric("Écart interquartile", ui.money(d.salary.quantile(.75) - d.salary.quantile(.25)))

col = ui.colors()
left, right = st.columns(2)
with left:
    fig = px.histogram(d, x="salary", nbins=80, labels={"salary": ui.feature_label("salary")},
                       color_discrete_sequence=[col["series"][0]])
    fig.update_traces(hovertemplate="%{x:,.0f} $ : %{y} profils<extra></extra>", marker_line_width=0)
    fig.update_layout(title="Distribution des salaires", yaxis_title="Profils")
    ui.show(ui.style(fig, legend=False))
with right:
    g = d.groupby("experience_years")["salary"].agg(["mean", "median"]).reset_index()
    fig = ui.go.Figure()
    for i, (k, name) in enumerate([("mean", "Moyenne"), ("median", "Médiane")]):
        fig.add_scatter(x=g.experience_years, y=g[k], name=name, mode="lines+markers",
                        line=dict(width=2, color=col["series"][i]), marker=dict(size=8),
                        hovertemplate=f"{name} : %{{y:,.0f}} $<extra></extra>")
    fig.update_layout(title="Salaire selon l'expérience", xaxis_title="Années d'expérience",
                      yaxis_title="Salaire ($)", hovermode="x unified")
    ui.show(ui.style(fig))

feat = st.selectbox("Comparer le salaire par", cfg.CATEGORICAL + cfg.NUMERIC,
                    format_func=ui.feature_label)
order = cfg.CATEGORY_LEVELS.get(feat)
if order is None:
    order = sorted(d[feat].unique())
elif feat in cfg.NOMINAL:
    order = d.groupby(feat)["salary"].median().sort_values().index.tolist()
fig = px.box(ui.sample(d, 30_000), x=feat, y="salary", category_orders={feat: order},
             labels={feat: ui.feature_label(feat), "salary": "Salaire ($)"},
             color_discrete_sequence=[col["series"][0]], points=False)
fig.update_layout(title=f"Salaire par {ui.feature_label(feat).lower()}")
ui.show(ui.style(fig, height=420, legend=False))

tab = (d.groupby(feat)["salary"].agg(["count", "mean", "median", "std"]).reindex(order)
       .rename(columns={"count": "Profils", "mean": "Moyenne", "median": "Médiane", "std": "Écart-type"}))
st.dataframe(tab.style.format("{:,.0f}"), width="stretch")

with st.expander("Aperçu des données brutes"):
    st.dataframe(d.head(200).rename(columns=ui.LABELS), width="stretch", hide_index=True)
