import plotly.express as px

import ui
from salary import config as cfg
from ui import st

ui.header("Modèles & métriques",
          "Cible : log(salaire), métriques en dollars. Sélection du modèle sur la validation croisée "
          "(jamais sur le test). Hyper-paramètres des boosters réglés avec Optuna.", art="models", kicker="Machine learning")
col = ui.colors()
m = ui.metrics()
best = m["best_model"]
rows = []
for name, r in m["models"].items():
    rows.append({"key": name, "Modèle": r["display_name"],
                 "CV RMSE": r["cv"]["rmse_mean"], "± CV": r["cv"]["rmse_std"],
                 "Test RMSE": r["test"]["rmse"], "Test MAE": r["test"]["mae"],
                 "MedAE": r["test"]["medae"], "MAPE %": r["test"]["mape"], "R²": r["test"]["r2"],
                 "±10 % (%)": r["test"]["within_10pct"], "Train RMSE": r["train"]["rmse"],
                 "Écart test−train": r["overfit_gap_rmse"], "Entraînement (s)": r["fit_time_s"],
                 "Prédiction (ms/1k)": r["predict_ms_per_1k"]})
tab = ui.pd.DataFrame(rows).sort_values("CV RMSE")

b = m["models"][best]
c = st.columns(4)
c[0].metric("Modèle retenu", ui.model_label(best))
c[1].metric("RMSE test", ui.money(b["test"]["rmse"]))
c[2].metric("R² test", f"{b['test']['r2']:.4f}".replace(".", ","))
c[3].metric("Couverture de l'intervalle 80 %", f"{m['interval']['coverage']:.1f} %".replace(".", ","),
            help=f"Largeur moyenne {ui.money(m['interval']['mean_width'])} (régression quantile conformalisée)")

t1, t2, t3, t4 = st.tabs(["Comparaison", "Diagnostic", "Erreurs par segment", "Réglage Optuna"])

with t1:
    with_base = st.toggle("Inclure la baseline (médiane)", value=False)
    d = tab if with_base else tab[tab.key != "dummy"]
    d = d.sort_values("CV RMSE", ascending=False)
    fig = px.bar(d, x="CV RMSE", y="Modèle", orientation="h", error_x="± CV",
                 color_discrete_sequence=[col["series"][0]],
                 text=[("★ " if k == best else "") + ui.money(v) for k, v in zip(d.key, d["CV RMSE"], strict=False)],
                 labels={"CV RMSE": "RMSE en validation croisée ($, moyenne ± écart-type)", "Modèle": ""})
    fig.update_traces(textposition="outside", hovertemplate="%{y} : %{x:,.0f} $<extra></extra>")
    fig.update_layout(title=f"Validation croisée à {m['metadata']['cv_folds']} plis — plus bas = meilleur")
    ui.show(ui.style(fig, height=380, legend=False))

    long = d.rename(columns={"Train RMSE": "Entraînement", "Test RMSE": "Test"}).melt(
        id_vars="Modèle", value_vars=["Entraînement", "Test"], var_name="Jeu", value_name="RMSE")
    fig = px.bar(long, x="RMSE", y="Modèle", color="Jeu", barmode="group", orientation="h",
                 color_discrete_sequence=col["series"][:2], labels={"Modèle": ""})
    fig.update_traces(hovertemplate="%{y} : %{x:,.0f} $<extra>%{fullData.name}</extra>")
    fig.update_layout(title="Surapprentissage : RMSE entraînement vs test", legend_title_text="")
    ui.show(ui.style(fig, height=380))

    st.dataframe(tab.drop(columns="key").style.format(
        {c: "{:,.0f}" for c in ["CV RMSE", "± CV", "Test RMSE", "Test MAE", "MedAE", "Train RMSE", "Écart test−train"]}
        | {"MAPE %": "{:.2f}", "R²": "{:.4f}", "±10 % (%)": "{:.1f}", "Entraînement (s)": "{:.1f}",
           "Prédiction (ms/1k)": "{:.1f}"}), width="stretch", hide_index=True)

preds = ui.load_parquet("models", "predictions_test.parquet")
names = [k for k in tab.key if f"pred_{k}" in preds.columns and k != "dummy"]

with t2:
    sel = st.selectbox("Modèle", names, index=names.index(best) if best in names else 0,
                       format_func=ui.model_label)
    s = ui.sample(preds, 4000)
    left, right = st.columns(2)
    with left:
        fig = px.scatter(s, x="salary", y=f"pred_{sel}", opacity=0.45,
                         color_discrete_sequence=[col["series"][0]],
                         labels={"salary": "Salaire réel ($)", f"pred_{sel}": "Salaire prédit ($)"})
        fig.update_traces(marker_size=5, hovertemplate="réel %{x:,.0f} $ · prédit %{y:,.0f} $<extra></extra>")
        lo, hi = s.salary.min(), s.salary.max()
        fig.add_scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=col["muted"], width=2, dash="dash"),
                        name="Prédiction parfaite", hoverinfo="skip")
        fig.update_layout(title="Prédit vs réel (échantillon test)")
        ui.show(ui.style(fig, height=420))
    with right:
        res = (preds[f"pred_{sel}"] - preds.salary) / preds.salary * 100
        fig = px.histogram(res, nbins=100, color_discrete_sequence=[col["series"][0]],
                           labels={"value": "Erreur relative (%)"})
        fig.update_traces(hovertemplate="%{x:.1f} % : %{y}<extra></extra>", marker_line_width=0)
        fig.update_layout(title="Distribution des erreurs relatives", yaxis_title="Profils")
        ui.show(ui.style(fig, height=420, legend=False))
    q = res.abs().quantile([.5, .8, .95])
    st.caption(f"50 % des erreurs < {q[.5]:.1f} % · 80 % < {q[.8]:.1f} % · 95 % < {q[.95]:.1f} %. "
               "Le bruit résiduel (≈ 4–5 %) est irréductible : il n'est porté par aucune variable.")

with t3:
    seg = ui.load_csv("models", "segment_errors.csv")
    f = st.selectbox("Segment", seg.feature.unique(), format_func=ui.feature_label)
    d = seg[seg.feature == f].sort_values("mape")
    fig = px.bar(d, x="mape", y="segment", orientation="h", color_discrete_sequence=[col["series"][0]],
                 text=d.mape.map(lambda v: f"{v:.2f} %"), labels={"mape": "MAPE (%)", "segment": ""})
    fig.update_traces(textposition="outside", hovertemplate="%{y} : %{x:.2f} %<extra></extra>")
    fig.update_layout(title=f"Erreur du modèle retenu par {ui.feature_label(f).lower()}")
    ui.show(ui.style(fig, height=420, legend=False))
    st.dataframe(d.drop(columns="feature").rename(columns={"segment": "Segment", "n": "n", "mae": "MAE ($)",
                 "mape": "MAPE (%)", "bias": "Biais moyen ($)"}).style.format(
        {"MAE ($)": "{:,.0f}", "MAPE (%)": "{:.2f}", "Biais moyen ($)": "{:+,.0f}"}),
        width="stretch", hide_index=True)

with t4:
    boosters = [b for b in ["xgboost", "lightgbm", "catboost"] if (cfg.REPORTS_DIR / "models" / f"optuna_{b}.csv").exists()]
    if not boosters:
        st.info("Aucun historique Optuna (entraînement lancé avec --no-tune).")
    else:
        bsel = st.selectbox("Booster", boosters, format_func=ui.model_label)
        tr = ui.load_csv("models", f"optuna_{bsel}.csv").dropna(subset=["value"])
        tr["best"] = tr.value.cummin()
        fig = ui.go.Figure()
        fig.add_scatter(x=tr.number, y=tr.value, mode="markers", name="Essai",
                        marker=dict(size=8, color=col["series"][0]),
                        hovertemplate="essai %{x} : %{y:,.0f} $<extra></extra>")
        fig.add_scatter(x=tr.number, y=tr.best, mode="lines", name="Meilleur jusqu'ici",
                        line=dict(width=2, color=col["series"][1], shape="hv"),
                        hovertemplate="meilleur : %{y:,.0f} $<extra></extra>")
        fig.update_layout(title="Historique de l'optimisation (RMSE en CV sur 50 000 lignes)",
                          xaxis_title="Essai", yaxis_title="RMSE ($)")
        ui.show(ui.style(fig, height=380))
        st.json(m["models"][bsel]["params"], expanded=False)

with st.expander("Métadonnées d'entraînement (reproductibilité)"):
    st.json(m["metadata"])
