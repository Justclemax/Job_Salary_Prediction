import numpy as np
import plotly.express as px

import ui
from ui import st

ui.header("Analyse statistique",
          "Avec n = 250 000, presque toutes les p-valeurs sont ≈ 0 : on juge l'importance "
          "pratique sur les tailles d'effet (η², ε², r).", art="statistics", kicker="Analyse")
col = ui.colors()
L = ui.feature_label

t1, t2, t3, t4 = st.tabs(["Distribution du salaire", "Tests univariés",
                          "Modèle log-linéaire", "Indépendance des variables"])

with t1:
    ts = ui.load_csv("stats", "target_summary.csv").set_index("variable")
    s, lg = ts.loc["salary"], ts.loc["log_salary"]
    c = st.columns(4)
    c[0].metric("Asymétrie (skewness)", f"{s.skewness:.2f}", help="0 = symétrique")
    c[1].metric("Kurtosis (excès)", f"{s.excess_kurtosis:.2f}")
    c[2].metric("Asymétrie du log", f"{lg.skewness:.2f}")
    c[3].metric("Shapiro-Wilk W (n=5000)", f"{s.shapiro_w_5000:.4f}")
    st.markdown(
        "Le salaire est **légèrement asymétrique à droite** et sa dispersion croît avec son niveau "
        "(variance hétérogène, test de Levene). Ces deux signes indiquent une structure "
        "**multiplicative** : on modélise donc **log(salaire)**, où les effets deviennent des "
        "pourcentages. Les tests de normalité rejettent H₀ (attendu avec un tel n) : on complète "
        "chaque ANOVA par son équivalent non paramétrique (Kruskal-Wallis).")
    st.dataframe(ts.T.style.format("{:,.4g}"), width="stretch")

with t2:
    cat = ui.load_csv("stats", "categorical_tests.csv")
    num = ui.load_csv("stats", "numeric_correlations.csv")
    left, right = st.columns(2)
    with left:
        d = cat.sort_values("eta2")
        fig = px.bar(d, x="eta2", y=d.feature.map(L), orientation="h",
                     color_discrete_sequence=[col["series"][0]],
                     labels={"eta2": "η² (part de variance)", "y": ""},
                     text=d.eta2.map(lambda v: f"{v:.3f}"))
        fig.update_traces(hovertemplate="%{y} : η² = %{x:.4f}<extra></extra>", textposition="outside")
        fig.update_layout(title="Variables catégorielles — ANOVA")
        ui.show(ui.style(fig, legend=False))
    with right:
        d = num.sort_values("pearson_r")
        fig = px.bar(d, x="pearson_r", y=d.feature.map(L), orientation="h",
                     color_discrete_sequence=[col["series"][0]], error_x=d.r_ci_high - d.pearson_r,
                     labels={"pearson_r": "r de Pearson (IC 95 %)", "y": ""},
                     text=d.pearson_r.map(lambda v: f"{v:.3f}"))
        fig.update_traces(hovertemplate="%{y} : r = %{x:.4f}<extra></extra>", textposition="outside")
        fig.update_layout(title="Variables numériques — corrélation")
        ui.show(ui.style(fig, legend=False))

    show = cat[["feature", "k_groups", "anova_F", "anova_p", "eta2", "eta2_effect", "kruskal_H",
                "kruskal_p", "epsilon2", "levene_p", "highest", "lowest", "max_gap_means"]].copy()
    show["feature"] = show.feature.map(L)
    st.dataframe(show.rename(columns={
        "feature": "Variable", "k_groups": "Groupes", "anova_F": "F", "anova_p": "p (ANOVA)",
        "eta2": "η²", "eta2_effect": "Effet", "kruskal_H": "H", "kruskal_p": "p (Kruskal)",
        "epsilon2": "ε²", "levene_p": "p (Levene)", "highest": "Plus haut", "lowest": "Plus bas",
        "max_gap_means": "Écart max ($)"}).style.format(
        {"F": "{:,.1f}", "H": "{:,.0f}", "η²": "{:.4f}", "ε²": "{:.4f}", "Écart max ($)": "{:,.0f}",
         "p (ANOVA)": "{:.2g}", "p (Kruskal)": "{:.2g}", "p (Levene)": "{:.2g}"}),
        width="stretch", hide_index=True)
    n2 = num.copy()
    n2["feature"] = n2.feature.map(L)
    st.dataframe(n2[["feature", "pearson_r", "r_ci_low", "r_ci_high", "spearman_rho", "effect",
                     "slope_per_unit"]].rename(columns={
        "feature": "Variable", "pearson_r": "r", "r_ci_low": "IC bas", "r_ci_high": "IC haut",
        "spearman_rho": "ρ Spearman", "effect": "Effet", "slope_per_unit": "Pente ($/unité)"}).style.format(
        {"r": "{:.4f}", "IC bas": "{:.4f}", "IC haut": "{:.4f}", "ρ Spearman": "{:.4f}",
         "Pente ($/unité)": "{:,.0f}"}), width="stretch", hide_index=True)
    with st.expander("Comparaisons post-hoc (Tukey HSD)"):
        st.dataframe(ui.load_csv("stats", "tukey_posthoc.csv"), width="stretch", hide_index=True)

with t3:
    summ = ui.load_json("reports/stats/summary.json")["log_linear"]
    aov = ui.load_csv("stats", "anova_type2.csv")
    coefs = ui.load_csv("stats", "log_linear_coefficients.csv")
    c = st.columns(3)
    c[0].metric("R² du modèle log-linéaire", f"{summ['r2']:.4f}".replace(".", ","))
    c[1].metric("Écart-type résiduel", ui.pct((np.exp(summ['residual_std_log']) - 1) * 100, False),
                help="Bruit multiplicatif typique autour de la prédiction")
    c[2].metric("Observations", f"{summ['n']:,}".replace(",", " "))
    st.markdown("Régression MCO de **log(salaire)** sur toutes les variables, erreurs standard "
                "robustes HC3. Chaque coefficient se lit comme un **% d'écart de salaire, toutes "
                "choses égales par ailleurs**, par rapport à la modalité de référence.")
    left, right = st.columns([2, 3])
    with left:
        d = aov.sort_values("share_explained_pct")
        fig = px.bar(d, x="share_explained_pct", y=d.feature.map(L), orientation="h",
                     color_discrete_sequence=[col["series"][0]],
                     text=d.share_explained_pct.map(lambda v: f"{v:.1f} %"),
                     labels={"share_explained_pct": "% de la variance expliquée", "y": ""})
        fig.update_traces(textposition="outside", hovertemplate="%{y} : %{x:.2f} %<extra></extra>")
        fig.update_layout(title="Décomposition (ANOVA type II)")
        ui.show(ui.style(fig, height=420, legend=False))
    with right:
        feats = [f for f in coefs.feature.unique() if f != "Intercept"]
        f = st.selectbox("Variable", feats, format_func=L)
        d = coefs[coefs.feature == f].copy()
        d["name"] = d.level.fillna("+1 unité")
        d = d.sort_values("pct_effect")
        fig = px.scatter(d, x="pct_effect", y="name", color_discrete_sequence=[col["series"][0]],
                         error_x=d.pct_ci_high - d.pct_effect, error_x_minus=d.pct_effect - d.pct_ci_low,
                         labels={"pct_effect": "Effet sur le salaire (%)", "name": ""})
        fig.update_traces(marker_size=10, hovertemplate="%{y} : %{x:+.2f} %<extra></extra>")
        fig.add_vline(x=0, line_width=1, line_color=col["muted"])
        ref = {"job_title": "Data Analyst", "education_level": "High School", "company_size": "Startup",
               "location": "India", "remote_work": "No", "industry": "Retail"}.get(f)
        fig.update_layout(title=f"{L(f)}" + (f" — référence : {ref}" if ref else ""))
        ui.show(ui.style(fig, height=420, legend=False))
    sep = ui.load_csv("stats", "separability_tests.csv")
    st.markdown("**Les effets sont-ils purement multiplicatifs ?** Tests F emboîtés d'interactions :")
    st.dataframe(sep.style.format({"F": "{:,.1f}", "p_value": "{:.2g}", "delta_r2": "{:.5f}"}),
                 width="stretch", hide_index=True)
    st.caption("Les interactions expérience × poste et expérience × études sont significatives mais "
               "n'ajoutent que ~0,1–0,2 point de R² : c'est ce gain que les modèles de boosting captent "
               "en plus du modèle log-linéaire.")

with t4:
    mat = ui.load_csv("stats", "feature_association_matrix.csv").set_index("Unnamed: 0")
    labels = [L(c) for c in mat.columns]
    z = mat.values.copy()
    np.fill_diagonal(z, np.nan)
    fig = ui.go.Figure(ui.go.Heatmap(z=z, x=labels, y=labels, colorscale=ui.seq_scale(), zmin=0,
                                     zmax=0.05, xgap=2, ygap=2, colorbar=dict(title="Assoc."),
                                     hovertemplate="%{y} × %{x} : %{z:.4f}<extra></extra>"))
    fig.update_layout(title="Association entre variables (V de Cramér, η, |ρ|) — échelle 0 à 0,05")
    ui.show(ui.style(fig, height=520, legend=False))
    pairs = ui.load_csv("stats", "feature_association_pairs.csv")
    st.success(f"Association maximale entre deux variables : **{pairs.value.max():.4f}** "
               f"({L(pairs.iloc[0].feature_a)} × {L(pairs.iloc[0].feature_b)}). Les variables sont "
               "indépendantes : aucune ne confond l'effet d'une autre, ce qui simplifie l'analyse causale.",
               icon=":material/check_circle:")
