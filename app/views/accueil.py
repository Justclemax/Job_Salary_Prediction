import plotly.express as px

import ui
from ui import st

df = ui.load_data()
m = ui.metrics()
best = m["best_model"]
bt = m["models"][best]["test"]
aov = ui.load_csv("stats", "anova_type2.csv").set_index("feature")
ate = ui.load_csv("causal", "ate.csv").set_index("treatment")
refs = ui.load_csv("causal", "refutations.csv")
coefs = ui.load_csv("stats", "log_linear_coefficients.csv")
edu = ate.loc["education_advanced"]
usa = coefs[(coefs.feature == "location") & (coefs.level == "USA")].pct_effect.iloc[0]
fmt = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")  # noqa: E731

ui.hero(
    "Salary Lab · Data science appliquée",
    "Qu'est-ce qui fait vraiment le salaire d'un profil tech&nbsp;?",
    f"{len(df):,}".replace(",", "&nbsp;") + " profils, 12 métiers, 10 pays. Je sépare ce qui est <b>associé</b> au salaire "
    "de ce qui le <b>cause</b>, puis je construis un modèle capable de l'estimer pour n'importe quel profil, "
    "avec une marge d'erreur honnête.",
    pills=[(f"✦ Modèle retenu : {ui.model_label(best)}", "claire"),
           (f"R² {fmt(bt['r2'], 3)} sur le test", "sombre"),
           (f"{fmt(bt['mape'])} % d'erreur moyenne", "sombre")],
    art="home",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi("Précision du modèle", f"{fmt(bt['mape'])} %", f"d'erreur moyenne · {fmt(bt['within_10pct'], 0)} % des "
           "prédictions à ±10 %", "lavande", badge=f"R² {fmt(bt['r2'], 3)}", badge_cls="vert")
with c2:
    ui.kpi("Effet causal d'un Master/PhD", ui.money(edu.dml_effect, True),
           f"par an · IC 95 % [{ui.money(edu.dml_ci_low)} ; {ui.money(edu.dml_ci_high)}]", "menthe",
           badge=ui.pct(edu.dml_pct), badge_cls="lavande")
with c3:
    ui.kpi("Facteur n°1 : le pays", f"{aov.loc['location', 'share_explained_pct']:.0f} %",
           "de la variance expliquée du log-salaire", "jaune", badge=f"USA vs Inde {ui.pct(usa, digits=0)}",
           badge_cls="jaune")
with c4:
    ui.kpi("Intervalle de prédiction", f"{fmt(m['interval']['coverage'])} %",
           "des vrais salaires tombent dans l'intervalle à 80 % annoncé", "gris",
           badge=f"± {ui.money(m['interval']['mean_width'] / 2)}", badge_cls="gris")

st.write("")
left, right = st.columns([1.5, 1], gap="medium")
with left:
    g = df.groupby("location").salary.median().sort_values()
    col = ui.colors()
    fig = px.bar(x=g.values, y=g.index, orientation="h", color_discrete_sequence=[col["series"][0]],
                 text=[ui.money(v) for v in g.values], labels={"x": "Salaire médian ($)", "y": ""})
    fig.update_traces(textposition="outside", hovertemplate="%{y} : %{x:,.0f} $<extra></extra>", cliponaxis=False)
    fig.update_layout(title="Salaire médian par pays", xaxis_range=[0, g.max() * 1.18])
    ui.show(ui.style(fig, height=430, legend=False))
    ui.reading(f"un même profil gagne près de deux fois plus aux USA qu'en Inde (≈ {ui.pct(usa, digits=0)} à profil égal "
               "dans le modèle log-linéaire). Le pays pèse plus que le métier, l'expérience ou le diplôme.")
with right:
    d = ate.sort_values("dml_pct", ascending=False)
    colors = ["lavande", "lavande", "jaune", "jaune", "menthe", "gris"]
    rows = [(i + 1, r.label, f"IC 95 % [{ui.pct(r.dml_pct_ci_low)} ; {ui.pct(r.dml_pct_ci_high)}] · {ui.money(r.dml_effect, True)}"
             + (f" {r.unit}" if isinstance(r.unit, str) and r.unit else ""), ui.pct(r.dml_pct), colors[i])
            for i, (_, r) in enumerate(d.iterrows())]
    ui.ranking("Ce qui fait monter le salaire", "Effet causal moyen (Double ML), toutes choses égales par ailleurs", rows)
    st.page_link("views/prediction.py", label="Simuler un profil", icon=":material/arrow_forward:")

n_ok = int(((refs.refuter == "placebo_treatment_refuter") & (refs.new_effect.abs() < 0.05 * refs.estimated_effect.abs())).sum()
           + ((refs.refuter != "placebo_treatment_refuter")
              & ((refs.new_effect - refs.estimated_effect).abs() < 0.02 * refs.estimated_effect.abs())).sum())
gap = float(((ate.naive_effect - ate.dml_effect).abs() / ate.dml_effect.abs()).max() * 100)
ui.explain_box(
    "Corrélation ou causalité : pourquoi les deux coïncident ici",
    f"Les variables explicatives sont <b>indépendantes entre elles</b> (association maximale &lt; 0,01) : aucune ne brouille "
    f"l'effet d'une autre. L'écart naïf et l'effet ajusté diffèrent donc de {fmt(gap)} % au plus, et "
    f"<b>{n_ok} tests de réfutation sur {len(refs)}</b> confirment les estimations (placebo ≈ 0, résultats stables). "
    "Sur de vraies données, études, poste et pays seraient liés, et l'écart naïf serait biaisé.")

best_rmse = ui.money(m["models"][best]["cv"]["rmse_mean"])
ridge_rmse = ui.money(m["models"]["ridge"]["cv"]["rmse_mean"])
ui.decision("📌 Conclusions en huit lignes", [
    "<b>Données :</b> 250 000 profils propres et parfaitement équilibrés, donc très probablement synthétiques. Les conclusions "
    "décrivent le générateur, pas le marché réel.",
    f"<b>Déterminants :</b> pays ({aov.loc['location', 'share_explained_pct']:.0f} %), expérience "
    f"({aov.loc['experience_years', 'share_explained_pct']:.0f} %), métier ({aov.loc['job_title', 'share_explained_pct']:.0f} %), "
    f"taille d'entreprise ({aov.loc['company_size', 'share_explained_pct']:.0f} %), diplôme "
    f"({aov.loc['education_level', 'share_explained_pct']:.0f} %) de la variance expliquée.",
    f"<b>Le secteur ne compte pas :</b> aucun effet (p = {fmt(aov.loc['industry', 'PR(>F)'], 2)}), seul résultat nul malgré n = 250 000.",
    "<b>Structure multiplicative :</b> chaque facteur agit en %, d'où une cible en log(salaire). Le modèle log-linéaire explique "
    "déjà 97,3 % de la variance.",
    f"<b>Effets causaux :</b> grande entreprise {ui.pct(ate.loc['large_company', 'dml_pct'])}, Master/PhD "
    f"{ui.pct(edu.dml_pct)}, télétravail total {ui.pct(ate.loc['remote_yes', 'dml_pct'])}, +1 an d'expérience "
    f"{ui.pct(ate.loc['experience_years', 'dml_pct'])}.",
    f"<b>Modèles :</b> {ui.model_label(best)} est retenu en validation croisée (RMSE {best_rmse}, contre {ridge_rmse} pour "
    "Ridge). XGBoost et LightGBM sont à quelques dizaines de dollars près.",
    "<b>Limite :</b> environ 4 à 5 % de bruit ne s'explique par aucune variable. C'est le plancher d'erreur de tout modèle.",
    f"<b>En pratique :</b> le simulateur donne une estimation et un intervalle à 80 % (couverture mesurée "
    f"{fmt(m['interval']['coverage'])} %), et explique chaque facteur avec SHAP.",
])
st.write("")
ui.synthetic_notice()
