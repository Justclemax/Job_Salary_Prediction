import plotly.express as px

import ui
from salary import config as cfg
from ui import st

ui.header("Analyse causale",
          "Que se passerait-il si l'on changeait une seule caractéristique, tout le reste égal par "
          "ailleurs ? Graphe causal → identification → estimation → réfutation (DoWhy + EconML).", art="causal", kicker="Analyse")
col = ui.colors()
L = ui.feature_label
ate = ui.load_csv("causal", "ate.csv")
refs = ui.load_csv("causal", "refutations.csv")
cate = ui.load_csv("causal", "cate.csv")
labels = dict(zip(ate.treatment, ate.label, strict=False))

t1, t2, t3, t4 = st.tabs(["1 · Graphe causal", "2 · Effets moyens (ATE)", "3 · Réfutations",
                          "4 · Hétérogénéité (CATE)"])

with t1:
    left, right = st.columns([3, 2])
    with left:
        fg, bg = ("#f0efec", "#1a1a19") if ui.mode() == "dark" else ("#0b0b0b", "#fcfcfb")
        edge = col["series"][0]
        nodes = "\n".join(f'  "{f}" [label="{L(f)}"];' for f in cfg.FEATURES)
        dot = (f'digraph {{ rankdir=LR; bgcolor="transparent"; size="6.5,4.5"; nodesep=0.12; ranksep=1.1; '
               f'node [shape=box style="rounded,filled" fillcolor="{bg}" color="{edge}" fontcolor="{fg}" fontname="Helvetica"]; '
               f'edge [color="{edge}"];\n{nodes}\n  "salary" [label="Salaire" fillcolor="{edge}" fontcolor="white"];\n'
               + "\n".join(f'  "{f}" -> "salary";' for f in cfg.FEATURES) + "\n}")
        st.graphviz_chart(dot, width="content")
    with right:
        st.markdown(
            "**Hypothèse :** chaque caractéristique est une cause directe du salaire et aucune "
            "n'en cause une autre.\n\n"
            "**Implication testable :** les variables doivent être deux à deux indépendantes. "
            "C'est le cas (association max < 0,01, onglet *Indépendance* de l'analyse statistique).\n\n"
            "**Identification :** critère *backdoor*. Pour chaque traitement, l'ensemble "
            "d'ajustement est « toutes les autres variables ».\n\n"
            "**Dans le monde réel**, études → poste ou expérience → certifications seraient des "
            "liens plausibles, avec de vrais facteurs de confusion. Ici, les données ne les "
            "contiennent pas.")
    ui.synthetic_notice()

with t2:
    d = ate.sort_values("dml_pct")
    fig = px.scatter(d, x="dml_pct", y=d.label, color_discrete_sequence=[col["series"][0]],
                     error_x=d.dml_pct_ci_high - d.dml_pct, error_x_minus=d.dml_pct - d.dml_pct_ci_low,
                     text=d.dml_pct.map(lambda v: ui.pct(v)),
                     labels={"dml_pct": "Effet causal sur le salaire (%, Double ML, IC 95 %)", "y": ""})
    fig.update_traces(marker_size=11, textposition="top center",
                      hovertemplate="%{y} : %{x:+.2f} %<extra></extra>")
    fig.add_vline(x=0, line_width=1, line_color=col["muted"])
    fig.update_layout(title="Effet moyen de chaque traitement (en %)")
    ui.show(ui.style(fig, height=400, legend=False))

    key = st.selectbox("Comparer les estimateurs pour", ate.treatment, format_func=labels.get)
    r = ate.set_index("treatment").loc[key]
    est = [("Différence naïve", r.naive_effect), ("Régression ajustée (DoWhy)", r.ols_effect)]
    if r.binary:
        est.append(("Pondération par propension (IPW)", r.ipw_effect))
    est.append(("Double ML (EconML)", r.dml_effect))
    e = ui.pd.DataFrame(est, columns=["Estimateur", "Effet"])
    fig = px.bar(e, x="Effet", y="Estimateur", orientation="h", color_discrete_sequence=[col["series"][0]],
                 text=e.Effet.map(ui.money), labels={"Effet": f"Effet en $ {r.unit}".strip(), "Estimateur": ""})
    fig.update_traces(textposition="outside", hovertemplate="%{y} : %{x:,.0f} $<extra></extra>")
    fig.update_layout(title=f"{r.label} — tous les estimateurs concordent")
    ui.show(ui.style(fig, height=300, legend=False))
    gap = abs(r.naive_effect - r.dml_effect) / abs(r.dml_effect) * 100
    st.caption(f"Écart naïf vs ajusté : {gap:.1f} %. Un écart faible confirme l'absence de confusion. "
               "Sur des données réelles, cet écart mesurerait le biais de l'analyse naïve.")

    tab = ate[["label", "n", "naive_effect", "ols_effect", "ols_ci_low", "ols_ci_high", "ipw_effect",
               "dml_effect", "dml_ci_low", "dml_ci_high", "dml_pct"]].rename(columns={
        "label": "Traitement", "n": "n", "naive_effect": "Naïf ($)", "ols_effect": "Ajusté ($)",
        "ols_ci_low": "IC bas", "ols_ci_high": "IC haut", "ipw_effect": "IPW ($)",
        "dml_effect": "DML ($)", "dml_ci_low": "DML IC bas", "dml_ci_high": "DML IC haut", "dml_pct": "DML (%)"})
    st.dataframe(tab.style.format({c: "{:,.0f}" for c in tab.columns if c not in ("Traitement", "DML (%)")}
                                  | {"DML (%)": "{:+.2f}"}, na_rep="—"),
                 width="stretch", hide_index=True)

with t3:
    st.markdown(
        "Trois tests cherchent à **invalider** l'estimation (régression ajustée, 20 simulations chacun) :\n"
        "- **Traitement placebo** : traitement permuté au hasard → l'effet doit tomber à ≈ 0.\n"
        "- **Cause commune aléatoire** : ajout d'un faux facteur de confusion → l'effet ne doit pas bouger.\n"
        "- **Sous-échantillon** : 80 % des données → l'effet doit rester stable.")
    r = refs.copy()
    rel = (r.new_effect - r.estimated_effect).abs() / r.estimated_effect.abs()
    placebo = r.refuter.eq("placebo_treatment_refuter")
    r["ok"] = (placebo & (r.new_effect.abs() / r.estimated_effect.abs() < 0.05)) | (~placebo & (rel < 0.02))
    r["Résultat"] = r.ok.map({True: "✅ Réussi", False: "⚠️ À examiner"})
    r["Traitement"] = r.treatment.map(labels)
    r["Test"] = r.refuter.map({"placebo_treatment_refuter": "Traitement placebo",
                               "random_common_cause": "Cause commune aléatoire",
                               "data_subset_refuter": "Sous-échantillon 80 %"})
    passed = int(r.ok.sum())
    (st.success if passed == len(r) else st.warning)(
        f"{passed} / {len(r)} tests de réfutation réussis.", icon=":material/verified:")
    st.dataframe(r[["Traitement", "Test", "estimated_effect", "new_effect", "p_value", "Résultat"]].rename(
        columns={"estimated_effect": "Effet estimé", "new_effect": "Effet après test", "p_value": "p"}).style.format(
        {"Effet estimé": "{:,.1f}", "Effet après test": "{:,.1f}", "p": "{:.2f}"}),
        width="stretch", hide_index=True)

with t4:
    c1, c2 = st.columns(2)
    key = c1.selectbox("Traitement", cate.treatment.unique(), format_func=labels.get)
    by = c2.selectbox("Selon", ["location", "job_title"], format_func=L)
    d = cate[(cate.treatment == key) & (cate.by == by)].sort_values("effect")
    left, right = st.columns(2)
    with left:
        fig = px.scatter(d, x="effect", y="level", color_discrete_sequence=[col["series"][0]],
                         error_x=d.ci_high - d.effect, error_x_minus=d.effect - d.ci_low,
                         labels={"effect": "Effet ($)", "level": ""})
        fig.update_traces(marker_size=10, hovertemplate="%{y} : %{x:,.0f} $<extra></extra>")
        fig.update_layout(title="Effet en dollars")
        ui.show(ui.style(fig, height=440, legend=False))
    with right:
        fig = px.scatter(d, x="effect_pct", y="level", color_discrete_sequence=[col["series"][1]],
                         error_x=d.pct_ci_high - d.effect_pct, error_x_minus=d.effect_pct - d.pct_ci_low,
                         labels={"effect_pct": "Effet (%)", "level": ""})
        fig.update_traces(marker_size=10, hovertemplate="%{y} : %{x:+.2f} %<extra></extra>")
        fig.update_layout(title="Effet en pourcentage")
        ui.show(ui.style(fig, height=440, legend=False))
    spread_usd = d.effect.max() / d.effect.min()
    spread_pct = d.effect_pct.max() - d.effect_pct.min()
    st.markdown(
        f"En dollars, l'effet varie d'un facteur **{spread_usd:.1f}** entre groupes. En pourcentage, "
        f"l'écart n'est que de **{spread_pct:.1f} point(s)**. L'effet est donc surtout un "
        "**pourcentage stable** : il pèse plus en dollars là où les salaires sont élevés.")
    st.caption("Estimation : LinearDML (EconML), le groupe sert de modificateur d'effet, modèles "
               "de nuisance LightGBM, cross-fitting à 3 plis, IC 95 %.")
