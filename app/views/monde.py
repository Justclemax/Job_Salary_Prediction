import unicodedata

import pandas as pd
import plotly.graph_objects as go

import ui
from salary.sources import stackoverflow as so
from ui import st

S = ui.load_json("reports/world/summary.json")
DAG_EDGES = ui.load_json("reports/world/dag_edges.json")
col = ui.colors()
c_syn, c_usd, c_ppp = col["series"]
fmt = lambda v, d=1: f"{v:.{d}f}".replace(".", ",")  # noqa: E731
nb = lambda v: f"{v:,.0f}".replace(",", "&nbsp;")  # noqa: E731
SHORT = {"education_advanced": "Master/doctorat<br>vs licence", "large_company": "Entreprise ≥ 1 000<br>vs plus petite",
         "remote_full": "Télétravail total<br>vs présentiel", "experience_years": "+1 année<br>d'expérience"}
LBL = {"country": "Pays", "job_family": "Métier", "experience_years": "Années d'expérience",
       "education": "Diplôme", "company_size": "Taille d'entreprise", "remote": "Mode de travail",
       "age": "Tranche d'âge", "year": "Année d'enquête", "salary": "Salaire"}

by_country = ui.load_csv("world", "by_country.csv")
ate = ui.load_csv("world", "ate.csv")
cmp_c = ui.load_csv("world", "compare_countries.csv")
cmp_e = ui.load_csv("world", "compare_effects.csv")
cv = ui.load_csv("world", "model_cv.csv").set_index("model")
shares = ui.load_csv("world", "anova_shares.csv").set_index("feature")
assoc = ui.load_csv("world", "associations.csv")
synth_assoc = ui.load_csv("stats", "feature_association_pairs.csv")["value"].max()
m = ui.metrics()
synth_r2 = m["models"][m["best_model"]]["test"]["r2"]
synth_gap = ui.load_json("reports/causal/summary.json")["max_naive_vs_adjusted_gap_pct"]
real_r2 = S["best_cv"]["r2_mean"]

ui.hero(
    "Salary Lab · Données réelles",
    "Et dans le vrai monde&nbsp;?",
    f"Même étude, refaite sur <b>{nb(S['n'])} développeurs salariés</b> de {S['n_countries']} pays "
    f"(Stack Overflow Developer Survey {' et '.join(S['years'])}), avec la parité de pouvoir d'achat de la "
    "Banque mondiale. Ici, les variables sont liées entre elles : la causalité devient indispensable.",
    pills=[(f"✦ {nb(S['n'])} salaires réels", "claire"), (f"{S['n_countries']} pays", "sombre"),
           (f"PPA {S['ppp_year']} · Banque mondiale", "sombre")],
    art="world",
)

c1, c2, c3, c4 = st.columns(4)
with c1:
    ui.kpi("Pouvoir prédictif", f"R² {fmt(real_r2, 2)}", f"en validation croisée, contre {fmt(synth_r2, 3)} "
           "sur les données synthétiques", "lavande", badge="réel", badge_cls="lavande")
with c2:
    ui.kpi("Poids du pays", f"{shares.loc['country', 'share_explained_pct']:.0f} %",
           "de la variance expliquée du log-salaire", "jaune", badge="facteur n°1", badge_cls="jaune")
with c3:
    ui.kpi("Liens entre variables", fmt(S["max_association"], 2), f"association max (âge ↔ expérience), "
           f"contre {fmt(synth_assoc, 3)} en synthétique", "menthe", badge="confusion", badge_cls="vert")
with c4:
    ui.kpi("Biais max de l'écart naïf", f"{S['max_naive_vs_adjusted_gap_pct']:.0f} %",
           f"d'écart avec l'effet ajusté, contre {fmt(synth_gap)} % en synthétique", "gris",
           badge="corrigé par DML", badge_cls="gris")

t1, t2, t3, t4, t5 = st.tabs(["Pays et pouvoir d'achat", "Synthétique vs réel", "Causalité",
                              "Modèle", "Simulateur mondial"])

# ----------------------------------------------------------------------------- pays
with t1:
    d = by_country.sort_values("median_usd")
    fig = go.Figure()
    for _, r in d.iterrows():
        fig.add_trace(go.Scatter(x=[r.median_usd, r.median_ppp], y=[r.country] * 2, mode="lines",
                                 line=dict(color=col["muted"], width=2), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=d.median_usd, y=d.country, mode="markers", name="Dollars courants",
                             marker=dict(size=11, color=c_usd, line=dict(width=2, color=col["surface"])),
                             customdata=d[["n"]], hovertemplate="%{y} : %{x:,.0f} $ (n = %{customdata[0]})<extra>USD</extra>"))
    fig.add_trace(go.Scatter(x=d.median_ppp, y=d.country, mode="markers", name="Pouvoir d'achat (PPA)",
                             marker=dict(size=11, color=c_ppp, symbol="diamond", line=dict(width=2, color=col["surface"])),
                             hovertemplate="%{y} : %{x:,.0f} $ PPA<extra>PPA</extra>"))
    fig.update_layout(title="Salaire médian d'un développeur : en dollars et à pouvoir d'achat égal",
                      xaxis_title="Salaire annuel médian ($)", xaxis_tickformat=",.0f")
    ui.show(ui.style(fig, height=760))
    us = by_country.set_index("country").loc["États-Unis"]
    ind = by_country.set_index("country").loc["Inde"]
    fr = by_country.set_index("country").loc["France"]
    ui.reading(
        f"aux taux de change, un développeur américain gagne {fmt(us.median_usd / ind.median_usd)} fois plus qu'un "
        f"développeur indien ({ui.money(us.median_usd)} contre {ui.money(ind.median_usd)}). À pouvoir d'achat égal, "
        f"l'écart tombe à {fmt(us.median_ppp / ind.median_ppp)} fois. En France, la médiane est de "
        f"{ui.money(fr.median_usd)}, soit {ui.money(fr.median_ppp)} en PPA.")
    jobs = ui.load_csv("world", "by_job.csv").sort_values("median_usd")
    fig = go.Figure(go.Bar(x=jobs.median_usd, y=jobs.job_family, orientation="h", marker_color=c_syn,
                           text=[ui.money(v) for v in jobs.median_usd], textposition="outside",
                           customdata=jobs[["n", "median_experience"]],
                           hovertemplate="%{y} : %{x:,.0f} $<br>n = %{customdata[0]} · expérience médiane "
                                         "%{customdata[1]} ans<extra></extra>"))
    fig.update_layout(title="Salaire médian par métier (tous pays)", xaxis_range=[0, jobs.median_usd.max() * 1.2])
    ui.show(ui.style(fig, height=560, legend=False))
    ui.warn_box("Ce classement mélange les pays",
                "Les managers et les architectes ont aussi plus d'expérience, et la part d'Américains varie selon le "
                "métier. L'onglet <b>Causalité</b> sépare ces effets.")

# ----------------------------------------------------------------------------- synthétique vs réel
with t2:
    d = cmp_c.sort_values("real_ratio_usa")
    fig = go.Figure()
    for name, c, colr in [("Synthétique", "synthetic_ratio_usa", c_syn), ("Réel, dollars", "real_ratio_usa", c_usd),
                          ("Réel, PPA", "real_ppp_ratio_usa", c_ppp)]:
        fig.add_trace(go.Bar(x=d[c], y=d.country, orientation="h", name=name, marker_color=colr,
                             hovertemplate="%{y} : %{x:.0f} % du salaire américain<extra>" + name + "</extra>"))
    fig.add_vline(x=100, line_dash="dot", line_color=col["muted"])
    fig.update_layout(title="Salaire médian en % de celui des États-Unis", barmode="group",
                      xaxis_title="% du salaire médian américain")
    ui.show(ui.style(fig, height=520))
    india = cmp_c.set_index("country").loc["Inde"]
    ui.reading(f"le jeu synthétique écrase les écarts entre pays. L'Inde y est à {india.synthetic_ratio_usa:.0f} % du "
               f"salaire américain, contre {india.real_ratio_usa:.0f} % en réalité (en dollars). En pouvoir d'achat, "
               f"l'Inde remonte à {india.real_ppp_ratio_usa:.0f} %.")

    d = cmp_e.assign(label=cmp_e.treatment.map(SHORT))
    fig = go.Figure()
    fig.add_trace(go.Bar(x=d.label, y=d.synthetic_dml_pct, name="Synthétique (Double ML)", marker_color=c_syn,
                         hovertemplate="%{x} : %{y:+.1f} %<extra>Synthétique</extra>"))
    fig.add_trace(go.Bar(x=d.label, y=d.real_naive_pct, name="Réel, écart naïf", marker_color=c_usd,
                         hovertemplate="%{x} : %{y:+.1f} %<extra>Réel naïf</extra>"))
    fig.add_trace(go.Bar(x=d.label, y=d.real_dml_pct, name="Réel, effet ajusté (Double ML)", marker_color=c_ppp,
                         error_y=dict(type="data", symmetric=False, array=d.real_dml_pct_ci_high - d.real_dml_pct,
                                      arrayminus=d.real_dml_pct - d.real_dml_pct_ci_low, color=col["muted"]),
                         hovertemplate="%{x} : %{y:+.1f} %<extra>Réel ajusté</extra>"))
    fig.update_layout(title="Effet sur le salaire (%) : synthétique vs réel", barmode="group",
                      yaxis_title="Effet sur le salaire (%)", yaxis_ticksuffix=" %")
    ui.show(ui.style(fig, height=460))
    e = cmp_e.set_index("treatment")
    ui.reading(
        f"dans les données réelles, l'écart naïf trompe. Un Master ou un doctorat semble ne rien rapporter "
        f"({ui.pct(e.loc['education_advanced', 'real_naive_pct'])}), mais à pays et âge égaux, il rapporte "
        f"{ui.pct(e.loc['education_advanced', 'real_dml_pct'])}. Le télétravail total semble valoir "
        f"{ui.pct(e.loc['remote_full', 'real_naive_pct'], digits=0)}, mais l'essentiel vient du fait que les "
        f"télétravailleurs sont souvent américains. L'effet ajusté est de {ui.pct(e.loc['remote_full', 'real_dml_pct'], digits=0)}. "
        f"Le jeu synthétique surestime nettement le diplôme ({ui.pct(e.loc['education_advanced', 'synthetic_dml_pct'])}).")

    a = assoc.head(8).iloc[::-1]
    pair = [f"{LBL[x]} ↔ {LBL[y]}" for x, y in zip(a.feature_a, a.feature_b, strict=True)]
    fig = go.Figure(go.Bar(x=a.value, y=pair, orientation="h", marker_color=c_syn,
                           text=[fmt(v, 2) for v in a.value], textposition="outside",
                           customdata=a[["measure"]], hovertemplate="%{y} : %{x:.2f} (%{customdata[0]})<extra></extra>"))
    fig.update_layout(title="Les variables réelles sont liées entre elles (association de 0 à 1)",
                      xaxis_range=[0, 1])
    ui.show(ui.style(fig, height=380, legend=False))
    ui.explain_box("Pourquoi c'est la différence la plus importante",
                   f"Dans le jeu synthétique, aucune association ne dépasse {fmt(synth_assoc, 3)}. Comparer deux groupes y "
                   "revient donc à mesurer un effet causal. Dans les vraies données, le pays est lié au diplôme et au "
                   "télétravail, et l'âge à l'expérience. Un écart brut mélange alors plusieurs causes, et il faut "
                   "ajuster pour les séparer.")

# ----------------------------------------------------------------------------- causalité
with t3:
    left, right = st.columns([3, 2])
    with left:
        fg, bg = ("#f0efec", "#1a1a19") if ui.mode() == "dark" else ("#0b0b0b", "#fcfcfb")
        nodes = sorted({n for e in DAG_EDGES for n in e} - {"salary"})
        dot = (f'digraph {{ rankdir=TB; bgcolor="transparent"; nodesep=0.3; ranksep=0.45; '
               f'node [shape=box style="rounded,filled" fillcolor="{bg}" color="{c_syn}" fontcolor="{fg}" '
               f'fontname="Helvetica" fontsize=14 margin="0.15,0.06"]; edge [color="{c_syn}" penwidth=1.4];\n'
               + "\n".join(f'  "{n}" [label="{LBL[n]}"];' for n in nodes)
               + f'\n  "salary" [label="Salaire" fillcolor="{c_syn}" fontcolor="white"];\n'
               + "\n".join(f'  "{a}" -> "{b}";' for a, b in DAG_EDGES) + "\n}")
        st.graphviz_chart(dot, width="stretch")
    with right:
        st.markdown(
            "**Graphe supposé :** le pays et l'âge influencent le diplôme, l'expérience, le métier et le mode de "
            "travail, qui influencent ensuite le salaire.\n\n"
            "**Identification :** pour chaque traitement, l'ensemble d'ajustement bloque les chemins *backdoor* "
            "sans contrôler les médiateurs. On n'ajuste pas sur le métier pour mesurer l'effet du diplôme, car le "
            "diplôme mène au métier.\n\n"
            "**Estimation :** écart naïf, régression OLS (DoWhy), puis Double ML (EconML, LightGBM).")
    t = ate.copy()
    t["Effet naïf"] = t.naive_pct.map(ui.pct)
    t["Effet ajusté (DML)"] = t.dml_pct.map(ui.pct)
    t["IC 95 %"] = [f"[{ui.pct(a)} ; {ui.pct(b)}]" for a, b in zip(t.dml_pct_ci_low, t.dml_pct_ci_high, strict=True)]
    t["En dollars"] = [ui.money(v, True) + (f" {u}" if isinstance(u, str) else "") for v, u in zip(t.dml_effect, t.unit, strict=True)]
    t["Ajusté sur"] = t.adjustment_set.map(lambda s: ", ".join(LBL[x.strip()] for x in s.split(",")))
    st.dataframe(t[["label", "n", "Effet naïf", "Effet ajusté (DML)", "IC 95 %", "En dollars", "Ajusté sur"]]
                 .rename(columns={"label": "Traitement"}), hide_index=True, width="stretch")
    refs = ui.load_csv("world", "refutations.csv")
    plc = refs[refs.refuter == "placebo_treatment_refuter"]
    worst = float((plc.new_effect.abs() / plc.estimated_effect.abs()).max() * 100)
    stable = refs[refs.refuter != "placebo_treatment_refuter"]
    drift = float(((stable.new_effect - stable.estimated_effect).abs() / stable.estimated_effect.abs()).max() * 100)
    ui.reading(f"les {len(refs)} tests de réfutation passent. Un traitement placebo donne au plus {fmt(worst)} % de "
               f"l'effet estimé, et l'ajout d'une cause aléatoire ou un sous-échantillon déplacent l'effet d'au plus "
               f"{fmt(drift)} %.")

    cate = ui.load_csv("world", "cate_country.csv")
    which = st.radio("Effet par pays", ["education_advanced", "remote_full"], horizontal=True,
                     format_func=lambda k: ate.set_index("treatment").loc[k, "label"])
    d = cate[cate.treatment == which].sort_values("effect_pct")
    fig = go.Figure(go.Scatter(
        x=d.effect_pct, y=d.country, mode="markers", marker=dict(size=10, color=c_ppp),
        error_x=dict(type="data", symmetric=False, array=d.pct_ci_high - d.effect_pct,
                     arrayminus=d.effect_pct - d.pct_ci_low, color=col["muted"]),
        customdata=d[["n", "pct_ci_low", "pct_ci_high"]],
        hovertemplate="%{y} : %{x:+.1f} %<br>IC 95 % [%{customdata[1]:+.1f} ; %{customdata[2]:+.1f}] · "
                      "n = %{customdata[0]}<extra></extra>"))
    fig.add_vline(x=0, line_dash="dot", line_color=col["muted"])
    fig.update_layout(title="Effet causal par pays (Double ML), avec IC 95 %", xaxis_ticksuffix=" %")
    ui.show(ui.style(fig, height=460, legend=False))
    ui.warn_box("Limites d'une analyse causale sur données d'enquête",
                "Le graphe ne contient pas le talent, la ville (San Francisco ou une petite ville), le genre ni la "
                "capacité à négocier. Si ces facteurs jouent sur le traitement et sur le salaire, les effets restent "
                "biaisés. Les répondants Stack Overflow ne représentent pas tous les développeurs, et les salaires "
                "sont déclarés. Lisez ces effets comme des ordres de grandeur plausibles, pas comme des preuves.")

# ----------------------------------------------------------------------------- modèle
with t4:
    d = cv.reset_index()
    tbl = d[["label"]].copy()
    tbl["R² (CV)"] = [f"{fmt(a, 3)} ± {fmt(b, 3)}" for a, b in zip(d.r2_mean, d.r2_std, strict=True)]
    tbl["RMSE"] = d.rmse_mean.map(ui.money)
    tbl["Erreur médiane"] = d.medae_mean.map(ui.money)
    tbl["MAPE"] = d.mape_mean.map(lambda v: f"{fmt(v)} %")
    tbl["À ±10 %"] = d.within_10pct_mean.map(lambda v: f"{fmt(v)} %")
    st.dataframe(tbl.rename(columns={"label": "Modèle"}), hide_index=True, width="stretch")
    ui.reading(
        f"le pays seul donne déjà un R² de {fmt(cv.loc['country_median', 'r2_mean'], 2)}. Toutes les autres "
        f"variables ne le portent qu'à {fmt(real_r2, 2)}, et l'erreur médiane reste de "
        f"{ui.money(S['best_cv']['medae_mean'])}. C'est le vrai visage d'un salaire : une grande part dépend de "
        f"facteurs absents de l'enquête. Sur les données synthétiques, le R² atteignait {fmt(synth_r2, 3)}.")
    left, right = st.columns(2)
    with left:
        shap = ui.load_csv("world", "shap_importance.csv").iloc[::-1]
        fig = go.Figure(go.Bar(x=shap.typical_effect_pct, y=[LBL[f] for f in shap.feature], orientation="h",
                               marker_color=c_syn, text=[f"± {fmt(v)} %" for v in shap.typical_effect_pct],
                               textposition="outside", hovertemplate="%{y} : ± %{x:.1f} %<extra></extra>"))
        fig.update_layout(title="Effet typique de chaque variable (SHAP, CatBoost)",
                          xaxis_range=[0, shap.typical_effect_pct.max() * 1.25], xaxis_ticksuffix=" %")
        ui.show(ui.style(fig, height=380, legend=False))
    with right:
        oof = ui.load_parquet("world", "oof_sample.parquet").sample(2500, random_state=42)
        fig = go.Figure(go.Scattergl(x=oof.salary_usd, y=oof.pred, mode="markers",
                                     marker=dict(size=4, color=c_syn, opacity=0.35),
                                     customdata=oof[["country", "job_family"]],
                                     hovertemplate="%{customdata[0]} · %{customdata[1]}<br>réel %{x:,.0f} $<br>"
                                                   "prédit %{y:,.0f} $<extra></extra>"))
        lim = [oof[["salary_usd", "pred"]].min().min(), oof[["salary_usd", "pred"]].max().max()]
        fig.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(color=c_usd, dash="dot"), hoverinfo="skip"))
        ticks = dict(tickvals=[1e3, 1e4, 1e5, 1e6], ticktext=["1 k$", "10 k$", "100 k$", "1 M$"])
        fig.update_layout(title="Prédit vs réel (hors échantillon, échelle log)", xaxis_type="log", yaxis_type="log",
                          xaxis_title="Salaire réel", yaxis_title="Salaire prédit")
        fig.update_xaxes(**ticks)
        fig.update_yaxes(**ticks)
        ui.show(ui.style(fig, height=380, legend=False))

# ----------------------------------------------------------------------------- simulateur
def _key(s: str) -> str:
    """Sort French country names ignoring accents (États-Unis between Espagne and Finlande)."""
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode()


MODEL_NAMES = {"catboost": "CatBoost", "lightgbm": "LightGBM", "ridge": "Ridge"}
_AU = {"Brésil", "Canada", "Danemark", "Mexique", "Portugal", "Royaume-Uni"}


def dans(country: str) -> str:
    """French preposition: en France, au Canada, aux États-Unis, à Singapour."""
    if country in {"États-Unis", "Pays-Bas"}:
        return f"aux {country}"
    if country == "Singapour":
        return f"à {country}"
    return f"{'au' if country in _AU else 'en'} {country}"


def local_money(v: float, symbol: str) -> str:
    step = 1000 if v >= 1e6 else 100 if v >= 1e4 else 10
    return f"{round(v / step) * step:,.0f}".replace(",", "\u202f") + f"\u00a0{symbol}"


@st.dialog("Votre salaire estimé", width="large")
def estimate_popup(country: str, job: str, edu: str, exp: int, remote: str, same: pd.DataFrame) -> None:
    row = same.set_index("country").loc[country]
    c = by_country.set_index("country").loc[country]
    yearly = row.pred_usd * c.fx
    rank = int((same.pred_usd > row.pred_usd).sum()) + 1
    rank_ppp = int((same.pred_ppp > row.pred_ppp).sum()) + 1
    st.html(f"<div class='estim'><div class='pays'>{country} · {c.currency}</div>"
            f"<div class='profil'>{job} · {edu} · {exp} an{'s' if exp > 1 else ''} d'expérience · {remote.lower()}</div>"
            f"<div class='montant'>{local_money(yearly, c.currency_symbol)}<span> / an</span></div>"
            f"<div class='mois'>soit environ <b>{local_money(yearly / 12, c.currency_symbol)} par mois</b> "
            f"(rémunération annuelle brute déclarée, divisée par 12)</div></div>")
    k1, k2, k3 = st.columns(3)
    with k1:
        rate = "" if c.currency == "USD" else f"au taux moyen {c.fx_year} : 1 $ = {fmt(c.fx, 2)} {c.currency_symbol}"
        ui.kpi("En dollars", ui.money(row.pred_usd), rate or "monnaie du pays", "lavande")
    with k2:
        ui.kpi("Pouvoir d'achat", ui.money(row.pred_ppp), "en dollars PPA, à prix américains", "menthe")
    with k3:
        sup = lambda r: f"{r}<sup>{'er' if r == 1 else 'e'}</sup>"  # noqa: E731
        ui.kpi("Rang mondial", f"{sup(rank)} / {len(same)}",
               f"en dollars · {sup(rank_ppp)} en pouvoir d'achat", "jaune")
    gap = (row.pred_usd / c.median_usd - 1) * 100
    ui.reading(f"le salaire médian de tous les développeurs salariés {dans(country)} est de "
               f"{local_money(c.median_usd * c.fx, c.currency_symbol)} (n = {nb(c.n)}). Ce profil se situe "
               f"{ui.pct(abs(gap), signed=False, digits=0)} {'au-dessus' if gap >= 0 else 'en dessous'} de cette médiane.")
    st.caption(f"Estimation du modèle {MODEL_NAMES.get(S['grid_model'], S['grid_model'])} entraîné sur les salaires déclarés "
               f"(Stack Overflow {' et '.join(S['years'])}), pour une entreprise de {S['grid_company_size']} salariés. "
               f"Conversion : taux de change annuel moyen {c.fx_year} de la Banque mondiale. "
               f"Erreur médiane du modèle : environ {ui.money(S['best_cv']['medae_mean'])} tous pays confondus. "
               "Une estimation, pas une offre.")


with t5:
    grid = ui.load_parquet("world", "prediction_grid.parquet")
    countries = sorted(grid.country.unique(), key=_key)
    edu_levels = [e for e in so.EDUCATION_LEVELS if e in set(grid.education)]
    with st.container(border=True, key="sim_monde"):
        st.markdown("**Votre profil** · choisissez un pays et un métier, puis ouvrez l'estimation dans la "
                    "monnaie du pays.")
        r1 = st.columns([1, 1.3, 1.2])
        country = r1[0].selectbox("Pays", countries, index=countries.index("France"))
        job = r1[1].selectbox("Métier", so.LEVELS["job_family"],
                              index=so.LEVELS["job_family"].index("Data scientist / IA"))
        edu = r1[2].select_slider("Diplôme", edu_levels, value="Master")
        r2 = st.columns([1, 1.3, 1.2], vertical_alignment="bottom")
        exp = r2[0].select_slider("Expérience (ans)", sorted(grid.experience_years.unique()), value=5)
        remote = r2[1].radio("Mode de travail", so.REMOTE_LEVELS, index=1, horizontal=True)
        clicked = r2[2].button("Voir mon salaire estimé", type="primary", icon=":material/payments:",
                               width="stretch")

    same = grid[(grid.job_family == job) & (grid.education == edu) & (grid.experience_years == exp)
                & (grid.remote == remote)]
    unit = st.segmented_control("Comparer les pays", ["Dollars courants", "Pouvoir d'achat (PPA)"],
                                default="Dollars courants") or "Dollars courants"
    key = "pred_ppp" if unit == "Pouvoir d'achat (PPA)" else "pred_usd"
    d = same.sort_values(key)
    hi = c_ppp if key == "pred_ppp" else c_usd
    is_sel = d.country == country
    fig = go.Figure(go.Bar(
        x=d[key], y=d.country, orientation="h",
        marker=dict(color=[hi if s else col["muted"] for s in is_sel],
                    opacity=[1 if s else 0.35 for s in is_sel], cornerradius=4),
        text=[ui.money(v) for v in d[key]], textposition="outside",
        hovertemplate="%{y} : %{x:,.0f} $<br><i>cliquez pour le détail</i><extra></extra>"))
    fig.update_layout(title=f"Même profil dans {len(d)} pays ({'dollars PPA' if key == 'pred_ppp' else 'dollars courants'})"
                            f" · {country} en couleur",
                      xaxis_range=[0, d[key].max() * 1.2], xaxis_tickformat=",.0f", clickmode="event+select")
    fig.update_yaxes(tickfont=dict(size=13))
    event = ui.show(ui.style(fig, height=760, legend=False), key="sim_monde_chart",
                    on_select="rerun", selection_mode="points")
    st.caption("Astuce : cliquez sur la barre d'un pays pour ouvrir son estimation.")

    picked = None
    if event and event.selection.points:
        picked = event.selection.points[0].get("y")
    sig = (picked, job, edu, exp, remote)
    if clicked:
        estimate_popup(country, job, edu, exp, remote, same)
    elif picked in set(same.country) and st.session_state.get("_sim_monde_pick") != sig:
        st.session_state["_sim_monde_pick"] = sig
        estimate_popup(picked, job, edu, exp, remote, same)

    top, bottom = d.iloc[-1], d.iloc[0]
    me = d.set_index("country").loc[country, key]
    ui.reading(f"pour ce profil, l'estimation la plus haute est {dans(top.country)} ({ui.money(top[key])}) et la plus "
               f"basse {dans(bottom.country)} ({ui.money(bottom[key])}). {country} : {ui.money(me)}.")

st.write("")
st.caption(f"Sources : {S['source']} (survey.stackoverflow.co). {S['ppp_source']}, année {S['ppp_year']}. "
           "Analyse : salariés uniquement, salaires extrêmes retirés par pays (2 % de chaque côté).")
