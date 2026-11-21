import numpy as np

import ui
from salary import config as cfg
from salary.data import DataValidationError, validate
from salary.explain import explain_one
from ui import st

ui.header("Prédiction de salaire",
          "Renseignez un profil : le modèle retenu estime le salaire annuel, avec un intervalle à "
          "80 % et l'explication de chaque facteur.", art="prediction", kicker="Machine learning")
m = ui.metrics()
best, shap_name = m["best_model"], m["shap_model"]
offset = m["interval"].get("conformal_offset", 0.0)
L = ui.feature_label
col = ui.colors()
LV = cfg.CATEGORY_LEVELS

with st.form("profil", border=True):
    c1, c2, c3 = st.columns(3)
    job = c1.selectbox(L("job_title"), LV["job_title"], index=LV["job_title"].index("Data Scientist"))
    loc = c2.selectbox(L("location"), LV["location"], index=LV["location"].index("Germany"))
    ind = c3.selectbox(L("industry"), LV["industry"], index=LV["industry"].index("Technology"))
    c1, c2, c3 = st.columns(3)
    edu = c1.select_slider(L("education_level"), LV["education_level"], value="Master")
    size = c2.select_slider(L("company_size"), LV["company_size"], value="Medium")
    remote = c3.radio(L("remote_work"), LV["remote_work"], index=1, horizontal=True,
                      format_func={"No": "Non", "Hybrid": "Hybride", "Yes": "Oui"}.get)
    c1, c2, c3 = st.columns(3)
    exp = c1.slider(L("experience_years"), *cfg.NUMERIC_RANGES["experience_years"], value=5)
    skills = c2.slider(L("skills_count"), *cfg.NUMERIC_RANGES["skills_count"], value=8)
    certs = c3.slider(L("certifications"), *cfg.NUMERIC_RANGES["certifications"], value=1)
    submitted = st.form_submit_button("Estimer le salaire", type="primary", icon=":material/calculate:")

row = ui.pd.DataFrame([{"job_title": job, "experience_years": exp, "education_level": edu,
                        "skills_count": skills, "industry": ind, "company_size": size,
                        "location": loc, "remote_work": remote, "certifications": certs}])[cfg.FEATURES]
try:
    validate(row, require_target=False)
except DataValidationError as e:
    st.error(f"Profil invalide : {e}")
    st.stop()

model = ui.load_model(best)
q_low, q_high = ui.load_model("quantile_low"), ui.load_model("quantile_high")
pred = float(model.predict(row)[0])
low = max(0.0, float(q_low.predict(row)[0]) - offset)
high = float(q_high.predict(row)[0]) + offset
low, high = min(low, pred), max(high, pred)

c = st.columns([2, 1, 1])
c[0].metric(f"Salaire estimé — {ui.model_label(best)}", ui.money(pred))
c[1].metric("Borne basse (10 %)", ui.money(low))
c[2].metric("Borne haute (90 %)", ui.money(high))
st.caption(f"Intervalle de prédiction à 80 % (régression quantile conformalisée) : couverture mesurée sur "
           f"le test {m['interval']['coverage']:.1f} %.".replace(".", ",", 1))

left, right = st.columns(2)
with left:
    ex = explain_one(ui.load_model(shap_name), row)
    ex = ex.sort_values("pct_effect")
    lab = [f"{L(f)} = {v}" for f, v in zip(ex.feature, ex.value, strict=False)]
    neg = col["neg"]
    fig = ui.go.Figure(ui.go.Bar(
        x=ex.pct_effect, y=lab, orientation="h",
        marker_color=[col["series"][0] if v >= 0 else neg for v in ex.pct_effect],
        text=[ui.pct(v) for v in ex.pct_effect], textposition="outside",
        hovertemplate="%{y} : %{x:+.2f} %<extra></extra>"))
    fig.add_vline(x=0, line_width=1, line_color=col["muted"])
    span = max(abs(ex.pct_effect.min()), abs(ex.pct_effect.max())) * 1.45
    fig.update_layout(title=f"Pourquoi ce salaire ? (référence {ui.money(ex.attrs['base_salary'])})",
                      xaxis_title="Effet multiplicatif (%)", xaxis_range=[-span, span])
    ui.show(ui.style(fig, height=420, legend=False))
    ui.reading("en violet, ce qui fait monter le salaire ; en ocre, ce qui le fait baisser (valeurs SHAP, "
               f"modèle {ui.model_label(shap_name)}). Les effets se multiplient à partir du salaire de référence.")
with right:
    grid = ui.pd.concat([row] * 21, ignore_index=True)
    grid["experience_years"] = range(21)
    p = model.predict(grid)
    lo = np.maximum(0, q_low.predict(grid) - offset)
    hi = q_high.predict(grid) + offset
    fig = ui.go.Figure()
    fig.add_scatter(x=list(grid.experience_years) + list(grid.experience_years[::-1]),
                    y=list(hi) + list(lo[::-1]), fill="toself", line=dict(width=0),
                    fillcolor=f"rgba({ui.tokens()['accent_rgb']},0.18)", name="Intervalle 80 %", hoverinfo="skip")
    fig.add_scatter(x=grid.experience_years, y=p, mode="lines", line=dict(width=2, color=col["series"][0]),
                    name="Salaire estimé", hovertemplate="%{x} ans : %{y:,.0f} $<extra></extra>")
    fig.add_scatter(x=[exp], y=[pred], mode="markers", marker=dict(size=12, color=col["series"][1],
                    line=dict(width=2, color=col["surface"])), name="Votre profil",
                    hovertemplate="%{y:,.0f} $<extra>Votre profil</extra>")
    fig.update_layout(title="Et avec plus d'expérience ?", xaxis_title="Années d'expérience",
                      yaxis_title="Salaire ($)", hovermode="x unified")
    ui.show(ui.style(fig, height=420))

with st.expander("Prédiction de chaque modèle"):
    out = []
    for name in m["models"]:
        if (cfg.MODELS_DIR / f"{name}.joblib").exists():
            out.append({"Modèle": ui.model_label(name), "Salaire prédit": float(ui.load_model(name).predict(row)[0]),
                        "RMSE test": m["models"][name]["test"]["rmse"]})
    st.dataframe(ui.pd.DataFrame(out).style.format({"Salaire prédit": "{:,.0f} $", "RMSE test": "{:,.0f} $"}),
                 width="stretch", hide_index=True)
