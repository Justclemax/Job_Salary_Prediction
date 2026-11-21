"""Statistical analysis of salary.

With n = 250,000 almost every p-value is ~0, so every test is reported with an
effect size and practical significance is judged on the effect size.

    uv run python -m salary.stats      # writes reports/stats/*
"""

import json
from itertools import combinations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from salary import config as cfg
from salary.data import load_raw

OUT = cfg.REPORTS_DIR / "stats"

# Reference levels for the regression (lowest-paid / most common baseline).
REFERENCE = {
    "job_title": "Data Analyst", "education_level": "High School", "company_size": "Startup",
    "location": "India", "remote_work": "No", "industry": "Retail",
}


def eta2_label(v: float) -> str:
    """Cohen's (1988) thresholds for eta² / epsilon²."""
    return "négligeable" if v < 0.01 else "faible" if v < 0.06 else "moyen" if v < 0.14 else "fort"


def r_label(r: float) -> str:
    r = abs(r)
    return "négligeable" if r < 0.1 else "faible" if r < 0.3 else "moyen" if r < 0.5 else "fort"


def target_summary(df: pd.DataFrame, seed: int = cfg.SEED) -> pd.DataFrame:
    rows = []
    for name, s in [("salary", df[cfg.TARGET]), ("log_salary", np.log(df[cfg.TARGET]))]:
        sample = s.sample(5000, random_state=seed)
        k2, p_k2 = stats.normaltest(s)
        jb, p_jb = stats.jarque_bera(s)
        sw, p_sw = stats.shapiro(sample)
        rows.append({
            "variable": name, "n": len(s), "mean": s.mean(), "std": s.std(), "min": s.min(),
            "q1": s.quantile(.25), "median": s.median(), "q3": s.quantile(.75), "max": s.max(),
            "skewness": stats.skew(s), "excess_kurtosis": stats.kurtosis(s),
            "dagostino_k2": k2, "dagostino_p": p_k2, "jarque_bera": jb, "jarque_bera_p": p_jb,
            "shapiro_w_5000": sw, "shapiro_p_5000": p_sw,
        })
    return pd.DataFrame(rows)


def categorical_tests(df: pd.DataFrame) -> pd.DataFrame:
    """One-way ANOVA (+eta²), Kruskal-Wallis (+epsilon²) and Levene for each categorical."""
    y = df[cfg.TARGET]
    n = len(df)
    sst = ((y - y.mean()) ** 2).sum()
    rows = []
    for col in cfg.CATEGORICAL:
        groups = [g.to_numpy() for _, g in y.groupby(df[col])]
        f, p_f = stats.f_oneway(*groups)
        ssb = sum(len(g) * (g.mean() - y.mean()) ** 2 for g in groups)
        h, p_h = stats.kruskal(*groups)
        lev, p_lev = stats.levene(*groups, center="median")
        eta2 = ssb / sst
        eps2 = h / (n - 1)
        means = y.groupby(df[col]).mean()
        rows.append({
            "feature": col, "k_groups": len(groups),
            "anova_F": f, "anova_p": p_f, "eta2": eta2, "eta2_effect": eta2_label(eta2),
            "kruskal_H": h, "kruskal_p": p_h, "epsilon2": eps2,
            "levene_W": lev, "levene_p": p_lev,
            "max_gap_means": means.max() - means.min(),
            "highest": means.idxmax(), "lowest": means.idxmin(),
        })
    return pd.DataFrame(rows).sort_values("eta2", ascending=False)


def numeric_correlations(df: pd.DataFrame) -> pd.DataFrame:
    y = df[cfg.TARGET]
    n = len(df)
    rows = []
    for col in cfg.NUMERIC:
        r, p = stats.pearsonr(df[col], y)
        z, se = np.arctanh(r), 1 / np.sqrt(n - 3)
        rho, p_rho = stats.spearmanr(df[col], y)
        slope = stats.linregress(df[col], y)
        rows.append({
            "feature": col, "pearson_r": r, "pearson_p": p,
            "r_ci_low": np.tanh(z - 1.96 * se), "r_ci_high": np.tanh(z + 1.96 * se),
            "r2": r ** 2, "spearman_rho": rho, "spearman_p": p_rho,
            "effect": r_label(r), "slope_per_unit": slope.slope,
        })
    return pd.DataFrame(rows).sort_values("pearson_r", ascending=False)


def _cramers_v(a: pd.Series, b: pd.Series) -> tuple[float, float]:
    ct = pd.crosstab(a, b)
    chi2, p, _, _ = stats.chi2_contingency(ct)
    return float(np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1)))), float(p)


def _correlation_ratio(cat: pd.Series, num: pd.Series) -> float:
    means = num.groupby(cat).mean()
    counts = num.groupby(cat).size()
    ssb = (counts * (means - num.mean()) ** 2).sum()
    return float(np.sqrt(ssb / ((num - num.mean()) ** 2).sum()))


def feature_associations(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Association between every pair of features (not the target):
    Cramér's V (cat-cat), correlation ratio eta (cat-num), |Spearman| (num-num).
    All on a 0-1 scale. Near-zero everywhere => features are independent, which
    matters for causal analysis (no confounding between them).
    """
    feats = cfg.FEATURES
    mat = pd.DataFrame(np.eye(len(feats)), index=feats, columns=feats)
    rows = []
    for a, b in combinations(feats, 2):
        ca, cb = a in cfg.CATEGORICAL, b in cfg.CATEGORICAL
        if ca and cb:
            v, p = _cramers_v(df[a], df[b])
            kind = "Cramér V"
        elif ca or cb:
            cat, num = (a, b) if ca else (b, a)
            v = _correlation_ratio(df[cat], df[num])
            p = stats.f_oneway(*[g.to_numpy() for _, g in df[num].groupby(df[cat])]).pvalue
            kind = "eta"
        else:
            rho, p = stats.spearmanr(df[a], df[b])
            v, kind = abs(rho), "|Spearman|"
        mat.loc[a, b] = mat.loc[b, a] = v
        rows.append({"feature_a": a, "feature_b": b, "measure": kind, "value": v, "p_value": p})
    return mat, pd.DataFrame(rows).sort_values("value", ascending=False)


def tukey(df: pd.DataFrame, col: str) -> pd.DataFrame:
    res = pairwise_tukeyhsd(df[cfg.TARGET], df[col], alpha=0.05)
    t = pd.DataFrame(res.summary().data[1:], columns=res.summary().data[0])
    t.insert(0, "feature", col)
    return t


def _formula(interaction: str | None = None) -> str:
    terms = list(cfg.NUMERIC) + [f"C({c}, Treatment('{REFERENCE[c]}'))" for c in cfg.CATEGORICAL]
    if interaction:
        terms.append(interaction)
    return "np.log(salary) ~ " + " + ".join(terms)


def log_linear_model(df: pd.DataFrame):
    """OLS on log(salary) with HC3 robust SEs. exp(beta)-1 = % effect, all else equal."""
    model = smf.ols(_formula(), data=df).fit(cov_type="HC3")
    ci = model.conf_int()
    coefs = pd.DataFrame({
        "term": model.params.index, "coef_log": model.params.values,
        "se_hc3": model.bse.values, "p_value": model.pvalues.values,
        "pct_effect": (np.exp(model.params.values) - 1) * 100,
        "pct_ci_low": (np.exp(ci[0].values) - 1) * 100,
        "pct_ci_high": (np.exp(ci[1].values) - 1) * 100,
    })
    coefs["feature"] = coefs["term"].str.extract(r"C\((\w+),")[0].fillna(coefs["term"])
    coefs["level"] = coefs["term"].str.extract(r"\[T\.(.+)\]")[0]
    # Type-II ANOVA (classical SEs) -> partial eta² per feature = variance decomposition.
    ols_plain = smf.ols(_formula(), data=df).fit()
    aov = anova_lm(ols_plain, typ=2)
    ss_res = aov.loc["Residual", "sum_sq"]
    aov = aov.drop(index="Residual")
    aov["partial_eta2"] = aov["sum_sq"] / (aov["sum_sq"] + ss_res)
    aov["share_explained_pct"] = aov["sum_sq"] / aov["sum_sq"].sum() * 100
    aov.index = aov.index.str.extract(r"C\((\w+),")[0].fillna(pd.Series(aov.index)).values
    summary = {"r2": model.rsquared, "adj_r2": model.rsquared_adj, "n": int(model.nobs),
               "aic": model.aic, "bic": model.bic, "residual_std_log": float(np.sqrt(model.scale))}
    return coefs, aov.reset_index(names="feature").sort_values("sum_sq", ascending=False), summary


def separability_tests(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nested F-tests: does adding an interaction improve the log-linear model?
    If not, effects are purely multiplicative (same % effect for every group).
    """
    base = smf.ols(_formula(), data=df).fit()
    rows = []
    for inter in ["experience_years:C(job_title)", "C(education_level):C(job_title)",
                  "C(location):C(job_title)", "experience_years:C(education_level)"]:
        full = smf.ols(_formula(inter), data=df).fit()
        f, p, ddf = full.compare_f_test(base)
        if f < 0:  # numerical noise when the interaction adds nothing (delta R² ~ 0)
            f, p = 0.0, 1.0
        rows.append({"interaction": inter, "F": f, "p_value": p, "df_added": ddf,
                     "delta_r2": full.rsquared - base.rsquared})
    return pd.DataFrame(rows)


def run_all(df: pd.DataFrame | None = None) -> dict:
    df = load_raw() if df is None else df
    OUT.mkdir(parents=True, exist_ok=True)
    target_summary(df).to_csv(OUT / "target_summary.csv", index=False)
    cat = categorical_tests(df)
    cat.to_csv(OUT / "categorical_tests.csv", index=False)
    num = numeric_correlations(df)
    num.to_csv(OUT / "numeric_correlations.csv", index=False)
    mat, pairs = feature_associations(df)
    mat.to_csv(OUT / "feature_association_matrix.csv")
    pairs.to_csv(OUT / "feature_association_pairs.csv", index=False)
    pd.concat([tukey(df, c) for c in ["education_level", "company_size", "remote_work"]]).to_csv(
        OUT / "tukey_posthoc.csv", index=False)
    coefs, aov, summary = log_linear_model(df)
    coefs.to_csv(OUT / "log_linear_coefficients.csv", index=False)
    aov.to_csv(OUT / "anova_type2.csv", index=False)
    sep = separability_tests(df)
    sep.to_csv(OUT / "separability_tests.csv", index=False)
    out = {
        "log_linear": summary,
        "max_feature_association": float(pairs["value"].max()),
        "strongest_categorical": cat.iloc[0]["feature"],
        "strongest_numeric": num.iloc[0]["feature"],
    }
    (OUT / "summary.json").write_text(json.dumps(out, indent=2, default=float))
    return out


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2, default=float))
