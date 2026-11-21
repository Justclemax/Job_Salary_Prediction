"""Shared helpers for the Streamlit app: theme-aware colors, cached loaders, formatting."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from salary import config as cfg
from salary.models import DISPLAY_NAMES

LABELS = cfg.LABELS_FR

# Chart palettes, checked with a palette validator (lightness band, chroma, color-blind
# separation >= 9 on ALL pairs, contrast >= 3:1 on the card surface) for each mode.
PALETTE = {
    "light": {"series": ["#6250d6", "#b8780a", "#0e8a7a"], "muted": "#8b8b97",
              "grid": "rgba(20,20,30,0.08)", "surface": "#ffffff", "neg": "#b8780a",
              "seq": ["#ece8fd", "#c9bff8", "#a596f2", "#7d69e6", "#5a46cf", "#3f2fa3"]},
    "dark": {"series": ["#8f7cf0", "#c4801a", "#1aa37a"], "muted": "#8b8b97",
             "grid": "rgba(255,255,255,0.08)", "surface": "#16161b", "neg": "#c4801a",
             "seq": ["#221d40", "#352c6e", "#4c3ea0", "#6b58d6", "#9a88f2", "#c9bff8"]},
}

# UI tokens (cards, text, accents) per mode.
TOKENS = {
    "light": {"fond": "#f6f6f9", "carte": "#ffffff", "carte2": "#f1f0f6", "bord": "#e3e2ec",
              "texte": "#15151a", "doux": "#5d5d6c", "fort": "#000000", "accent": "#5b47d1",
              "accent_rgb": "98,80,214", "jaune": "#9a6400", "jaune_rgb": "184,120,10",
              "menthe": "#0b7a6c", "menthe_rgb": "14,138,122", "actif_bg": "#15151a", "actif_txt": "#ffffff",
              "header": "rgba(246,246,249,.85)"},
    "dark": {"fond": "#0b0b0e", "carte": "#16161b", "carte2": "#1d1d24", "bord": "#26262e",
             "texte": "#ececf1", "doux": "#9a9aa8", "fort": "#ffffff", "accent": "#b8a4ff",
             "accent_rgb": "184,164,255", "jaune": "#f2d16b", "jaune_rgb": "242,209,107",
             "menthe": "#7ee2a8", "menthe_rgb": "126,226,168", "actif_bg": "#f4f4f6", "actif_txt": "#0b0b0e",
             "header": "rgba(11,11,14,.82)"},
}


def mode() -> str:
    try:
        return "dark" if st.context.theme.type == "dark" else "light"
    except Exception:
        return "light"


def colors() -> dict:
    return PALETTE[mode()]


def money(x: float, signed: bool = False) -> str:
    s = f"{abs(x):,.0f}".replace(",", " ")
    sign = ("+" if x >= 0 else "−") if signed else ("−" if x < 0 else "")
    return f"{sign}{s} $"


def pct(x: float, signed: bool = True, digits: int = 1) -> str:
    sign = ("+" if x >= 0 else "−") if signed else ""
    return f"{sign}{abs(x):.{digits}f} %".replace(".", ",")


def style(fig: go.Figure, height: int = 380, legend: bool = True) -> go.Figure:
    c = colors()
    top = 78 if legend else 44  # leave a row for the legend under the title
    fig.update_layout(
        height=height, margin=dict(l=8, r=28, t=top, b=8),
        title=dict(y=0.985, yref="container", yanchor="top", x=0.01, xanchor="left"),
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0),
        showlegend=legend, hoverlabel=dict(namelength=-1),
        bargap=0.25, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    # Outside bar labels must never be clipped by the plot area.
    if any(getattr(tr, "type", "") == "bar" and getattr(tr, "textposition", None) == "outside" for tr in fig.data):
        fig.update_traces(cliponaxis=False, selector=dict(type="bar"))
        fig.update_layout(margin=dict(l=8, r=72, t=top, b=8))
    fig.update_xaxes(gridcolor=c["grid"], zeroline=False)
    fig.update_yaxes(gridcolor=c["grid"], zeroline=False)
    return fig


def show(fig: go.Figure, **kw) -> None:
    st.plotly_chart(fig, theme="streamlit", width="stretch",
                    config={"displaylogo": False}, **kw)


def tokens() -> dict:
    return TOKENS[mode()]


def _css(t: dict) -> str:
    return f"""
<style>
.stApp {{ background: radial-gradient(1200px 520px at 85% -120px, rgba({t['accent_rgb']},.13), transparent 60%), {t['fond']}; }}
.block-container {{ max-width: 1220px; padding-top: 4.6rem; padding-bottom: 3rem; }}
h1, h2, h3, h4, h5 {{ letter-spacing: -0.02em; }}

[data-testid="stHeader"] {{ background: {t['header']}; backdrop-filter: blur(10px); }}
[data-testid="stToolbar"] > div:first-child {{ flex: 1 1 auto !important; }}
[data-testid="stTopNav"] {{ flex: 1 1 auto !important; justify-content: center; gap: 6px; }}
[data-testid="stTopNavLink"] {{ border-radius: 999px !important; padding: 4px 14px !important; background: {t['carte']}; border: 1px solid {t['bord']}; }}
[data-testid="stTopNavLink"]:hover {{ border-color: {t['doux']}; }}
[data-testid="stTopNavLink"][aria-current="page"] {{ background: {t['actif_bg']} !important; }}
[data-testid="stTopNavLink"][aria-current="page"] p, [data-testid="stTopNavLink"][aria-current="page"] span {{ color: {t['actif_txt']} !important; font-weight: 700; }}

[role="tablist"] {{ gap: 4px; background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 999px; padding: 5px;
                    width: fit-content; max-width: 100%; overflow-x: auto; margin-bottom: .6rem; }}
[role="tablist"] .react-aria-SelectionIndicator {{ display: none; }}
[data-testid="stTab"] {{ border-radius: 999px; padding: 6px 16px !important; margin: 0 !important; }}
[data-testid="stTab"] p {{ color: {t['doux']}; }}
[data-testid="stTab"][aria-selected="true"] {{ background: {t['actif_bg']}; }}
[data-testid="stTab"][aria-selected="true"] p {{ color: {t['actif_txt']} !important; font-weight: 700; }}

[data-testid="stPlotlyChart"] {{ background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 22px; padding: 10px 8px 4px; }}
[data-testid="stExpander"] details {{ background: {t['carte']}; border-radius: 18px; border-color: {t['bord']}; }}
[data-testid="stMetric"] {{ background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 20px; padding: 16px 20px; }}
[data-testid="stForm"] {{ background: {t['carte']}; border-radius: 22px; }}

.sl-hero {{ position: relative; overflow: hidden; display: flex; align-items: center; justify-content: space-between; gap: 28px;
           border-radius: 26px; padding: 30px 36px; margin-bottom: 1.1rem; border: 1px solid {t['bord']};
           background: radial-gradient(90% 140% at 100% 0%, rgba({t['accent_rgb']},.30), transparent 55%),
                       radial-gradient(60% 120% at 0% 100%, rgba({t['jaune_rgb']},.10), transparent 60%), {t['carte']}; }}
.sl-hero .sl-text {{ flex: 1 1 auto; min-width: 0; }}
.sl-hero .surtitre {{ color: {t['doux']}; text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; font-weight: 700; }}
.sl-hero h1 {{ color: {t['texte']}; font-size: 2.35rem; line-height: 1.12; margin: .5rem 0 .8rem; padding: 0; }}
.sl-hero p {{ color: {t['doux']}; font-size: 1.03rem; line-height: 1.6; max-width: 760px; margin: 0; }}
.sl-hero .rangee {{ display: flex; flex-wrap: wrap; gap: 10px; margin-top: 1.2rem; }}
.sl-hero .sl-art {{ flex: 0 0 min(300px, 34%); color: {t['texte']}; }}
.sl-hero .sl-art img {{ width: 100%; height: auto; display: block; }}
.pilule {{ display: inline-flex; align-items: center; gap: 6px; padding: 7px 15px; border-radius: 999px; font-weight: 700; font-size: .9rem; }}
.pilule.claire {{ background: {t['actif_bg']}; color: {t['actif_txt']}; }}
.pilule.sombre {{ background: rgba(128,128,128,.10); color: {t['texte']}; border: 1px solid {t['bord']}; font-weight: 600; }}

.kpi {{ background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 22px; padding: 20px 22px 18px; min-height: 150px; }}
.kpi .etiquette {{ color: {t['doux']}; font-size: .86rem; font-weight: 600; display: flex; align-items: center; gap: 8px; }}
.kpi .point {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; flex-shrink: 0; }}
.kpi .valeur {{ font-size: 2.05rem; font-weight: 800; color: {t['texte']}; margin: .45rem 0 .5rem; line-height: 1.05; letter-spacing: -0.02em; }}
.kpi .detail {{ color: {t['doux']}; font-size: .83rem; line-height: 1.45; }}
.badge {{ display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: .78rem; font-weight: 700; margin-right: 6px; }}
.badge.vert {{ background: rgba({t['menthe_rgb']},.14); color: {t['menthe']}; }}
.badge.lavande {{ background: rgba({t['accent_rgb']},.15); color: {t['accent']}; }}
.badge.jaune {{ background: rgba({t['jaune_rgb']},.14); color: {t['jaune']}; }}
.badge.gris {{ background: rgba(128,128,128,.12); color: {t['doux']}; }}

.encadre {{ border-radius: 18px; padding: 16px 20px; margin: .7rem 0 1.2rem; font-size: .97rem; line-height: 1.6; color: {t['texte']}; border: 1px solid {t['bord']}; }}
.encadre div {{ margin: 3px 0; }}
.encadre b {{ color: {t['fort']}; }}
.interp {{ background: linear-gradient(90deg, rgba({t['accent_rgb']},.10), rgba({t['accent_rgb']},.03)); border-left: 4px solid {t['accent']}; }}
.prof {{ background: linear-gradient(90deg, rgba({t['menthe_rgb']},.10), rgba({t['menthe_rgb']},.02)); border-left: 4px solid {t['menthe']}; }}
.attention {{ background: linear-gradient(90deg, rgba({t['jaune_rgb']},.12), rgba({t['jaune_rgb']},.02)); border-left: 4px solid {t['jaune']}; }}
.lecture {{ color: {t['doux']}; font-size: .9rem; margin: .1rem 0 1.2rem .4rem; line-height: 1.5; }}
.lecture b {{ color: {t['texte']}; }}

.carte {{ background: {t['carte']}; border: 1px solid {t['bord']}; border-radius: 22px; padding: 20px 22px; }}
.carte h4 {{ margin: 0 0 .2rem; padding: 0; font-size: 1.15rem; color: {t['texte']}; }}
.carte .sous {{ color: {t['doux']}; font-size: .85rem; margin-bottom: .9rem; }}
.priorite {{ display: flex; gap: 12px; align-items: center; padding: 12px 14px; border-radius: 16px; margin-bottom: 8px; background: {t['carte2']}; }}
.priorite .rang {{ font-weight: 800; font-size: .95rem; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center;
                   justify-content: center; color: #fff; flex-shrink: 0; }}
.priorite .texte {{ flex: 1; line-height: 1.35; font-size: .95rem; color: {t['texte']}; }}
.priorite .texte span {{ color: {t['doux']}; font-size: .82rem; }}
.priorite .montant {{ font-weight: 800; font-size: 1.1rem; white-space: nowrap; }}

.decision {{ border: 1px solid rgba({t['accent_rgb']},.35); border-radius: 24px; padding: 22px 26px; background: {t['carte']}; margin-top: 1rem; }}
.decision .titre {{ font-size: 1.2rem; font-weight: 800; color: {t['texte']}; }}
.decision ol {{ margin: .6rem 0 0 1.1rem; padding: 0; color: {t['texte']}; }}
.decision li {{ margin: .4rem 0; line-height: 1.55; }}
.decision li::marker {{ color: {t['accent']}; font-weight: 700; }}

.pied {{ color: {t['doux']}; font-size: .8rem; margin-top: 2.5rem; padding-top: 1rem; border-top: 1px solid {t['bord']}; }}

@media (max-width: 700px) {{
  .sl-hero {{ flex-direction: column-reverse; align-items: flex-start; padding: 22px 20px; border-radius: 22px; }}
  .sl-hero .sl-art {{ flex-basis: auto; width: 70%; align-self: center; }}
  .sl-hero h1 {{ font-size: 1.7rem; }}
  .kpi .valeur {{ font-size: 1.8rem; }}
}}
</style>
"""


def apply_style() -> None:
    st.html(_css(tokens()))


ACCENTS = {"lavande": "accent", "jaune": "jaune", "menthe": "menthe"}


def accent(name: str) -> str:
    t = tokens()
    return t[ACCENTS[name]] if name in ACCENTS else (t["doux"] if name == "gris" else name)


def hero(kicker: str, title: str, text: str, pills=(), art: str | None = None) -> None:
    """Page banner with gradient, pills and an inline SVG illustration."""
    from illustrations import ART

    svg = ""
    if art in ART:
        import base64

        raw = ART[art](colors()).replace("currentColor", tokens()["texte"])
        b64 = base64.b64encode(raw.encode()).decode()
        svg = f"<img src='data:image/svg+xml;base64,{b64}' alt='' width='300' height='188'>"
    row = "".join(f"<span class='pilule {cls}'>{label}</span>" for label, cls in pills)
    row = f"<div class='rangee'>{row}</div>" if row else ""
    st.html(f"<div class='sl-hero'><div class='sl-text'><div class='surtitre'>{kicker}</div><h1>{title}</h1>"
            f"<p>{text}</p>{row}</div><div class='sl-art'>{svg}</div></div>")


def header(title: str, subtitle: str, art: str | None = None, kicker: str = "Salary Lab") -> None:
    hero(kicker, title, _md_inline(subtitle), art=art)


def _md_inline(text: str) -> str:
    import html
    import re

    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", html.escape(text))


def kpi(label: str, value: str, detail: str = "", color: str = "lavande", badge: str | None = None,
        badge_cls: str = "vert") -> None:
    pastille = f"<span class='badge {badge_cls}'>{badge}</span>" if badge else ""
    st.html(f"<div class='kpi'><div class='etiquette'><span class='point' style='background:{accent(color)}'></span>"
            f"{label}</div><div class='valeur'>{value}</div><div class='detail'>{pastille}{detail}</div></div>")


def interpret(result: str, plain: str, scope: str, limit: str) -> None:
    rows = [("🎯 Résultat", result), ("💬 En clair", plain), ("👥 Population concernée", scope), ("⚠️ Limite", limit)]
    st.html("<div class='encadre interp'>" + "".join(f"<div><b>{t} :</b> {c}</div>" for t, c in rows) + "</div>")


def explain_box(title: str, text: str) -> None:
    st.html(f"<div class='encadre prof'>🎓 <b>{title}</b><br>{text}</div>")


def warn_box(title: str, text: str) -> None:
    st.html(f"<div class='encadre attention'>⚠️ <b>{title}</b><br>{text}</div>")


def reading(text: str) -> None:
    st.html(f"<div class='lecture'>👁️ <b>Lecture :</b> {text}</div>")


def ranking(title: str, subtitle: str, rows) -> None:
    """rows: iterable of (rank, name, detail, amount_text, color_name)."""
    items = "".join(
        f"<div class='priorite'><div class='rang' style='background:{accent(c)}'>{r}</div>"
        f"<div class='texte'><b>{n}</b><br><span>{d}</span></div>"
        f"<div class='montant' style='color:{accent(c)}'>{a}</div></div>" for r, n, d, a, c in rows)
    st.html(f"<div class='carte'><h4>{title}</h4><div class='sous'>{subtitle}</div>{items}</div>")


def decision(title: str, items) -> None:
    st.html(f"<div class='decision'><div class='titre'>{title}</div><ol>"
            + "".join(f"<li>{i}</li>" for i in items) + "</ol></div>")


def footer() -> None:
    st.html("<div class='pied'>Salary Lab · 250 000 profils (Kaggle, CC0) · statistiques SciPy/statsmodels, causalité "
            "DoWhy/EconML, modèles XGBoost/LightGBM/CatBoost, graphiques Plotly. Thème clair ou sombre : menu ⋮ → "
            "Settings → Theme.</div>")


def synthetic_notice() -> None:
    warn_box("Données probablement synthétiques",
             "Catégories parfaitement équilibrées, aucune valeur manquante, variables explicatives indépendantes. "
             "Les résultats décrivent le processus qui a généré ces données, pas le marché du travail réel.")


# ---------- loaders ----------

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    from salary.data import load_raw

    return load_raw()


def _require(path: Path, how: str) -> Path:
    if not path.exists():
        st.error(f"Artefact manquant : `{path.relative_to(cfg.ROOT)}`. Lancez `{how}`.")
        st.stop()
    return path


@st.cache_data(show_spinner=False)
def load_csv(group: str, name: str) -> pd.DataFrame:
    how = {"stats": "uv run python -m salary.stats", "causal": "uv run python -m salary.causal",
           "models": "uv run python -m salary.train"}[group]
    return pd.read_csv(_require(cfg.REPORTS_DIR / group / name, how))


@st.cache_data(show_spinner=False)
def load_parquet(group: str, name: str) -> pd.DataFrame:
    return pd.read_parquet(_require(cfg.REPORTS_DIR / group / name, "uv run python -m salary.train"))


@st.cache_data(show_spinner=False)
def load_json(path: str) -> dict:
    p = cfg.ROOT / path
    return json.loads(_require(p, "uv run python -m salary.train").read_text())


def metrics() -> dict:
    return load_json("models/metrics.json")


@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model(name: str):
    return joblib.load(_require(cfg.MODELS_DIR / f"{name}.joblib", "uv run python -m salary.train"))


def model_label(name: str) -> str:
    return DISPLAY_NAMES.get(name, name)


def feature_label(name: str) -> str:
    return LABELS.get(name, name)


def seq_scale() -> list:
    s = colors()["seq"]
    return [[i / (len(s) - 1), c] for i, c in enumerate(s)]


def sample(df: pd.DataFrame, n: int, seed: int = cfg.SEED) -> pd.DataFrame:
    return df.sample(min(n, len(df)), random_state=seed) if len(df) > n else df


__all__ = ["np", "pd", "go", "st"]
