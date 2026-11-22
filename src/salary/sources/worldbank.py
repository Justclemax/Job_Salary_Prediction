"""World Bank API (https://api.worldbank.org/v2, free, no key): purchasing power.

A salary in US dollars at market exchange rates overstates the gap between rich
and poor countries, because prices differ. To compare purchasing power we convert
to international dollars (PPP):

    salary_ppp = salary_usd * FX / PPP

where FX = PA.NUS.FCRF (local currency per USD) and PPP = PA.NUS.PPP (local
currency per international dollar). FX / PPP is the inverse of the price level.
"""

import json
import urllib.request

import pandas as pd

from salary import config as cfg

API = "https://api.worldbank.org/v2/country/{countries}/indicator/{indicator}"
INDICATORS = {"ppp": "PA.NUS.PPP", "fx": "PA.NUS.FCRF"}
CACHE = cfg.ROOT / "data" / "external" / "worldbank"


def fetch(indicator: str, iso3: list[str], start: int = 2018, end: int = 2026,
          force: bool = False) -> pd.DataFrame:
    """Return (iso3, year, value) rows for one indicator, cached as JSON."""
    path = CACHE / f"{indicator}_{start}_{end}.json"
    if force or not path.exists():
        url = (API.format(countries=";".join(sorted(iso3)), indicator=indicator)
               + f"?format=json&date={start}:{end}&per_page=2000")
        with urllib.request.urlopen(url, timeout=60) as r:  # noqa: S310 (fixed https URL)
            payload = json.load(r)
        if len(payload) < 2 or payload[1] is None:
            raise RuntimeError(f"Banque mondiale : réponse vide pour {indicator} ({payload[0]})")
        CACHE.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload[1]))
    rows = json.loads(path.read_text())
    df = pd.DataFrame([{"iso3": r["countryiso3code"], "year": int(r["date"]), "value": r["value"]}
                       for r in rows])
    return df.dropna()


def price_level(iso3: list[str], year: int, force: bool = False) -> pd.DataFrame:
    """Price level (PPP / FX, USA = 1) per country, using the latest year <= `year`."""
    out = None
    for name, code in INDICATORS.items():
        d = fetch(code, iso3, force=force)
        d = d[d["year"] <= year].sort_values("year").groupby("iso3").tail(1)
        d = d.rename(columns={"value": name, "year": f"{name}_year"})
        out = d if out is None else out.merge(d, on="iso3")
    out["price_level"] = out["ppp"] / out["fx"]
    missing = set(iso3) - set(out["iso3"])
    if missing:
        raise RuntimeError(f"Banque mondiale : pas de PPA pour {sorted(missing)}")
    return out.reset_index(drop=True)


def to_ppp(df: pd.DataFrame, iso3_col: str = "iso3", year: int = 2025) -> pd.Series:
    """International-dollar salary for each row of `df` (needs `salary_usd` and `iso3_col`)."""
    pl = price_level(df[iso3_col].unique().tolist(), year).set_index("iso3")["price_level"]
    return df["salary_usd"] / df[iso3_col].map(pl)
