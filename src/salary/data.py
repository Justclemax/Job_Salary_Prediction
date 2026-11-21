"""Loading, validation and the canonical train/test split."""

import hashlib

import pandas as pd
from sklearn.model_selection import train_test_split

from salary import config as C


class DataValidationError(ValueError):
    pass


def validate(df: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Check columns, nulls, category levels and numeric ranges. Returns df unchanged."""
    expected = C.FEATURES + ([C.TARGET] if require_target else [])
    missing = set(expected) - set(df.columns)
    if missing:
        raise DataValidationError(f"Colonnes manquantes : {sorted(missing)}")
    nulls = df[expected].isna().sum()
    if nulls.any():
        raise DataValidationError(f"Valeurs manquantes : {nulls[nulls > 0].to_dict()}")
    for col, levels in C.CATEGORY_LEVELS.items():
        bad = set(df[col].unique()) - set(levels)
        if bad:
            raise DataValidationError(f"Modalités inconnues pour {col} : {sorted(bad)}")
    for col, (lo, hi) in C.NUMERIC_RANGES.items():
        if not df[col].between(lo, hi).all():
            raise DataValidationError(f"{col} hors de l'intervalle [{lo}, {hi}]")
    if require_target and (df[C.TARGET] <= 0).any():
        raise DataValidationError("Le salaire doit être strictement positif")
    return df


def load_raw(path=C.DATA_RAW) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in C.CATEGORICAL:
        df[col] = df[col].astype(str)
    return validate(df)


def file_md5(path=C.DATA_RAW) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fixed, stratified 80/20 split. The test set is only for final reporting."""
    train, test = train_test_split(
        df, test_size=C.TEST_SIZE, random_state=C.SEED, stratify=df["job_title"]
    )
    return train, test


def load_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the persisted split (created by `salary.train`), or create it."""
    train_p = C.DATA_PROCESSED / "train.parquet"
    test_p = C.DATA_PROCESSED / "test.parquet"
    if train_p.exists() and test_p.exists():
        return pd.read_parquet(train_p), pd.read_parquet(test_p)
    train, test = split(load_raw())
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    train.to_parquet(train_p)
    test.to_parquet(test_p)
    return train, test
