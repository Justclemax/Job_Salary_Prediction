"""Preprocessing shared by training, evaluation and the app."""

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from salary import config as C


def make_preprocessor(kind: str) -> ColumnTransformer:
    """
    kind="linear": scaled numerics + one-hot of every categorical (no ordinality assumed).
    kind="tree":   raw numerics + ordinal codes (true order for ordinal columns).
    kind="native": columns passed through unchanged (CatBoost handles categoricals itself).
    """
    if kind == "linear":
        ct = ColumnTransformer(
            [
                ("num", StandardScaler(), C.NUMERIC),
                (
                    "cat",
                    OneHotEncoder(
                        categories=[C.CATEGORY_LEVELS[c] for c in C.CATEGORICAL],
                        drop="first",
                        sparse_output=False,
                        handle_unknown="ignore",
                    ),
                    C.CATEGORICAL,
                ),
            ],
            verbose_feature_names_out=False,
        )
    elif kind == "tree":
        ct = ColumnTransformer(
            [
                ("num", "passthrough", C.NUMERIC),
                (
                    "cat",
                    OrdinalEncoder(
                        categories=[C.CATEGORY_LEVELS[c] for c in C.CATEGORICAL],
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    C.CATEGORICAL,
                ),
            ],
            verbose_feature_names_out=False,
        )
    elif kind == "native":
        ct = ColumnTransformer(
            [("all", "passthrough", C.FEATURES)], verbose_feature_names_out=False
        )
    else:
        raise ValueError(f"Unknown preprocessor kind: {kind}")
    return ct.set_output(transform="pandas")
