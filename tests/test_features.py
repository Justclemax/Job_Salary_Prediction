import numpy as np
import pytest

from salary import config as C
from salary.features import make_preprocessor


@pytest.mark.parametrize("kind", ["linear", "tree", "native"])
def test_preprocessor_outputs(small, kind):
    out = make_preprocessor(kind).fit_transform(small[C.FEATURES])
    assert len(out) == len(small)
    if kind != "native":
        assert np.isfinite(out.to_numpy(dtype=float)).all()


def test_tree_encoding_respects_ordinal_order(small):
    out = make_preprocessor("tree").fit(small[C.FEATURES])
    enc = out.transform(small[C.FEATURES])
    phd = enc.loc[small.education_level == "PhD", "education_level"].iloc[0]
    hs = enc.loc[small.education_level == "High School", "education_level"].iloc[0]
    assert phd > hs


def test_unknown_category_does_not_crash(small, profile):
    pre = make_preprocessor("linear").fit(small[C.FEATURES])
    p = profile.copy()
    p["industry"] = "Space"
    assert pre.transform(p).shape[0] == 1
