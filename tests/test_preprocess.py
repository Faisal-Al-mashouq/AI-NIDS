import numpy as np
import pandas as pd

from src.config import settings
from src.data.clean import clean
from src.features.preprocess import make_binary_label
from src.features.selection import select_features


def test_clean_removes_duplicates_and_invalid_rows():
    df = pd.DataFrame({"a": [1.0, 1.0, np.inf, 3.0], "b": [1.0, 1.0, 2.0, 4.0]})
    out = clean(df)
    # second [1, 1] is a duplicate; [inf, 2] -> NaN and is dropped
    assert len(out) == 2


def test_make_binary_label_maps_benign_to_zero():
    df = pd.DataFrame({settings.raw_label_col: ["BENIGN", "DDoS", "PortScan"]})
    out = make_binary_label(df)
    assert out[settings.label_col].tolist() == [0, 1, 1]


def test_select_features_keeps_numeric_and_label():
    df = pd.DataFrame(
        {"num": [1, 2, 3], "text": ["x", "y", "z"], settings.label_col: [0, 1, 0]}
    )
    out = select_features(df)
    assert "text" not in out.columns
    assert "num" in out.columns
    assert settings.label_col in out.columns
