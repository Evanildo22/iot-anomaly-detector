"""Preprocessing, the canonical split, and the split-first leakage guarantee (BUG-25/26)."""

import numpy as np
import pandas as pd
import pytest

from iot_anomaly.data.preprocessor import (
    DataPreprocessor,
    engineer_nbaiot_features,
    load_nbaiot_dataset,
    select_top_features,
)
from iot_anomaly.training.trainer import Trainer, TrainingConfig, split_indices

from conftest import make_tabular


# ── Feature selection and pipeline ────────────────────────────────────────────

def test_select_top_features_drops_correlated_columns(tabular):
    X, y = tabular
    top = select_top_features(X, pd.Series(y), n_top=5)
    assert len(top) == 5
    assert "f05" not in top  # near-duplicate of f00
    assert set(top) <= set(X.columns)


def test_fit_transform_returns_3d_balanced_training_data(tabular):
    X, y = tabular
    prep = DataPreprocessor(n_top_features=8)
    X3, y_res = prep.fit_transform(X, pd.Series(y))
    assert X3.ndim == 3 and X3.shape[2] == 1
    assert X3.shape[1] == prep.n_components_
    counts = np.bincount(y_res)
    assert counts[0] == counts[1]  # SMOTE + under-sampling balance the classes


def test_fit_transform_without_resampling_keeps_rows(tabular):
    X, y = tabular
    X3, y_out = DataPreprocessor(n_top_features=8, resample=False).fit_transform(X, pd.Series(y))
    assert len(X3) == len(X)
    np.testing.assert_array_equal(y_out, y)


def test_transform_never_resamples(tabular):
    X, y = tabular
    prep = DataPreprocessor(n_top_features=8)
    prep.fit_transform(X.iloc[:600], pd.Series(y[:600]))
    out = prep.transform(X.iloc[600:])
    assert out.shape == (200, prep.n_components_, 1)


def test_transform_before_fit_raises(tabular):
    prep = DataPreprocessor()
    with pytest.raises(RuntimeError):
        prep.transform(tabular[0])
    with pytest.raises(RuntimeError):
        _ = prep.n_components_


def test_engineer_nbaiot_features_adds_aggregates():
    df = pd.DataFrame({
        "MI_dir_L5_mean": [1.0], "MI_dir_L3_mean": [2.0], "MI_dir_L1_mean": [3.0],
        "MI_dir_L5_variance": [4.0], "MI_dir_L3_variance": [5.0], "MI_dir_L1_variance": [6.0],
        "MI_dir_L5_weight": [7.0], "MI_dir_L3_weight": [8.0], "MI_dir_L1_weight": [9.0],
    })
    out = engineer_nbaiot_features(df)
    assert out.loc[0, "mean_of_means"] == 2.0
    assert out.loc[0, "mean_of_variances"] == 5.0
    assert out.loc[0, "mean_of_weights"] == 8.0
    assert "mean_of_means" not in df  # input not mutated


def test_load_nbaiot_dataset_labels_from_filename(tmp_path):
    for name, n in [("1.benign.csv", 3), ("1.mirai.ack.csv", 2)]:
        pd.DataFrame({"a": range(n), "b": range(n)}).to_csv(tmp_path / name, index=False)
    summary = tmp_path / "summary.csv"
    pd.DataFrame({"File Name": ["1.benign.csv", "1.mirai.ack.csv", "missing.csv"]}).to_csv(
        summary, index=False)
    df = load_nbaiot_dataset(str(tmp_path), str(summary))
    assert len(df) == 5
    assert df["label"].tolist() == [0, 0, 0, 1, 1]


def test_load_nbaiot_dataset_with_no_files_raises(tmp_path):
    summary = tmp_path / "summary.csv"
    pd.DataFrame({"File Name": ["missing.csv"]}).to_csv(summary, index=False)
    with pytest.raises(FileNotFoundError):
        load_nbaiot_dataset(str(tmp_path), str(summary))


# ── Canonical split ───────────────────────────────────────────────────────────

def test_split_indices_are_disjoint_complete_and_stratified():
    y = np.array([0] * 300 + [1] * 1700)
    idx = split_indices(y, TrainingConfig())
    all_idx = np.concatenate(list(idx.values()))
    assert len(all_idx) == len(y) == len(np.unique(all_idx))
    sizes = {k: len(v) / len(y) for k, v in idx.items()}
    assert sizes["train"] == pytest.approx(0.65, abs=0.01)
    assert sizes["val"] == pytest.approx(0.15, abs=0.01)
    assert sizes["test"] == pytest.approx(0.15, abs=0.01)
    assert sizes["realtime"] == pytest.approx(0.05, abs=0.01)
    for part in idx.values():
        assert y[part].mean() == pytest.approx(0.85, abs=0.02)


def test_split_indices_are_deterministic():
    y = np.array([0, 1] * 200)
    a, b = split_indices(y, TrainingConfig()), split_indices(y, TrainingConfig())
    for k in a:
        np.testing.assert_array_equal(a[k], b[k])


def test_split_indices_rejects_single_class():
    with pytest.raises(ValueError, match="single class"):
        split_indices(np.ones(100), TrainingConfig())


# ── Leakage guarantee ─────────────────────────────────────────────────────────

class SpyPreprocessor(DataPreprocessor):
    """Records which raw rows each method sees."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.fit_rows = None

    def fit_transform(self, features, labels, resample=None):
        self.fit_rows = set(features.index)
        return super().fit_transform(features, labels, resample)


@pytest.fixture
def captured_split(tmp_path, monkeypatch):
    """Run fit_from_features up to model training and capture what it passes on."""
    X, y = make_tabular(n=1000)
    X.index = pd.RangeIndex(1000) + 10_000  # non-default index to catch iloc/loc mix-ups
    captured = {}

    def fake_fit_presplit(self, *arrays, X_pretrain=None, pretrain=True):
        names = ["X_train", "y_train", "X_val", "y_val", "X_test", "y_test",
                 "X_realtime", "y_realtime"]
        captured.update(zip(names, arrays))
        captured["X_pretrain"] = X_pretrain
        return self

    monkeypatch.setattr(Trainer, "fit_presplit", fake_fit_presplit)
    cfg = TrainingConfig(save_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"))
    prep = SpyPreprocessor(n_top_features=8)
    Trainer(cfg).fit_from_features(X, y, prep)
    idx = split_indices(y, cfg)
    return X, y, idx, prep, captured


def test_preprocessor_is_fitted_on_training_rows_only(captured_split):
    X, _, idx, prep, _ = captured_split
    train_rows = set(X.index[idx["train"]])
    assert prep.fit_rows == train_rows
    for part in ("val", "test", "realtime"):
        assert prep.fit_rows.isdisjoint(X.index[idx[part]])


def test_evaluation_splits_are_real_rows_with_natural_balance(captured_split):
    _, y, idx, prep, cap = captured_split
    for part in ("val", "test", "realtime"):
        # Same row count as the raw split: no synthetic SMOTE rows were added.
        assert len(cap[f"X_{part}"]) == len(idx[part])
        np.testing.assert_array_equal(cap[f"y_{part}"], y[idx[part]])
    # Only the training split is resampled.
    assert len(cap["X_train"]) != len(idx["train"])
    assert cap["X_test"].shape[1:] == (prep.n_components_, 1)


def test_pretraining_uses_real_benign_training_rows_only(captured_split):
    _, y, idx, _, cap = captured_split
    n_benign_train = int((y[idx["train"]] == 0).sum())
    assert len(cap["X_pretrain"]) == n_benign_train
