"""
Sklearn preprocessing pipeline for both CTU-IoT (PCAP-derived) and
N-BaIoT (CSV-based) feature sets.

Fixed bugs vs original:
  BUG-17  sklearn.Pipeline and imblearn.Pipeline imported with aliases;
          no name collision.
  ARCH-01 Unified Dataset abstraction with shared preprocessing contract.
  ARCH-03 Hyperparameters / column lists passed as arguments, not hard-coded.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline as SklearnPipeline
from sklearn.preprocessing import StandardScaler

# FIX BUG-17: explicit alias so imblearn.Pipeline does not shadow sklearn.Pipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

logger = logging.getLogger(__name__)


# ── N-BaIoT dataset loader ─────────────────────────────────────────────────────

def load_nbaiot_dataset(dataset_folder: str, summary_csv: str) -> pd.DataFrame:
    """
    Load and combine all N-BaIoT CSV files listed in *summary_csv*.

    Binary label: 0 = benign (filename contains 'benign'), 1 = attack.
    """
    summary = pd.read_csv(summary_csv)
    # Simple binary label based on filename
    summary["label"] = summary["File Name"].apply(
        lambda name: 0 if "benign" in str(name).lower() else 1
    )

    frames = []
    for _, row in summary.iterrows():
        path = os.path.join(dataset_folder, row["File Name"])
        if not os.path.exists(path):
            logger.warning("File not found, skipping: %s", path)
            continue
        df = pd.read_csv(path)
        df["label"] = row["label"]
        frames.append(df)

    if not frames:
        raise FileNotFoundError("No N-BaIoT CSV files could be loaded from %s" % dataset_folder)

    combined = pd.concat(frames, ignore_index=True)
    logger.info(
        "N-BaIoT: loaded %d rows, label distribution:\n%s",
        len(combined), combined["label"].value_counts().to_string(),
    )
    return combined


# ── Feature engineering helpers ────────────────────────────────────────────────

def engineer_nbaiot_features(features: pd.DataFrame) -> pd.DataFrame:
    """Add cross-window aggregate features for N-BaIoT data."""
    mean_cols     = ["MI_dir_L5_mean",     "MI_dir_L3_mean",     "MI_dir_L1_mean"]
    var_cols      = ["MI_dir_L5_variance", "MI_dir_L3_variance", "MI_dir_L1_variance"]
    weight_cols   = ["MI_dir_L5_weight",   "MI_dir_L3_weight",   "MI_dir_L1_weight"]

    present = lambda cols: [c for c in cols if c in features.columns]
    if present(mean_cols):
        features = features.copy()
        features["mean_of_means"]     = features[present(mean_cols)].mean(axis=1)
        features["mean_of_variances"] = features[present(var_cols)].mean(axis=1)
        features["mean_of_weights"]   = features[present(weight_cols)].mean(axis=1)
    return features


def select_top_features(
    features: pd.DataFrame,
    labels: pd.Series,
    n_top: int = 15,
    correlation_threshold: float = 0.80,
) -> list[str]:
    """
    Remove highly-correlated features, then rank by Random Forest importance.
    Returns a list of *n_top* feature column names.
    """
    # Remove highly correlated columns
    corr = features.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > correlation_threshold)]
    reduced = features.drop(columns=to_drop, errors="ignore")
    logger.info("Removed %d highly-correlated features", len(to_drop))

    # Random Forest importance ranking
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(reduced, labels)
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_cols = reduced.columns[indices[:n_top]].tolist()
    logger.info("Top %d features: %s", n_top, top_cols)
    return top_cols


# ── Preprocessing pipeline ─────────────────────────────────────────────────────

def build_preprocessing_pipeline(
    selected_features: list[str],
    pca_variance: float = 0.95,
) -> ColumnTransformer:
    """
    Build a sklearn ColumnTransformer that imputes, scales, and applies PCA
    to *selected_features*.

    FIX BUG-17: uses SklearnPipeline alias — no conflict with imblearn.Pipeline.

    Parameters
    ----------
    selected_features : list of str
        Column names to include in the numeric transformer.
    pca_variance : float
        Fraction of variance to retain in PCA (default 0.95).
    """
    numeric_pipeline = SklearnPipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("pca",     PCA(n_components=pca_variance, random_state=42)),
    ])
    return ColumnTransformer([
        ("num", numeric_pipeline, selected_features)
    ])


def build_resampling_pipeline(
    sampling_strategy_over: str | float = "minority",
    sampling_strategy_under: str | float = "majority",
    random_state: int = 42,
) -> ImbPipeline:
    """
    Build an imblearn pipeline that over-samples the minority class with SMOTE
    then under-samples the majority class.

    FIX BUG-16: this pipeline's output is explicitly returned and *must* be used
    by the trainer — the original code computed it but then trained on the raw data.
    """
    return ImbPipeline([
        ("smote", SMOTE(
            sampling_strategy=sampling_strategy_over,
            random_state=random_state,
        )),
        ("under", RandomUnderSampler(
            sampling_strategy=sampling_strategy_under,
            random_state=random_state,
        )),
    ])


# ── Convenience end-to-end preprocessor ───────────────────────────────────────

class DataPreprocessor:
    """
    Stateful preprocessor that wraps feature selection, preprocessing pipeline,
    and resampling in a single object.

    Usage
    -----
    >>> prep = DataPreprocessor()
    >>> X_train_3d, y_train = prep.fit_transform(features_df, labels, resample=True)
    >>> X_test_3d  = prep.transform(test_features_df)
    """

    def __init__(
        self,
        n_top_features: int = 15,
        pca_variance: float = 0.95,
        resample: bool = True,
        random_state: int = 42,
    ):
        self.n_top_features = n_top_features
        self.pca_variance = pca_variance
        self.resample = resample
        self.random_state = random_state

        self._selected_features: Optional[list[str]] = None
        self._preprocessing_pipeline: Optional[ColumnTransformer] = None

    @property
    def n_components_(self) -> int:
        """Number of PCA components after fitting."""
        if self._preprocessing_pipeline is None:
            raise RuntimeError("Call fit_transform first.")
        pca = self._preprocessing_pipeline.named_transformers_["num"].named_steps["pca"]
        return pca.n_components_

    def fit_transform(
        self,
        features: pd.DataFrame,
        labels: pd.Series,
        resample: Optional[bool] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Fit all transformers on *features* / *labels* and return (X_3d, y).

        The returned X_3d is shaped (n_samples, n_pca_components, 1) to match
        the Conv1D input convention.
        """
        if resample is None:
            resample = self.resample

        # Feature selection
        self._selected_features = select_top_features(
            features, labels, n_top=self.n_top_features
        )

        # Build and fit preprocessing pipeline
        self._preprocessing_pipeline = build_preprocessing_pipeline(
            self._selected_features, pca_variance=self.pca_variance
        )
        X = self._preprocessing_pipeline.fit_transform(features[self._selected_features])
        y = labels.values

        # FIX BUG-16: actually use the resampled arrays
        if resample:
            resampler = build_resampling_pipeline(random_state=self.random_state)
            X, y = resampler.fit_resample(X, y)
            logger.info("After resampling: %d samples", len(y))

        return X.reshape(X.shape[0], X.shape[1], 1), y

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted pipeline (no resampling)."""
        if self._preprocessing_pipeline is None or self._selected_features is None:
            raise RuntimeError("Call fit_transform before transform.")
        X = self._preprocessing_pipeline.transform(features[self._selected_features])
        return X.reshape(X.shape[0], X.shape[1], 1)
