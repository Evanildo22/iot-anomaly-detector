"""
Training orchestrator — two-phase training:

  Phase 1 (optional): Unsupervised autoencoder pretraining on reconstruction
                      of benign training traffic only.
  Phase 2:            Supervised classifier fine-tuning with attention head.

Fixed bugs vs original:
  BUG-15  All splits performed once from a canonical hold-out; test set is
          never touched during training.
  BUG-16  Resampled arrays are passed to the model, not the original imbalanced data.
  BUG-19  Split proportions documented and verified to match stated intent.
  BUG-20  find_optimal_threshold aligns array lengths before indexing.
  BUG-21  One canonical save path; ModelCheckpoint and manual save agree.
  BUG-22  Removed triple-quoted dead-code block.
  BUG-25  Raw rows are split *before* any fitting: feature selection,
          imputation, scaling, PCA and SMOTE are fitted on the training
          split only (fit_from_features). Validation, test and real-time
          sets are real rows with their natural class balance.
  BUG-26  The autoencoder is pretrained on real benign training rows only,
          never on attack traffic or SMOTE-synthesised rows.
  ARCH-03 All hyperparameters come from a config dict, not magic numbers.
  ARCH-04 Global seeds set for numpy, python, and tensorflow.
"""

from __future__ import annotations

import datetime
import logging
import os
import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

from ..models.autoencoder import build_autoencoder
from ..models.classifier import build_classifier

if TYPE_CHECKING:
    from ..data.preprocessor import DataPreprocessor

logger = logging.getLogger(__name__)


# ── Configuration ──────────────────────────────────────────────────────────────

@dataclass
class TrainingConfig:
    # Autoencoder architecture
    filters: int = 112
    kernel_size: int = 6
    dropout_rate: float = 0.43
    # Classifier head
    dense_units: int = 43
    l1_reg: float = 2.9e-6
    l2_reg: float = 6.25e-4
    # Optimisation
    learning_rate: float = 3.01e-4
    # Training schedule
    pretrain_epochs: int = 30
    finetune_epochs: int = 50
    batch_size: int = 256
    early_stopping_patience: int = 10
    # Data split (FIX BUG-19: proportions verified to match stated intent)
    # Train: 65%, Validation: 15%, Test: 15%, Real-time eval: 5%
    val_size: float = 0.15
    test_size: float = 0.15
    realtime_size: float = 0.05
    # Label value of benign traffic (used to select autoencoder pretraining rows)
    benign_label: int = 0
    # Misc
    random_state: int = 42
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


# ── Canonical split ────────────────────────────────────────────────────────────

def split_indices(labels: np.ndarray, cfg: TrainingConfig) -> dict[str, np.ndarray]:
    """
    Stratified train / val / test / realtime split of row *indices*.

    Splitting indices (not transformed arrays) lets callers split raw data
    before anything is fitted on it (FIX BUG-25).
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    if len(classes) < 2:
        raise ValueError(
            f"Training data contains a single class ({classes.tolist()}); a "
            "binary detector needs both benign and malicious rows. For CTU "
            "captures, use a PCAP that includes benign background traffic."
        )
    idx = np.arange(len(labels))

    # Step 1: carve out realtime set (5%)
    idx_main, idx_rt = train_test_split(
        idx, test_size=cfg.realtime_size, stratify=labels,
        random_state=cfg.random_state,
    )
    # Step 2: from remaining (95%), carve out test set (≈15.8% of 95% ≈ 15% total)
    test_fraction_of_main = cfg.test_size / (1 - cfg.realtime_size)
    idx_main2, idx_test = train_test_split(
        idx_main, test_size=test_fraction_of_main, stratify=labels[idx_main],
        random_state=cfg.random_state,
    )
    # Step 3: from remaining, carve out validation set (≈18.75% of 80% ≈ 15% total)
    val_fraction_of_main2 = cfg.val_size / (1 - cfg.realtime_size - cfg.test_size)
    idx_train, idx_val = train_test_split(
        idx_main2, test_size=val_fraction_of_main2, stratify=labels[idx_main2],
        random_state=cfg.random_state,
    )
    return {"train": idx_train, "val": idx_val, "test": idx_test, "realtime": idx_rt}


# ── Optimal threshold finder ───────────────────────────────────────────────────

def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
) -> tuple[float, float]:
    """
    Find the decision threshold that maximises F1.

    FIX BUG-20: aligns precision, recall, thresholds arrays before indexing.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    # precision_recall_curve returns len(n+1) for precision/recall, len(n) for thresholds
    precision = precision[:-1]
    recall    = recall[:-1]
    denom     = precision + recall
    f1_scores = np.where(denom > 0, 2 * precision * recall / denom, 0.0)
    best_idx  = int(np.argmax(f1_scores))
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


# ── Trainer ────────────────────────────────────────────────────────────────────

class Trainer:
    """
    Two-phase trainer for the CAE + Attention anomaly classifier.

    Usage (recommended — leak-free end to end)
    -----
    >>> trainer = Trainer(config)
    >>> trainer.fit_from_features(features_df, labels, DataPreprocessor())
    >>> results = trainer.evaluate()
    """

    def __init__(self, config: Optional[TrainingConfig] = None):
        self.cfg = config or TrainingConfig()
        self._set_seeds(self.cfg.random_state)
        os.makedirs(self.cfg.save_dir, exist_ok=True)
        os.makedirs(self.cfg.log_dir,  exist_ok=True)

        self.autoencoder: Optional[tf.keras.Model] = None
        self.encoder:     Optional[tf.keras.Model] = None
        self.classifier:  Optional[tf.keras.Model] = None
        self.threshold:   float = 0.5

        # Split arrays populated by fit_*()
        self._X_train  = self._X_val = self._X_test = self._X_realtime = None
        self._y_train  = self._y_val = self._y_test = self._y_realtime = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit_from_features(
        self,
        features: pd.DataFrame,
        labels: pd.Series | np.ndarray,
        preprocessor: "DataPreprocessor",
        pretrain: bool = True,
    ) -> "Trainer":
        """
        Leak-free pipeline from raw feature rows (FIX BUG-25 / BUG-26).

        1. Split raw rows into train / val / test / realtime (stratified).
        2. Fit feature selection, imputer, scaler, PCA and SMOTE on the
           training rows only.
        3. Transform val / test / realtime with the fitted pipeline
           (no resampling — they keep their natural class balance).
        4. Pretrain the autoencoder on real benign training rows only.
        """
        y_all = np.asarray(labels)
        idx = split_indices(y_all, self.cfg)
        logger.info(
            "Raw split sizes — train: %d | val: %d | test: %d | realtime: %d",
            *(len(idx[k]) for k in ("train", "val", "test", "realtime")),
        )

        f_train = features.iloc[idx["train"]]
        y_train_raw = y_all[idx["train"]]

        # Fitted on training rows only; resampling (if enabled) applies here only.
        X_train, y_train = preprocessor.fit_transform(f_train, pd.Series(y_train_raw))

        X_val = preprocessor.transform(features.iloc[idx["val"]])
        X_test = preprocessor.transform(features.iloc[idx["test"]])
        X_rt = preprocessor.transform(features.iloc[idx["realtime"]])

        # Real (non-synthetic) benign training rows for autoencoder pretraining
        X_train_real = preprocessor.transform(f_train)
        X_pretrain = X_train_real[y_train_raw == self.cfg.benign_label]

        return self.fit_presplit(
            X_train, y_train,
            X_val, y_all[idx["val"]],
            X_test, y_all[idx["test"]],
            X_rt, y_all[idx["realtime"]],
            X_pretrain=X_pretrain,
            pretrain=pretrain,
        )

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pretrain: bool = True,
    ) -> "Trainer":
        """
        Split already-transformed arrays, then train.

        Only leak-free if *X* was NOT produced by a preprocessor fitted on the
        same rows and was NOT resampled — otherwise use fit_from_features().
        Kept for arrays that need no fitted preprocessing.
        """
        y = np.asarray(y)
        idx = split_indices(y, self.cfg)
        X_train, y_train = X[idx["train"]], y[idx["train"]]
        return self.fit_presplit(
            X_train, y_train,
            X[idx["val"]], y[idx["val"]],
            X[idx["test"]], y[idx["test"]],
            X[idx["realtime"]], y[idx["realtime"]],
            X_pretrain=X_train[y_train == self.cfg.benign_label],
            pretrain=pretrain,
        )

    def fit_presplit(
        self,
        X_train: np.ndarray, y_train: np.ndarray,
        X_val: np.ndarray, y_val: np.ndarray,
        X_test: np.ndarray, y_test: np.ndarray,
        X_realtime: np.ndarray, y_realtime: np.ndarray,
        X_pretrain: Optional[np.ndarray] = None,
        pretrain: bool = True,
    ) -> "Trainer":
        """Train on explicit, already-separated splits."""
        cfg = self.cfg
        input_shape = X_train.shape[1:]

        self._X_train, self._y_train = X_train, y_train
        self._X_val,   self._y_val   = X_val,   y_val
        self._X_test,  self._y_test  = X_test,  y_test
        self._X_realtime, self._y_realtime = X_realtime, y_realtime

        logger.info(
            "Model split sizes — train: %d | val: %d | test: %d | realtime: %d",
            len(y_train), len(y_val), len(y_test), len(y_realtime),
        )

        # ── Build models ───────────────────────────────────────────────────
        self.autoencoder, self.encoder = build_autoencoder(
            input_shape,
            filters=cfg.filters,
            kernel_size=cfg.kernel_size,
            dropout_rate=cfg.dropout_rate,
            learning_rate=cfg.learning_rate,
        )
        self.classifier = build_classifier(
            self.encoder,
            input_shape,
            dense_units=cfg.dense_units,
            dropout_rate=cfg.dropout_rate,
            l1_reg=cfg.l1_reg,
            l2_reg=cfg.l2_reg,
            learning_rate=cfg.learning_rate,
        )

        # ── Phase 1: autoencoder pretraining on benign traffic only ───────
        if pretrain:
            if X_pretrain is None or len(X_pretrain) == 0:
                raise ValueError("No benign training rows available for pretraining.")
            X_val_benign = X_val[y_val == cfg.benign_label]
            logger.info(
                "Phase 1: Autoencoder pretraining on %d benign rows …", len(X_pretrain)
            )
            ae_ckpt_path = os.path.join(cfg.save_dir, "autoencoder_best.weights.h5")
            self.autoencoder.fit(
                X_pretrain, X_pretrain,      # reconstruction target = input
                validation_data=(X_val_benign, X_val_benign),
                epochs=cfg.pretrain_epochs,
                batch_size=cfg.batch_size,
                callbacks=self._callbacks("ae", ae_ckpt_path),
                verbose=1,
            )

        # ── Phase 2: supervised classifier fine-tuning ────────────────────
        logger.info("Phase 2: Classifier fine-tuning …")
        clf_ckpt_path = os.path.join(cfg.save_dir, "classifier_best.weights.h5")
        self.classifier.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=cfg.finetune_epochs,
            batch_size=cfg.batch_size,
            callbacks=self._callbacks("clf", clf_ckpt_path),
            verbose=1,
        )

        # FIX BUG-21: load the best checkpoint (not the last epoch).
        # Weights-only: TF 2.13's ModelCheckpoint cannot write the native
        # .keras format, and load_weights also sidesteps custom-layer
        # deserialization for the attention head.
        self.classifier.load_weights(clf_ckpt_path)
        logger.info("Loaded best checkpoint: %s", clf_ckpt_path)

        # Full-model export for use in other processes. Load it with
        # tf.keras.models.load_model(path, compile=False): on Apple Silicon,
        # TF 2.13 substitutes the legacy Adam optimizer, whose saved state
        # cannot be restored. Inference does not need the optimizer.
        model_path = os.path.join(cfg.save_dir, "classifier_best.keras")
        self.classifier.save(model_path)
        logger.info("Exported best classifier: %s (load with compile=False)", model_path)

        # ── Threshold fine-tuning on validation set ────────────────────────
        y_val_proba = self.classifier.predict(X_val).ravel()
        self.threshold, best_f1 = find_optimal_threshold(y_val, y_val_proba)
        logger.info("Optimal threshold: %.4f  (val F1=%.4f)", self.threshold, best_f1)

        return self

    def evaluate(self) -> dict:
        """Evaluate on the held-out test set."""
        from ..evaluation.evaluator import compute_metrics
        if self.classifier is None:
            raise RuntimeError("Call fit_from_features() or fit() before evaluate().")
        y_proba = self.classifier.predict(self._X_test).ravel()
        y_pred  = (y_proba > self.threshold).astype(int)
        return compute_metrics(self._y_test, y_pred, y_proba)

    def predict_realtime(self, X: np.ndarray) -> np.ndarray:
        """Binary predictions for real-time inference."""
        proba = self.classifier.predict(X).ravel()
        return (proba > self.threshold).astype(int)

    # ── Internal helpers ───────────────────────────────────────────────────

    def _callbacks(self, prefix: str, ckpt_path: str) -> list:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return [
            EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.early_stopping_patience,
                restore_best_weights=True,
            ),
            # FIX BUG-21: single canonical checkpoint path per phase
            ModelCheckpoint(
                ckpt_path,
                save_best_only=True,
                save_weights_only=True,
                monitor="val_loss",
            ),
            TensorBoard(
                log_dir=os.path.join(self.cfg.log_dir, f"{prefix}_{ts}"),
                histogram_freq=1,
            ),
        ]

    @staticmethod
    def _set_seeds(seed: int) -> None:
        """FIX ARCH-04: set all global random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        logger.info("Random seed set to %d", seed)
