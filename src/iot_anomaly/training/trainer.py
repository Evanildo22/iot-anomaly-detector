"""
Training orchestrator — two-phase training:

  Phase 1 (optional): Unsupervised autoencoder pretraining on reconstruction.
  Phase 2:            Supervised classifier fine-tuning with attention head.

Fixed bugs vs original:
  BUG-15  All splits performed once from a canonical hold-out; test set is
          never touched during training.
  BUG-16  Resampled arrays are passed to the model, not the original imbalanced data.
  BUG-19  Split proportions documented and verified to match stated intent.
  BUG-20  find_optimal_threshold aligns array lengths before indexing.
  BUG-21  One canonical save path; ModelCheckpoint and manual save agree.
  BUG-22  Removed triple-quoted dead-code block.
  ARCH-03 All hyperparameters come from a config dict, not magic numbers.
  ARCH-04 Global seeds set for numpy, python, and tensorflow.
"""

from __future__ import annotations

import datetime
import logging
import os
import random
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
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
    # Misc
    random_state: int = 42
    save_dir: str = "checkpoints"
    log_dir: str = "logs"


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

    Usage
    -----
    >>> trainer = Trainer(config)
    >>> trainer.fit(X_3d, y)
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

        # Split arrays populated by fit()
        self._X_train  = self._X_val = self._X_test = self._X_realtime = None
        self._y_train  = self._y_val = self._y_test = self._y_realtime = None

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        pretrain: bool = True,
    ) -> "Trainer":
        """
        Perform canonical train/val/test/realtime split, then train models.

        FIX BUG-15: one canonical split; test set never touches training.
        FIX BUG-16: training uses the (already resampled) X, y passed in.
        """
        cfg = self.cfg
        input_shape = X.shape[1:]

        # ── Canonical split ────────────────────────────────────────────────
        # Step 1: carve out realtime set (5%)
        X_main, X_rt, y_main, y_rt = train_test_split(
            X, y,
            test_size=cfg.realtime_size,
            stratify=y,
            random_state=cfg.random_state,
        )
        # Step 2: from remaining (95%), carve out test set (≈15.8% of 95% ≈ 15% total)
        test_fraction_of_main = cfg.test_size / (1 - cfg.realtime_size)
        X_main2, X_test, y_main2, y_test = train_test_split(
            X_main, y_main,
            test_size=test_fraction_of_main,
            stratify=y_main,
            random_state=cfg.random_state,
        )
        # Step 3: from remaining, carve out validation set (≈17.6% of 80% ≈ 15% total)
        val_fraction_of_main2 = cfg.val_size / (1 - cfg.realtime_size - cfg.test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_main2, y_main2,
            test_size=val_fraction_of_main2,
            stratify=y_main2,
            random_state=cfg.random_state,
        )

        self._X_train, self._y_train  = X_train, y_train
        self._X_val,   self._y_val    = X_val,   y_val
        self._X_test,  self._y_test   = X_test,  y_test
        self._X_realtime, self._y_realtime = X_rt, y_rt

        logger.info(
            "Split sizes — train: %d | val: %d | test: %d | realtime: %d",
            len(y_train), len(y_val), len(y_test), len(y_rt),
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

        # ── Phase 1: autoencoder pretraining ──────────────────────────────
        if pretrain:
            logger.info("Phase 1: Autoencoder pretraining …")
            ae_ckpt_path = os.path.join(cfg.save_dir, "autoencoder_best.weights.h5")
            self.autoencoder.fit(
                X_train, X_train,          # reconstruction target = input
                validation_data=(X_val, X_val),
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

        # ── Threshold fine-tuning on validation set ────────────────────────
        y_val_proba = self.classifier.predict(X_val).ravel()
        self.threshold, best_f1 = find_optimal_threshold(y_val, y_val_proba)
        logger.info("Optimal threshold: %.4f  (val F1=%.4f)", self.threshold, best_f1)

        return self

    def evaluate(self) -> dict:
        """Evaluate on the held-out test set."""
        from ..evaluation.evaluator import compute_metrics
        if self.classifier is None:
            raise RuntimeError("Call fit() before evaluate().")
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
