"""
Model evaluation utilities.

Fixed bugs vs original:
  BUG-20  find_optimal_threshold moved to trainer.py with proper array alignment.
          This module focuses on metrics, plots, and real-time anomaly reporting.
"""

from __future__ import annotations

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    classification_report,
)

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None,
) -> dict[str, float]:
    """
    Return a comprehensive metrics dictionary.

    Confusion counts come directly from the predictions, and the
    false-positive rate (share of benign rows flagged as malicious) is
    reported because accuracy is weak on a 92%-malicious dataset.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    metrics = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
        "fpr":       fp / (fp + tn) if (fp + tn) else 0.0,
    }
    if y_pred_proba is not None:
        metrics["auc_roc"] = roc_auc_score(y_true, y_pred_proba)
    metrics.update({"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)})

    logger.info("Evaluation results:")
    for k, v in metrics.items():
        logger.info("  %-12s %s", k, v if isinstance(v, int) else f"{v:.4f}")
    logger.info("\n%s", classification_report(y_true, y_pred,
                target_names=["Benign", "Malicious"]))
    return metrics


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
    plt.plot([0, 1], [0, 1], "k--", lw=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")
    plt.legend(loc="lower right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("ROC curve saved to %s", save_path)
        plt.close()  # plt.show() would block a terminal run on a GUI backend
    else:
        plt.show()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Benign", "Malicious"],
        yticklabels=["Benign", "Malicious"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        logger.info("Confusion matrix saved to %s", save_path)
        plt.close()  # plt.show() would block a terminal run on a GUI backend
    else:
        plt.show()


def realtime_inference(
    model,
    X_stream: np.ndarray,
    threshold: float = 0.5,
    batch_size: int = 64,
) -> None:
    """
    Iterate over a data stream in batches and print anomaly alerts.

    Parameters
    ----------
    model : Keras model with predict() method.
    X_stream : array of shape (n, time_steps, 1).
    threshold : float — decision boundary.
    batch_size : int — inference batch size (use >1 for throughput).
    """
    total = len(X_stream)
    alerts = 0
    for start in range(0, total, batch_size):
        batch = X_stream[start : start + batch_size]
        proba = model.predict(batch, verbose=0).ravel()
        for i, p in enumerate(proba):
            if p >= threshold:
                alerts += 1
                logger.warning(
                    "ANOMALY DETECTED — sample %d | p=%.4f | threshold=%.4f",
                    start + i, p, threshold,
                )
    logger.info(
        "Real-time scan complete: %d/%d samples flagged as anomalous (%.2f%%)",
        alerts, total, 100 * alerts / max(total, 1),
    )
