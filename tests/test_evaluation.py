"""Metrics, threshold tuning and plotting."""

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from iot_anomaly.evaluation.evaluator import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from iot_anomaly.training.trainer import find_optimal_threshold


def test_compute_metrics_exact_counts():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1, 1, 1, 1, 0])
    m = compute_metrics(y_true, y_pred)
    assert (m["tn"], m["fp"], m["fn"], m["tp"]) == (3, 1, 1, 5)
    assert all(isinstance(m[k], int) for k in ("tn", "fp", "fn", "tp"))
    assert m["fpr"] == pytest.approx(0.25)
    assert m["accuracy"] == pytest.approx(0.8)
    assert m["precision"] == pytest.approx(5 / 6)
    assert m["recall"] == pytest.approx(5 / 6)
    assert "auc_roc" not in m


def test_compute_metrics_with_probabilities():
    y_true = np.array([0, 0, 1, 1])
    proba = np.array([0.1, 0.4, 0.35, 0.8])
    m = compute_metrics(y_true, (proba > 0.5).astype(int), proba)
    assert m["auc_roc"] == pytest.approx(0.75)


def test_find_optimal_threshold_separable():
    y = np.array([0, 0, 0, 1, 1, 1])
    proba = np.array([0.05, 0.1, 0.2, 0.3, 0.6, 0.9])
    threshold, f1 = find_optimal_threshold(y, proba)
    assert f1 == pytest.approx(1.0)
    # The returned threshold is itself a score: the positive row sitting on it
    # is only classified correctly with '>=' (BUG-27).
    assert threshold == 0.3
    assert ((proba >= threshold).astype(int) == y).all()
    assert not ((proba > threshold).astype(int) == y).all()


def test_plots_are_saved_and_closed(tmp_path):
    import matplotlib.pyplot as plt

    y = np.array([0, 1, 0, 1])
    proba = np.array([0.2, 0.7, 0.4, 0.9])
    roc, cm = tmp_path / "roc.png", tmp_path / "cm.png"
    plot_roc_curve(y, proba, save_path=str(roc))
    plot_confusion_matrix(y, (proba > 0.5).astype(int), save_path=str(cm))
    assert roc.stat().st_size > 0 and cm.stat().st_size > 0
    assert plt.get_fignums() == []  # figures closed; plt.show() would block a CLI run
