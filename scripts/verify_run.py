#!/usr/bin/env python3
"""
Verify a completed full-schedule run from a fresh process.

Rebuilds the run's deterministic split and fitted preprocessing, loads the
*exported* model (classifier_best.keras) rather than the training process's
in-memory model, and checks that:

  1. the split sizes, class counts and PCA width match results.json;
  2. the loaded model keeps the hidden ReLU (MaxReLU layer, BUG-29) and its
     predict() output matches a NumPy forward pass with the ReLU;
  3. no layer call has training=True stored in the graph (BUG-28), inference
     is deterministic, and eager and graph execution agree;
  4. on a GPU, graph-mode output matches the CPU;
  5. re-tuning the threshold on the validation split reproduces the run's;
  6. test and real-time metrics and exact confusion counts reproduce
     results.json.

Metrics, ROC points and plots are regenerated from the freshly loaded model
and written, with verification.json, to --out.

Example
-------
python scripts/verify_run.py --run_dir runs/full_schedule \
    --data_folder /path/to/N-BaIoT/Dataset \
    --summary_csv /path/to/N-BaIoT/Dataset/data_summary.csv
"""

import argparse
import json
import os
import re
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

import tensorflow as tf
from sklearn.metrics import roc_curve

import iot_anomaly.models  # noqa: F401  (registers MaxReLU for deserialisation)
from iot_anomaly.data.preprocessor import (DataPreprocessor, engineer_nbaiot_features,
                                           load_nbaiot_dataset)
from iot_anomaly.evaluation.evaluator import (compute_metrics, plot_confusion_matrix,
                                              plot_roc_curve)
from iot_anomaly.training.trainer import TrainingConfig, find_optimal_threshold, split_indices

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLE = 20_000  # rows used for the forward-pass and device checks


def git(*cmd) -> str:
    return subprocess.run(["git", "-C", REPO, *cmd], capture_output=True, text=True).stdout.strip()


def training_commit(run_dir: str, results: dict) -> str | None:
    if "provenance" in results:
        return results["provenance"]["commit"]
    launcher = os.path.join(run_dir, "launcher.log")
    if os.path.exists(launcher):
        m = re.search(r"at commit ([0-9a-f]{7,40})", open(launcher).read())
        if m:
            return git("rev-parse", m.group(1))
    return None


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Verify a full-schedule run from a fresh process")
    p.add_argument("--run_dir", required=True)
    p.add_argument("--data_folder", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out", help="default: <run_dir>/verification")
    args = p.parse_args(argv)
    out = args.out or os.path.join(args.run_dir, "verification")
    os.makedirs(out, exist_ok=True)

    results = json.load(open(os.path.join(args.run_dir, "results.json")))
    checks: dict[str, bool] = {}
    details: dict = {}

    # ── Provenance ────────────────────────────────────────────────────────
    commit = training_commit(args.run_dir, results)
    src_changes = git("diff", "--stat", commit, "HEAD", "--", "src/") if commit else "unknown"
    details["provenance"] = {
        "training_commit": commit,
        "training_commit_source": "results.json" if "provenance" in results else "launcher.log",
        "verification_commit": git("rev-parse", "HEAD"),
        "verification_worktree_dirty_paths": [l[3:] for l in git("status", "--porcelain").splitlines()],
        "src_changes_since_training": src_changes or "none",
    }

    # ── Rebuild the split and the fitted preprocessing ───────────────────
    combined = load_nbaiot_dataset(args.data_folder, args.summary_csv)
    labels = combined["label"].reset_index(drop=True)
    features = engineer_nbaiot_features(combined.drop(columns=["label"])).reset_index(drop=True)
    del combined
    rc = results["config"]
    cfg = TrainingConfig(val_size=rc["val_size"], test_size=rc["test_size"],
                         realtime_size=rc["realtime_size"], random_state=rc["random_state"])
    y = labels.values
    idx = split_indices(y, cfg)
    pp = results.get("preprocessing", {"n_top_features": 15, "pca_variance": 0.95})
    # Resampling runs after fitting and never touches evaluation rows, so the
    # fitted transform is identical without it.
    prep = DataPreprocessor(n_top_features=pp["n_top_features"], pca_variance=pp["pca_variance"],
                            resample=False, random_state=rc["random_state"])
    prep.fit_transform(features.iloc[idx["train"]], labels.iloc[idx["train"]])
    X = {k: prep.transform(features.iloc[idx[k]]) for k in ("val", "test", "realtime")}
    Y = {k: y[idx[k]] for k in X}
    del features

    checks["split_sizes_match"] = all(
        len(Y[k]) == results["split_sizes"][k] for k in ("val", "test", "realtime"))
    checks["test_class_counts_match"] = np.bincount(Y["test"]).tolist() == results["test_class_counts"]
    checks["pca_components_match"] = X["test"].shape[1] == results["pca_components"]

    # ── Load the exported model in this fresh process ────────────────────
    model_path = os.path.join(args.run_dir, "checkpoints", "classifier_best.keras")
    model = tf.keras.models.load_model(model_path, compile=False)
    layer_types = {l.name: type(l).__name__ for l in model.layers}
    checks["hidden_relu_layer_present"] = layer_types.get("clf_relu") == "MaxReLU"
    checks["hidden_dense_is_linear"] = (
        model.get_layer("clf_dense").activation.__name__ == "linear")
    config_text = json.dumps(model.get_config())
    checks["no_training_true_in_graph"] = '"training": true' not in config_text

    rng = np.random.default_rng(0)
    sample = X["test"][rng.choice(len(Y["test"]), size=min(SAMPLE, len(Y["test"])), replace=False)]

    p1 = model.predict(sample, verbose=0)
    p2 = model.predict(sample, verbose=0)
    checks["predict_deterministic"] = bool(np.array_equal(p1, p2))
    e1 = model(sample, training=False).numpy()
    e2 = model(sample, training=False).numpy()
    checks["eager_deterministic"] = bool(np.array_equal(e1, e2))
    details["max_abs_diff_eager_vs_predict"] = float(np.abs(e1 - p1).max())
    checks["eager_matches_predict"] = details["max_abs_diff_eager_vs_predict"] < 1e-5

    flat = tf.keras.Model(model.input, model.get_layer("clf_flatten").output)
    h = flat(sample, training=False).numpy()
    w1, b1 = model.get_layer("clf_dense").get_weights()
    w2, b2 = model.get_layer("clf_output").get_weights()
    with_relu = 1 / (1 + np.exp(-(np.maximum(h @ w1 + b1, 0) @ w2 + b2)))
    without_relu = 1 / (1 + np.exp(-((h @ w1 + b1) @ w2 + b2)))
    details["max_abs_diff_predict_vs_numpy_relu"] = float(np.abs(p1 - with_relu).max())
    details["max_abs_diff_predict_vs_numpy_no_relu"] = float(np.abs(p1 - without_relu).max())
    checks["predict_matches_numpy_forward_pass_with_relu"] = (
        details["max_abs_diff_predict_vs_numpy_relu"] < 1e-4)

    if tf.config.list_physical_devices("GPU"):
        with tf.device("/CPU:0"):
            cpu = model(sample, training=False).numpy()
        details["max_abs_diff_gpu_predict_vs_cpu"] = float(np.abs(p1 - cpu).max())
        checks["gpu_graph_matches_cpu"] = details["max_abs_diff_gpu_predict_vs_cpu"] < 1e-4
    else:
        details["gpu_graph_matches_cpu"] = "skipped: no GPU"

    # ── Threshold and metrics from the freshly loaded model ──────────────
    proba = {k: model.predict(X[k], verbose=0).ravel() for k in X}
    threshold, _ = find_optimal_threshold(Y["val"], proba["val"])
    details["threshold_run"] = results["threshold"]
    details["threshold_fresh"] = threshold
    checks["threshold_reproduced"] = abs(threshold - results["threshold"]) < 1e-7

    thr = results["threshold"]
    fresh = {}
    for split, key in (("test", "test_metrics"), ("realtime", "realtime_metrics")):
        m = compute_metrics(Y[split], (proba[split] >= thr).astype(int), proba[split])
        fresh[split] = {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
                        for k, v in m.items()}
        ref = results[key]
        checks[f"{split}_confusion_counts_exact"] = all(
            fresh[split][c] == ref[c] for c in ("tn", "fp", "fn", "tp"))
        checks[f"{split}_metrics_match"] = all(
            abs(fresh[split][k] - ref[k]) < 1e-9
            for k in ("accuracy", "precision", "recall", "f1", "fpr", "auc_roc"))

    fpr, tpr, roc_thr = roc_curve(Y["test"], proba["test"])
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thr}).to_csv(
        os.path.join(out, "roc_points_test.csv"), index=False)
    plot_roc_curve(Y["test"], proba["test"], save_path=os.path.join(out, "roc_curve.png"))
    plot_confusion_matrix(Y["test"], (proba["test"] >= thr).astype(int),
                          save_path=os.path.join(out, "confusion_matrix.png"))

    report = {
        "passed": all(checks.values()),
        "checks": checks,
        "details": details,
        "fresh_process_metrics": fresh,
    }
    with open(os.path.join(out, "verification.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("VERIFICATION", "PASSED" if report["passed"] else "FAILED")
    for name, ok in checks.items():
        print(f"  {'ok  ' if ok else 'FAIL'} {name}")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
