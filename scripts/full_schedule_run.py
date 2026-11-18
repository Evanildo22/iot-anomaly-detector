#!/usr/bin/env python3
"""
Full-schedule training run on N-BaIoT with the report's hyperparameters.

Trains the leak-free pipeline (split first, fit on training rows only),
evaluates the held-out test and real-time splits, and writes results.json,
ROC and confusion-matrix plots, checkpoints and TensorBoard logs to --out.
The git commit and working-tree state are recorded for provenance.

Example
-------
python scripts/full_schedule_run.py \
    --data_folder /path/to/N-BaIoT/Dataset \
    --summary_csv /path/to/N-BaIoT/Dataset/data_summary.csv \
    --out runs/full_schedule
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time

import matplotlib

matplotlib.use("Agg")
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))

from iot_anomaly.data.preprocessor import (DataPreprocessor, engineer_nbaiot_features,
                                           load_nbaiot_dataset)
from iot_anomaly.evaluation.evaluator import (compute_metrics, plot_confusion_matrix,
                                              plot_roc_curve)
from iot_anomaly.training.trainer import Trainer, TrainingConfig

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
log = logging.getLogger("full_schedule")


def git_provenance() -> dict:
    def run(*cmd):
        return subprocess.run(["git", "-C", REPO, *cmd], capture_output=True,
                              text=True).stdout.strip()
    status = run("status", "--porcelain")
    return {
        "commit": run("rev-parse", "HEAD"),
        "worktree_dirty": bool(status),
        "dirty_paths": [line[3:] for line in status.splitlines()],
    }


def to_jsonable(metrics: dict) -> dict:
    return {k: (int(v) if isinstance(v, (int, np.integer)) else float(v))
            for k, v in metrics.items()}


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--data_folder", required=True)
    p.add_argument("--summary_csv", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--finetune_epochs", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    os.makedirs(args.out, exist_ok=True)
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
    provenance = git_provenance()
    log.info("Commit %s (worktree dirty: %s)", provenance["commit"], provenance["worktree_dirty"])

    t0 = time.time()
    combined = load_nbaiot_dataset(args.data_folder, args.summary_csv)
    labels = combined["label"].reset_index(drop=True)
    features = engineer_nbaiot_features(combined.drop(columns=["label"])).reset_index(drop=True)
    del combined
    log.info("Loaded %d rows, class counts %s", len(labels), np.bincount(labels).tolist())

    cfg = TrainingConfig(pretrain_epochs=args.pretrain_epochs,
                         finetune_epochs=args.finetune_epochs, batch_size=256,
                         random_state=args.seed,
                         save_dir=os.path.join(args.out, "checkpoints"),
                         log_dir=os.path.join(args.out, "tensorboard"))
    prep = DataPreprocessor(n_top_features=15, pca_variance=0.95, resample=True,
                            random_state=args.seed)
    trainer = Trainer(cfg)
    trainer.fit_from_features(features, labels, prep, pretrain=True)
    t_train = time.time() - t0

    results = trainer.evaluate()
    proba = trainer.classifier.predict(trainer._X_test, verbose=0).ravel()
    pred = (proba >= trainer.threshold).astype(int)
    plot_roc_curve(trainer._y_test, proba, save_path=os.path.join(args.out, "roc_curve.png"))
    plot_confusion_matrix(trainer._y_test, pred,
                          save_path=os.path.join(args.out, "confusion_matrix.png"))
    rt_proba = trainer.classifier.predict(trainer._X_realtime, verbose=0).ravel()
    rt_metrics = compute_metrics(trainer._y_realtime,
                                 (rt_proba >= trainer.threshold).astype(int), rt_proba)

    summary = {
        "dataset": "N-BaIoT",
        "summary_csv": os.path.basename(args.summary_csv),
        "provenance": provenance,
        "rows": int(len(labels)),
        "config": {k: v for k, v in vars(cfg).items() if k not in ("save_dir", "log_dir")},
        "preprocessing": {"n_top_features": 15, "pca_variance": 0.95, "resample": True},
        "split_sizes": {"train_after_resampling": int(len(trainer._y_train)),
                        "val": int(len(trainer._y_val)), "test": int(len(trainer._y_test)),
                        "realtime": int(len(trainer._y_realtime))},
        "test_class_counts": np.bincount(trainer._y_test.astype(int)).tolist(),
        "pca_components": int(trainer._X_train.shape[1]),
        "threshold": float(trainer.threshold),
        "test_metrics": to_jsonable(results),
        "realtime_metrics": to_jsonable(rt_metrics),
        "wall_clock_hours": round((time.time() - t0) / 3600, 2),
        "training_hours": round(t_train / 3600, 2),
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    log.info("DONE — results written to %s", os.path.join(args.out, "results.json"))


if __name__ == "__main__":
    main()
