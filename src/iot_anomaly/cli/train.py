#!/usr/bin/env python3
"""
CLI: Train the CAE + Attention anomaly classifier.

Supports both N-BaIoT (pre-extracted CSV) and CTU-IoT (PCAP-derived CSV)
datasets, or any pre-processed CSV with a 'label' column.

Usage — N-BaIoT dataset:
    iot-train \
        --dataset nbaiot \
        --data_folder data/raw/N-BaIoT/Dataset \
        --summary_csv data/raw/N-BaIoT/Dataset/data_summary.csv \
        --save_dir checkpoints/nbaiot

Usage — CTU PCAP-derived CSV (output of iot-process-pcap):
    iot-train \
        --dataset ctu \
        --features_csv data/processed/ctu_features.csv \
        --save_dir checkpoints/ctu

Options:
    --no_pretrain   Skip autoencoder pretraining (Phase 1)
    --filters N     Conv1D filters (default 112)
    --kernel N      Kernel size (default 6)
    --dense N       Classifier dense units (default 43)
    --dropout F     Dropout rate (default 0.43)
    --lr F          Learning rate (default 3.01e-4)
    --epochs N      Fine-tuning epochs (default 50)
    --batch N       Batch size (default 256)
    --seed N        Random seed (default 42)
"""

import argparse
import logging
import os

import numpy as np
import pandas as pd

from iot_anomaly.data.preprocessor import (
    DataPreprocessor,
    engineer_nbaiot_features,
    load_nbaiot_dataset,
)
from iot_anomaly.training.trainer import Trainer, TrainingConfig
from iot_anomaly.evaluation.evaluator import (
    compute_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("train")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train IoT anomaly detector")
    p.add_argument("--dataset", choices=["nbaiot", "ctu"], required=True)

    # N-BaIoT paths
    p.add_argument("--data_folder",  help="N-BaIoT root dataset folder")
    p.add_argument("--summary_csv",  help="N-BaIoT data_summary.csv")

    # CTU / generic CSV path
    p.add_argument("--features_csv", help="Pre-processed features CSV with 'label' column")

    p.add_argument("--save_dir",   default="checkpoints")
    p.add_argument("--no_pretrain", action="store_true")

    # Hyperparameters
    p.add_argument("--filters", type=int,   default=112)
    p.add_argument("--kernel",  type=int,   default=6)
    p.add_argument("--dense",   type=int,   default=43)
    p.add_argument("--dropout", type=float, default=0.43)
    p.add_argument("--lr",      type=float, default=3.01e-4)
    p.add_argument("--epochs",  type=int,   default=50)
    p.add_argument("--pretrain_epochs", type=int, default=30)
    p.add_argument("--batch",   type=int,   default=256)
    p.add_argument("--seed",    type=int,   default=42)
    p.add_argument("--n_top_features", type=int, default=15)
    return p.parse_args(argv)


def load_features_labels(args: argparse.Namespace) -> tuple[pd.DataFrame, pd.Series]:
    if args.dataset == "nbaiot":
        if not args.data_folder or not args.summary_csv:
            raise ValueError("--data_folder and --summary_csv are required for nbaiot")
        combined = load_nbaiot_dataset(args.data_folder, args.summary_csv)
        labels   = combined["label"]
        features = combined.drop(columns=["label"])
        features = engineer_nbaiot_features(features)

    else:  # ctu or generic CSV
        if not args.features_csv:
            raise ValueError("--features_csv is required for ctu dataset")
        df       = pd.read_csv(args.features_csv)
        labels   = df["label"]
        # Drop non-numeric / identifier columns. 'protocol' is the raw string
        # ("TCP"/"UDP"/"ICMP"); the numeric TCP/UDP/ICMP fraction columns
        # already encode it, and a string column crashes the preprocessor.
        drop_cols = ["session_id", "src_ip", "dst_ip", "timestamp_session",
                     "protocol", "label", "label_raw"]
        features = df.drop(columns=[c for c in drop_cols if c in df.columns])

    return features, labels


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # ── Load data ─────────────────────────────────────────────────────────────
    features, labels = load_features_labels(args)
    logger.info("Loaded dataset: %d rows × %d features", *features.shape)

    # ── Preprocess (feature selection + PCA + SMOTE) ──────────────────────────
    preprocessor = DataPreprocessor(
        n_top_features=args.n_top_features,
        pca_variance=0.95,
        resample=True,
        random_state=args.seed,
    )
    X_3d, y = preprocessor.fit_transform(features, labels)
    logger.info("Preprocessed shape: %s  |  label distribution: %s",
                X_3d.shape, np.bincount(y.astype(int)).tolist())

    # ── Build training config ─────────────────────────────────────────────────
    cfg = TrainingConfig(
        filters=args.filters,
        kernel_size=args.kernel,
        dense_units=args.dense,
        dropout_rate=args.dropout,
        learning_rate=args.lr,
        pretrain_epochs=args.pretrain_epochs,
        finetune_epochs=args.epochs,
        batch_size=args.batch,
        random_state=args.seed,
        save_dir=args.save_dir,
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(cfg)
    trainer.fit(X_3d, y, pretrain=not args.no_pretrain)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    results = trainer.evaluate()
    print("\n── Test-set results ──────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<12}  {v:.4f}")

    # ── Plots ─────────────────────────────────────────────────────────────────
    os.makedirs(args.save_dir, exist_ok=True)
    y_proba = trainer.classifier.predict(trainer._X_test).ravel()
    y_pred  = (y_proba > trainer.threshold).astype(int)
    plot_roc_curve(trainer._y_test, y_proba,
                   save_path=os.path.join(args.save_dir, "roc_curve.png"))
    plot_confusion_matrix(trainer._y_test, y_pred,
                          save_path=os.path.join(args.save_dir, "confusion_matrix.png"))

    logger.info("Training complete. Checkpoints and plots saved to: %s", args.save_dir)


if __name__ == "__main__":
    main()
