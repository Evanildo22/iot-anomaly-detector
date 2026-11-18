"""End-to-end smoke tests: raw rows → split → preprocess → train → evaluate, and the CLI."""

import os

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from iot_anomaly.cli import train as train_cli
from iot_anomaly.data.preprocessor import DataPreprocessor
from iot_anomaly.training.trainer import Trainer, TrainingConfig

from conftest import make_tabular

TINY = dict(filters=8, kernel_size=3, dense_units=4, dropout_rate=0.1,
            pretrain_epochs=2, finetune_epochs=3, batch_size=64,
            early_stopping_patience=2, learning_rate=3e-3)


def tiny_config(tmp_path, **kw):
    return TrainingConfig(save_dir=str(tmp_path / "ckpt"), log_dir=str(tmp_path / "logs"),
                          **{**TINY, **kw})


def test_fit_from_features_evaluate_and_predict(tmp_path):
    X, y = make_tabular(n=1200)
    trainer = Trainer(tiny_config(tmp_path))
    trainer.fit_from_features(X, y, DataPreprocessor(n_top_features=8))

    results = trainer.evaluate()
    assert set(results) >= {"accuracy", "precision", "recall", "f1", "fpr",
                            "auc_roc", "tn", "fp", "fn", "tp"}
    assert results["tn"] + results["fp"] + results["fn"] + results["tp"] == len(trainer._y_test)
    # The synthetic classes are well separated, so even a tiny model learns them.
    assert results["auc_roc"] > 0.9
    assert 0.0 <= trainer.threshold <= 1.0

    preds = trainer.predict_realtime(trainer._X_realtime)
    assert preds.shape == (len(trainer._y_realtime),)
    assert set(np.unique(preds)) <= {0, 1}

    for name in ("autoencoder_best.weights.h5", "classifier_best.weights.h5",
                 "classifier_best.keras"):
        assert (tmp_path / "ckpt" / name).exists()


def test_exported_model_loads_with_compile_false(tmp_path):
    import tensorflow as tf

    X, y = make_tabular(n=800)
    trainer = Trainer(tiny_config(tmp_path, pretrain_epochs=1, finetune_epochs=1))
    trainer.fit_from_features(X, y, DataPreprocessor(n_top_features=8))
    model = tf.keras.models.load_model(tmp_path / "ckpt" / "classifier_best.keras",
                                       compile=False)
    np.testing.assert_allclose(
        model.predict(trainer._X_test, verbose=0),
        trainer.classifier.predict(trainer._X_test, verbose=0),
        rtol=1e-5, atol=1e-6,
    )


def test_pretraining_requires_benign_rows(tmp_path):
    X = np.random.default_rng(0).normal(size=(200, 8, 1)).astype("float32")
    y = np.array([0, 1] * 100)
    trainer = Trainer(tiny_config(tmp_path))
    with pytest.raises(ValueError, match="benign"):
        trainer.fit_presplit(X, y, X, y, X, y, X, y, X_pretrain=X[:0])


def test_evaluate_before_fit_raises(tmp_path):
    with pytest.raises(RuntimeError):
        Trainer(tiny_config(tmp_path)).evaluate()


# ── CLI ───────────────────────────────────────────────────────────────────────

def test_cli_parse_args_defaults_match_report():
    args = train_cli.parse_args(["--dataset", "nbaiot", "--data_folder", "d",
                                 "--summary_csv", "s.csv"])
    assert (args.filters, args.kernel, args.dense) == (112, 6, 43)
    assert args.epochs == 50 and args.pretrain_epochs == 30 and args.seed == 42


def test_cli_trains_on_nbaiot_layout(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # TensorBoard logs go to ./logs
    X, y = make_tabular(n=900, n_features=12)
    data = tmp_path / "data"
    data.mkdir()
    X[y == 0].to_csv(data / "1.benign.csv", index=False)
    X[y == 1].to_csv(data / "1.mirai.ack.csv", index=False)
    pd.DataFrame({"File Name": ["1.benign.csv", "1.mirai.ack.csv"]}).to_csv(
        data / "summary.csv", index=False)
    out = tmp_path / "out"

    train_cli.main([
        "--dataset", "nbaiot", "--data_folder", str(data),
        "--summary_csv", str(data / "summary.csv"), "--save_dir", str(out),
        "--filters", "8", "--kernel", "3", "--dense", "4", "--dropout", "0.1",
        "--epochs", "1", "--pretrain_epochs", "1", "--batch", "64",
        "--n_top_features", "6",
    ])
    assert (out / "classifier_best.weights.h5").exists()
    pngs = [f for f in os.listdir(out) if f.endswith(".png")]
    assert len(pngs) == 2  # ROC curve and confusion matrix
