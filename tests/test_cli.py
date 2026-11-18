"""Command-line entry points: argument validation, the CTU branch and iot-process-pcap."""

import matplotlib

matplotlib.use("Agg")

import pandas as pd
import pytest

from iot_anomaly.cli import process_pcap as pcap_cli
from iot_anomaly.cli import train as train_cli

from conftest import make_tabular

TINY_ARGS = ["--filters", "8", "--kernel", "3", "--dense", "4", "--dropout", "0.1",
             "--epochs", "1", "--pretrain_epochs", "1", "--batch", "64",
             "--n_top_features", "6"]


# ── iot-train argument validation ─────────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    [],                                         # --dataset is required
    ["--dataset", "unsw"],                      # unknown dataset
    ["--dataset", "nbaiot", "--epochs", "x"],   # non-integer epochs
])
def test_train_rejects_invalid_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        train_cli.parse_args(argv)
    assert exc.value.code == 2


@pytest.mark.parametrize("argv, message", [
    (["--dataset", "nbaiot"], "--data_folder and --summary_csv"),
    (["--dataset", "nbaiot", "--data_folder", "d"], "--data_folder and --summary_csv"),
    (["--dataset", "ctu"], "--features_csv"),
])
def test_train_requires_dataset_paths(argv, message):
    with pytest.raises(ValueError, match=message):
        train_cli.load_features_labels(train_cli.parse_args(argv))


# ── CTU branch ────────────────────────────────────────────────────────────────

def _ctu_features_csv(path, n=900):
    """A features CSV shaped like iot-process-pcap output, string columns included."""
    X, y = make_tabular(n=n, n_features=12)
    df = X.copy()
    df.insert(0, "session_id", [f"('10.0.0.{i % 250}', 'x')" for i in range(n)])
    df["src_ip"] = "192.168.1.10"
    df["dst_ip"] = "203.0.113.5"
    df["timestamp_session"] = 1000.0 + pd.RangeIndex(n)
    df["protocol"] = ["TCP" if i % 2 else "UDP" for i in range(n)]
    df["label"] = y
    df.to_csv(path, index=False)
    return df


def test_ctu_loader_drops_identifier_and_string_columns(tmp_path):
    csv = tmp_path / "ctu.csv"
    _ctu_features_csv(csv)
    args = train_cli.parse_args(["--dataset", "ctu", "--features_csv", str(csv)])
    features, labels = train_cli.load_features_labels(args)
    for col in ("session_id", "src_ip", "dst_ip", "timestamp_session", "protocol", "label"):
        assert col not in features
    # A string column left in would crash feature selection ('could not convert … TCP').
    assert all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes)
    assert len(labels) == len(features) == 900


def test_ctu_training_end_to_end(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # TensorBoard logs go to ./logs
    csv = tmp_path / "ctu.csv"
    _ctu_features_csv(csv)
    out = tmp_path / "out"
    train_cli.main(["--dataset", "ctu", "--features_csv", str(csv),
                    "--save_dir", str(out), *TINY_ARGS])
    assert (out / "classifier_best.weights.h5").exists()
    assert (out / "roc_curve.png").exists() and (out / "confusion_matrix.png").exists()


def test_train_no_pretrain_skips_autoencoder(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    csv = tmp_path / "ctu.csv"
    _ctu_features_csv(csv)
    out = tmp_path / "out"
    train_cli.main(["--dataset", "ctu", "--features_csv", str(csv),
                    "--save_dir", str(out), "--no_pretrain", *TINY_ARGS])
    assert not (out / "autoencoder_best.weights.h5").exists()
    assert (out / "classifier_best.weights.h5").exists()


# ── iot-process-pcap ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("argv", [
    ["--labels", "l", "--output", "o"],                                  # no input
    ["--pcap", "a", "--pcap_dir", "b", "--labels", "l", "--output", "o"],  # both inputs
    ["--pcap", "a", "--output", "o"],                                    # no labels
    ["--pcap", "a", "--labels", "l"],                                    # no output
])
def test_process_pcap_rejects_invalid_arguments(argv):
    with pytest.raises(SystemExit) as exc:
        pcap_cli.parse_args(argv)
    assert exc.value.code == 2


def test_process_pcap_single_file(tmp_path, pcap_file, conn_log_file):
    out = tmp_path / "nested" / "features.csv"
    pcap_cli.main(["--pcap", pcap_file, "--labels", conn_log_file, "--output", str(out)])
    df = pd.read_csv(out)
    assert sorted(df["label"]) == [0, 1, 1]
    assert {"session_id", "protocol", "inter_arrival_mean", "TCP"} <= set(df.columns)


def test_process_pcap_directory(tmp_path, pcap_file, conn_log_file):
    out = tmp_path / "features.csv"
    pcap_cli.main(["--pcap_dir", str(tmp_path), "--labels", conn_log_file,
                   "--output", str(out)])
    assert sorted(pd.read_csv(out)["label"]) == [0, 1, 1]


def test_process_pcap_output_feeds_ctu_training(tmp_path, pcap_file, conn_log_file):
    # The CSV written by iot-process-pcap must load cleanly in iot-train's CTU branch.
    out = tmp_path / "features.csv"
    pcap_cli.main(["--pcap", pcap_file, "--labels", conn_log_file, "--output", str(out)])
    features, labels = train_cli.load_features_labels(
        train_cli.parse_args(["--dataset", "ctu", "--features_csv", str(out)]))
    assert all(pd.api.types.is_numeric_dtype(t) for t in features.dtypes)
    assert labels.tolist().count(1) == 2


def test_process_pcap_exits_when_no_sessions(tmp_path, conn_log_file):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SystemExit) as exc:
        pcap_cli.main(["--pcap_dir", str(empty), "--labels", conn_log_file,
                       "--output", str(tmp_path / "out.csv")])
    assert exc.value.code == 1
    assert not (tmp_path / "out.csv").exists()


def test_process_pcap_rejects_label_directory(tmp_path, pcap_file):
    with pytest.raises(ValueError, match="directory"):
        pcap_cli.main(["--pcap", pcap_file, "--labels", str(tmp_path),
                       "--output", str(tmp_path / "out.csv")])
