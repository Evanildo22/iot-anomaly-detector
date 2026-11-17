# IoT Anomaly Detector

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue?logo=python)](https://www.python.org/)
[![TensorFlow 2.13+](https://img.shields.io/badge/TensorFlow-2.13%2B-orange?logo=tensorflow)](https://tensorflow.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://black.readthedocs.io/)

> **Network-level anomaly detection for IoT devices** using a two-phase
> Convolutional Autoencoder with Temporal Attention — supporting raw PCAP
> (CTU-IoT-Malware) and pre-extracted CSV (N-BaIoT) datasets through a
> unified preprocessing pipeline.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Results](#results)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

IoT devices are high-value targets — they are resource-constrained, often
unpatched, and generate large volumes of repetitive network traffic that is
ideal for statistical profiling. This project learns the "normal" traffic
fingerprint of a device using an **unsupervised Convolutional Autoencoder**
(Phase 1), then fine-tunes a **temporal attention classifier** (Phase 2)
to distinguish benign sessions from attack traffic such as Mirai, BASHLITE,
and other IoT malware families.

**Key features:**

- Two-phase training: unsupervised pretraining + supervised fine-tuning
- Temporal self-attention over Conv1D bottleneck embeddings
- Unified pipeline for both raw PCAP and pre-extracted N-BaIoT features
- SMOTE + random under-sampling for class imbalance correction
- Bayesian-optimised hyperparameters with full reproducibility controls
- Real-time streaming inference mode
- Clean modular package — importable as a library or run from CLI

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT  (n_features × 1)                      │
└───────────────────────────────────┬─────────────────────────────────┘
                                    │
                   ┌────────────────▼────────────────┐
                   │           E N C O D E R          │
                   │  Conv1D(F, k) → MaxPool → Drop   │
                   │  Conv1D(F/2, k) → MaxPool        │
                   │    bottleneck: (time/4, F/2)     │
                   └────────┬──────────────┬──────────┘
                            │              │
               ┌────────────▼──┐    ┌──────▼──────────────┐
               │   DECODER     │    │  TEMPORAL ATTENTION  │
               │ (Phase 1 only │    │  Permute → Dense     │
               │  pretraining) │    │  (softmax) → Permute │
               │  MSE loss     │    │  → Multiply          │
               └───────────────┘    └──────────┬───────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │   CLASSIFIER HEAD    │
                                    │  Flatten → Dense(ReLU│
                                    │  + L1/L2) → Dropout  │
                                    │  → Dense(sigmoid)    │
                                    └──────────┬───────────┘
                                               │
                                    ┌──────────▼───────────┐
                                    │   OUTPUT  0/1        │
                                    │  (Benign / Malicious)│
                                    └──────────────────────┘
```

### Training Phases

**Phase 1 — Unsupervised Pretraining**

The full autoencoder is trained to reconstruct normalised input features
using mean-squared-error loss, on **real benign training rows only** (no
attack traffic, no SMOTE-synthesised rows). This forces the encoder to learn
a compact, noise-robust representation of normal IoT traffic.

**Phase 2 — Supervised Fine-tuning**

The encoder weights are shared with a classifier head. Temporal attention
weights each time-step of the bottleneck embedding before classification,
focusing the model on the most anomaly-discriminative temporal features.
Training uses binary cross-entropy with an optimal F1-threshold tuned on
the validation set.

---

## Pipeline

```
Raw Data Sources
      │
      ├── PCAP files (CTU-IoT-Malware)          CSV files (N-BaIoT)
      │         │                                       │
      ▼         ▼                                       ▼
  PcapParser.parse_file()               load_nbaiot_dataset()
  (session extraction,                  (file concatenation,
   feature computation)                  binary labelling)
      │                                       │
      └─────────────────┬─────────────────────┘
                        │
                        ▼
       Trainer.fit_from_features()
         1. Stratified split of RAW rows, before any fitting:
            train 65% · val 15% · test 15% · real-time 5%
                        │
                        ▼
         2. DataPreprocessor.fit_transform()  ← TRAIN rows only
              - Correlation pruning (threshold 0.80)
              - Feature selection via RF importance
              - Median imputation → StandardScaler → PCA (95% var.)
              - SMOTE oversampling + random under-sampling
            DataPreprocessor.transform()      ← val / test / real-time
              (same fitted pipeline, no resampling: real rows,
               natural class balance)
                        │
                        ▼
              X_3d  (n_samples, n_pca_components, 1)
                        │
              ┌─────────┴──────────────────────┐
              │ Phase 1: AE MSE (benign only)  │
              │ Phase 2: Clf BCE               │
              │ Threshold tuning (val)         │
              └─────────┬──────────────────────┘
                        │
              ┌─────────┴──────────┐
              │ evaluate() / plot()│
              │  AUC · F1 · CM     │
              └────────────────────┘
```

---

## Datasets

### CTU-IoT-Malware Capture

Raw PCAP files from infected IoT devices recorded at the CTU University lab.
Sessions are extracted with bidirectional 5-tuple keys and labelled from
Zeek `conn.log.labeled` files.

| Scenario | Device | Malware family |
|---|---|---|
| 1-1 | Philips HUE | Mirai |
| 3-1 | Raspberry Pi | Mirai |
| 5-1 | Torii botnet | Torii |
| … | … | … |

**Download:** https://mcfp.felk.cvut.cz/publicDatasets/IoT-Malware-Capture/

### N-BaIoT

Pre-extracted statistical features from nine commercial IoT devices infected
with Mirai and BASHLITE. Five Lambda time windows (L1–L5) per flow direction.

| Device | Normal traffic | Attack variants |
|---|---|---|
| Danmini Doorbell | ✓ | Mirai × 5, BASHLITE × 5 |
| Ecobee Thermostat | ✓ | Mirai × 5, BASHLITE × 5 |
| Ennio Doorbell | ✓ | Mirai × 5 |
| … | … | … |

**Download:** https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot

---

## Project Structure

```
iot-anomaly-detector/
├── BUG_ANALYSIS.md                   ← full audit of original code
├── README.md
├── requirements.txt
├── setup.py
│
├── src/
│   └── iot_anomaly/
│       ├── data/
│       │   ├── parser.py             ← PCAP → session dicts (idle-timeout)
│       │   ├── features.py           ← all session-level feature functions
│       │   ├── label_mapper.py       ← conn.log.labeled → binary labels
│       │   └── preprocessor.py      ← sklearn pipeline + SMOTE
│       ├── models/
│       │   ├── autoencoder.py        ← encoder + full CAE (MSE loss)
│       │   └── classifier.py        ← attention head + KerasAnomalyClassifier
│       ├── training/
│       │   └── trainer.py           ← two-phase trainer, canonical splits
│       └── evaluation/
│           └── evaluator.py         ← metrics, ROC, CM, real-time inference
│
├── scripts/
│   ├── process_pcap.py              ← CLI: PCAP → features CSV
│   └── train.py                    ← CLI: train + evaluate
│
└── configs/
    └── default.yaml                ← hyperparameter reference
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Evanildo22/iot-anomaly-detector.git
cd iot-anomaly-detector

# 2. Create environment
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 3a. Train on N-BaIoT (pre-extracted CSV — fastest to reproduce)
python scripts/train.py \
    --dataset nbaiot \
    --data_folder /path/to/N-BaIoT/Dataset \
    --summary_csv  /path/to/N-BaIoT/Dataset/data_summary.csv \
    --save_dir checkpoints/nbaiot

# 3b. Train on CTU-IoT-Malware (raw PCAP — requires dpkt)
python scripts/process_pcap.py \
    --pcap   /path/to/capture.pcap \
    --labels /path/to/bro/conn.log.labeled \
    --output data/processed/ctu_features.csv

python scripts/train.py \
    --dataset ctu \
    --features_csv data/processed/ctu_features.csv \
    --save_dir checkpoints/ctu
```

---

## Usage

### As a library

```python
from iot_anomaly.data import PcapParser
from iot_anomaly.data.preprocessor import DataPreprocessor
from iot_anomaly.training.trainer import Trainer, TrainingConfig
from iot_anomaly.evaluation.evaluator import plot_roc_curve

# 1. Extract sessions from PCAP
parser = PcapParser(session_timeout=120.0)
df = parser.parse_file_with_labels("capture.pcap", "bro/conn.log.labeled")

# 2. Separate features and labels (drop identifiers and the raw protocol string)
features = df.drop(columns=["label", "session_id", "src_ip", "dst_ip",
                            "timestamp_session", "protocol"])
labels   = df["label"]

# 3. Train — the trainer splits the raw rows first, then fits feature
#    selection, PCA and SMOTE on the training split only (no leakage)
prep = DataPreprocessor(n_top_features=15, pca_variance=0.95, resample=True)
cfg = TrainingConfig(filters=112, kernel_size=6, dense_units=43,
                     dropout_rate=0.43, learning_rate=3.01e-4)
trainer = Trainer(cfg)
trainer.fit_from_features(features, labels, prep, pretrain=True)

# 4. Evaluate on the held-out test split (real rows, natural class balance)
results = trainer.evaluate()
print(results)  # {'accuracy': …, 'f1': …, 'auc_roc': …}

# 5. Real-time inference on new traffic, using the pipeline fitted in step 3
X_new_3d = prep.transform(new_features)
predictions = trainer.predict_realtime(X_new_3d)
```

### Real-time streaming

```python
from iot_anomaly.evaluation.evaluator import realtime_inference

realtime_inference(
    model=trainer.classifier,
    X_stream=X_realtime,
    threshold=trainer.threshold,
    batch_size=64,
)
```

---

## Results

N-BaIoT, all nine devices combined (7,062,606 rows, 92.1% malicious), with
the default hyperparameters and seed 42, evaluated on a held-out test split
of 1,059,391 real rows:

| Metric | 2024 report | 2026 leak-free pipeline |
|---|---|---|
| Accuracy | 99.41% | **99.88%** |
| Precision | 99.38% | **99.95%** |
| Recall | 99.54% | **99.92%** |
| F1-Score | 99.46% | **99.93%** |
| AUC-ROC | 99.97% | **99.97%** |
| False-positive rate | — | **0.57%** |

**How the evaluation is protected.** The raw rows are split first
(stratified 65/15/15/5). Feature selection, scaling, PCA and SMOTE are fitted
on the training split only, so the test set contains real rows with their
natural class balance and no synthetic rows.

**How to read these numbers.**

- The two columns are at the same general level, but the 2026 run is not a
  direct reproduction of the 2024 one: it uses a shortened schedule
  (2 pretraining + 3 fine-tuning epochs) and a different, leak-free
  evaluation protocol. Full-schedule training (30 + 50 epochs) is still to
  be rerun.
- Because 92.1% of rows are malicious, flagging everything would already
  reach 92.1% accuracy. The false-positive rate (0.57% of benign traffic
  flagged) is the more telling figure for the benign class.
- Rows are split at random, so test traffic comes from the same devices as
  the training traffic. Evaluation on held-out devices is planned.

Full details, including exact confusion counts, are in
[test-outputs/VALIDATION_2026-11-16.md](test-outputs/VALIDATION_2026-11-16.md).

---

## Configuration

All hyperparameters are controlled through `TrainingConfig` or CLI flags:

| Parameter | Default | Description |
|---|---|---|
| `filters` | 112 | Conv1D base filter count |
| `kernel_size` | 6 | Convolution kernel width |
| `dense_units` | 43 | Classifier hidden units |
| `dropout_rate` | 0.43 | Dropout fraction |
| `learning_rate` | 3.01e-4 | Adam learning rate |
| `l1_reg` | 2.9e-6 | L1 regularisation |
| `l2_reg` | 6.25e-4 | L2 regularisation |
| `pretrain_epochs` | 30 | Autoencoder pretraining |
| `finetune_epochs` | 50 | Classifier fine-tuning |
| `batch_size` | 256 | Training batch size |
| `val_size` | 0.15 | Validation split (15%) |
| `test_size` | 0.15 | Test split (15%) |
| `realtime_size` | 0.05 | Real-time eval split (5%) |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feat/my-improvement`)
3. Run the test suite (`pytest tests/ -v`)
4. Format the code (`black src/ scripts/ && isort src/ scripts/`)
5. Open a pull request

Please read [BUG_ANALYSIS.md](BUG_ANALYSIS.md) to understand the bugs fixed
in this refactor before making architectural changes.

---

## License

MIT — see [LICENSE](LICENSE) for details.

---

## Citation

Research building on this code can cite it as:

```bibtex
@misc{iot-anomaly-detector,
  title  = {IoT Anomaly Detector: CAE with Temporal Attention},
  author = {Evanildo Ribeiro},
  year   = {2026},
  url    = {https://github.com/Evanildo22/iot-anomaly-detector}
}
```

**Datasets:**

- Koroniotis et al. (2019). *Towards the Development of Realistic Botnet Dataset
  in the Internet of Things for Network Forensic Analytics: Bot-IoT Dataset.*
- Meidan et al. (2018). *N-BaIoT: Network-based Detection of IoT Botnet Attacks
  Using Deep Autoencoders.* IEEE Pervasive Computing.
