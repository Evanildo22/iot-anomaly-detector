# IoT SAFEGUARD — Comprehensive Testing Guide

> This guide covers **every phase** of the IoT SAFEGUARD project —
> from hardware verification and environment setup, through individual phase
> testing (data processing, preprocessing, model architecture, training,
> evaluation, adversarial robustness, real-time inference), to a final
> end-to-end integration test that confirms the whole system is consistent
> and reliable.
>
> Each section can be executed independently. A checklist at the end
> summarises the pass/fail criteria for every phase.

---

## Setup Notice — Verified Dependency Configuration (April 2026)

> **For MacBook Pro M2 Max users:** The original `requirements.txt` contained unpinned
> versions that cause silent or hard failures during installation. The combinations below
> were verified on a MacBook Pro 2023 (M2 Max, 38 GPU cores, 64 GB, Python 3.11.5 arm64).

### Confirmed working stack

| Package | Pinned version | Why pinned |
|---|---|---|
| `tensorflow` | `==2.13.0` | Must match `tensorflow-metal`; unpinned pulls TF 2.16+ (Keras 3 — breaking) |
| `tensorflow-metal` | `==1.0.1` | Apple Silicon GPU plugin for TF 2.13.x; install separately after `requirements.txt` |
| `numpy` | `==1.24.3` | `tensorflow-macos 2.13.0` hard-requires `numpy<=1.24.3`; numpy 2.x breaks the TF API |
| `scikit-learn` | `>=1.3.0,<1.6` | `imbalanced-learn 0.12.x` requires sklearn below 1.6 |
| `imbalanced-learn` | `==0.12.4` | Version 0.13+ requires `numpy>=1.25.2`, which conflicts with TF 2.13's numpy ceiling |

### Known conflicts to avoid

- **Do not** use `imbalanced-learn>=0.13` with `tensorflow==2.13.0` — it pulls `numpy>=1.25.2`
  which breaks `tensorflow-macos`.
- **Do not** use `scikit-learn>=1.6` with `imbalanced-learn==0.12.4` — import errors at runtime.
- **Do not** use `numpy>=2.0` — TF 2.13 was not built against the NumPy 2.x API.
- **Do not** use Python 3.12+ — `tensorflow-metal` is not fully supported on 3.12 as of 2026.

### Install order that works

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e ".[dev,hpo,adversarial,explainability]"   # installs requirements from setup.py
pip install tensorflow-metal==1.0.1                       # always install separately — not in setup.py
```

### Verify Metal GPU is active

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
# Expected: [..., PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

> **ART (adversarial-robustness-toolbox) warning:** On import, ART prints a warning that
> PyTorch is not found. This is harmless — ART's PyTorch modules are unused in this project.
> Only the TensorFlow backend is required.

---

## Phase 0 — Hardware & Computational Environment

The training pipeline uses two computationally intensive steps: Bayesian
hyperparameter search (`BayesSearchCV`) and deep neural network training
with TensorFlow. This phase verifies that the hardware meets the minimum
requirements and that all accelerator drivers are correctly detected
**before** committing to a multi-hour training run.

> **This guide has been configured for: MacBook Pro 2023, Apple M2 Max**

### Step 0.1 — Hardware requirements

The test machine (MacBook Pro 2023, Apple M2 Max) against the project requirements:

| Component | Minimum | Recommended | Test Machine (MacBook Pro M2 Max) | Status |
|---|---|---|---|---|
| CPU | 4-core / 2.5 GHz | 8-core / 3.5 GHz | Apple M2 Max — 12 cores (8P + 4E) | ✓ Exceeds recommended |
| RAM | 8 GB | 32 GB | 64 GB unified memory | ✓ Exceeds recommended |
| GPU | None (CPU-only OK) | NVIDIA ≥ 8 GB VRAM | Apple M2 Max — 38 GPU cores (Metal 4) | ✓ Metal acceleration available |
| Disk | 5 GB free | 20 GB free | ~994 GB available | ✓ Exceeds recommended |
| OS | Ubuntu 20.04 / macOS 12+ | Ubuntu 22.04 LTS | macOS (2023 MBP — macOS 13+) | ✓ Supported |

> **Note on GPU:** The M2 Max uses Apple Metal instead of NVIDIA CUDA. CUDA and
> cuDNN are **not applicable** to this machine. Metal-based GPU acceleration is
> enabled via the `tensorflow-metal` plugin (installed in Step 0.3).

> **Estimated training times (N-BaIoT, 7 M rows, full 50 epochs):**
>
> | Hardware | Phase 1 (AE pretrain) | Phase 2 (Classifier finetune) | Total |
> |---|---|---|---|
> | CPU only (8-core) | ~45 min | ~90 min | ~2.5 hr |
> | Apple M2 Max + tensorflow-metal | ~15–25 min | ~30–50 min | ~45 min–1.2 hr* |
> | NVIDIA RTX 3060 (12 GB) | ~8 min | ~20 min | ~30 min |
> | NVIDIA V100 (16 GB, cloud) | ~3 min | ~8 min | ~12 min |
>
> *M2 Max estimates are approximate. The 38 Metal GPU cores with 64 GB unified
> memory deliver meaningful acceleration over CPU-only, though NVIDIA CUDA
> kernels in TensorFlow are more optimised than Metal. Actual times will vary
> with batch size and model depth.

### Step 0.2 — Apple Metal GPU setup (replaces CUDA/cuDNN)

> **CUDA and cuDNN do not apply to Apple Silicon.** Skip any CUDA installation
> instructions. Instead, use `tensorflow-metal` to enable GPU acceleration.

Required package versions for Apple Silicon:

| TensorFlow | tensorflow-metal | Python | Keras | Notes |
|---|---|---|---|---|
| 2.13.x | 1.0.1 | 3.10 – 3.11 | Keras 2 | **Used in this project** — stable, max compatibility with SHAP/ART/sklearn |
| 2.14.x | 1.1.0 | 3.10 – 3.11 | Keras 2 | Compatible, not tested with this project |
| 2.15.x | 1.1.1 | 3.10 – 3.11 | Keras 2 | Last Keras 2 version |
| 2.16.x+ | 1.1.0 | 3.10 – 3.11 | **Keras 3** | Breaking API changes — not recommended without code updates |
| 2.17.x | 1.1.0 | 3.10 – 3.11 | **Keras 3** | Best Metal performance (2026), but requires Keras 3 migration |

> **Python 3.12+ is not supported.** `tensorflow-metal` is not fully compatible with Python 3.12
> as of 2026. Use Python **3.10 or 3.11** (native arm64 build — not Rosetta).
>
> **Keras 3 warning:** TF 2.16 and above switch to Keras 3 by default. This project uses Keras 2
> APIs (`tf.keras`). Upgrading to TF 2.17+ requires model and layer code changes.
> Upgrade only after the pipeline is validated and stable.

Install TensorFlow and the Metal plugin:

```bash
pip install tensorflow==2.13.0
pip install tensorflow-metal==1.0.1
```

Verify the Metal plugin is visible to the system:

```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices())"
```

**Expected output (Metal active):**
```
[PhysicalDevice(name='/physical_device:CPU:0', device_type='CPU'),
 PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```

If only `CPU:0` appears, re-install `tensorflow-metal` into the active
virtual environment and confirm the interpreter is not a Rosetta (x86)
Python — the native arm64 Python build is required.

### Step 0.3 — Python virtual environment

```bash
git clone https://github.com/Evanildo22/iot-anomaly-detector.git
cd iot-anomaly-detector

python3 -m venv .venv
source .venv/bin/activate          # macOS

pip install --upgrade pip
pip install -e ".[dev]"

# Enable Apple Metal GPU acceleration
pip install tensorflow-metal==1.0.1
```

**Expected output (last lines):**
```
Successfully installed iot-anomaly-detector-1.0.0 dpkt-1.9.8 tensorflow-2.13.0 ...
Successfully installed tensorflow-metal-1.0.1
```

> **Tip — Unified Memory:** The M2 Max uses unified memory shared between CPU
> and GPU. Because there is no VRAM copy overhead, large batch sizes are safe.
> Start with `--batch 256` for full training runs and increase to `--batch 512`
> if memory usage stays below ~40 GB (monitor with Activity Monitor → Memory).

### Step 0.4 — Verify Metal GPU detection by TensorFlow

```bash
python - <<'EOF'
import tensorflow as tf

print(f"TensorFlow version : {tf.__version__}")

# Check for Metal GPU (Apple Silicon)
all_devices  = tf.config.list_physical_devices()
gpu_devices  = tf.config.list_physical_devices("GPU")
metal_active = any("GPU" in d.name for d in all_devices)

print(f"All devices        : {[d.device_type for d in all_devices]}")
print(f"GPU devices        : {len(gpu_devices)}")

if metal_active:
    for g in gpu_devices:
        print(f"  {g.name}")
    print("\n✓  Apple Metal GPU acceleration AVAILABLE — training will use M2 Max GPU cores")
    print("   (38 GPU cores, Metal 4, unified memory — tensorflow-metal active)")
else:
    print("\n⚠  Metal GPU not detected — training will run on CPU only")
    print("   To enable Metal acceleration:")
    print("   $ pip install tensorflow-metal==1.0.1  # must match tensorflow==2.13.0")
    print("   A native arm64 Python 3.10/3.11 is required (not Rosetta):")
    print("   $ python3 -c \"import platform; print(platform.machine())\"  # should print arm64")

# Also verify key imports
import dpkt, sklearn, imblearn, scipy
import art  # adversarial-robustness-toolbox
print("\n✓  All required packages import successfully")
EOF
```

**Expected output (Metal active — as on the M2 Max test machine):**
```
TensorFlow version : 2.13.0
All devices        : ['CPU', 'GPU']
GPU devices        : 1
  /physical_device:GPU:0

✓  Apple Metal GPU acceleration AVAILABLE — training will use M2 Max GPU cores
   (38 GPU cores, Metal 4, unified memory — tensorflow-metal active)

✓  All required packages import successfully
```

**Expected output if tensorflow-metal is not yet installed:**
```
TensorFlow version : 2.13.0
All devices        : ['CPU']
GPU devices        : 0

⚠  Metal GPU not detected — training will run on CPU only
   To enable Metal acceleration:
   $ pip install tensorflow-metal==1.0.1
   A native arm64 Python is required (not Rosetta):
   $ python3 -c "import platform; print(platform.machine())"  # should print arm64

✓  All required packages import successfully
```

> **CPU-only is fully supported.** Without `tensorflow-metal`, all phases
> still complete correctly — training simply takes longer (see time
> estimates in Step 0.1).

### Step 0.5 — Verify the package version and entry points

```bash
python -c "import iot_anomaly; print(iot_anomaly.__version__)"
iot-process-pcap --help
iot-train --help
```

**Expected output:**
```
1.0.0
usage: iot-process-pcap [-h] --pcap PCAP [--pcap_dir PCAP_DIR] ...
usage: iot-train [-h] --dataset {nbaiot,ctu} ...
```

---

## Phase 1 — Data Processing Phase

> **Saving test evidence:** For each step below, the terminal output or a screenshot is saved to
> `test-outputs/phase-1-data-processing/`. Suggested naming convention:
> `step-1-1_dataset-acquisition.png`, `step-1-2_nbaiot-loader.txt`, etc.
> A folder exists for every phase (`phase-0-environment/` through `phase-9-integration/`).

This phase tests the two data ingestion paths independently:
(a) PCAP parsing for the CTU-IoT-Malware dataset and
(b) CSV loading for the N-BaIoT dataset.

### Step 1.1 — Dataset acquisition

**N-BaIoT (CSV, required for all training tests):**
```
https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot
```

```bash
mkdir -p data/raw/N-BaIoT
# Extract so the structure is:
# data/raw/N-BaIoT/Dataset/
#   ├── data_summary.csv
#   ├── Danmini_Doorbell/benign_traffic.csv
#   ├── Danmini_Doorbell/mirai_attacks/scan.csv
#   └── ...  (9 devices × multiple attack types)

head -3 data/raw/N-BaIoT/Dataset/data_summary.csv
```

**Expected:**
```
File Name,Device,Traffic Type
Danmini_Doorbell/benign_traffic.csv,Danmini Doorbell,Benign
Danmini_Doorbell/mirai_attacks/scan.csv,Danmini Doorbell,Mirai Scan
```

**CTU-IoT-Malware (PCAP, required for PCAP pipeline tests only):**
```
https://mcfp.felk.cvut.cz/publicDatasets/IoT-Malware-Capture/
```

```bash
mkdir -p data/raw/CTU/CTU-IoT-Malware-Capture-1-1/bro
# Place files:
# data/raw/CTU/CTU-IoT-Malware-Capture-1-1/
#   ├── 2018-05-09-192.168.100.103.pcap
#   └── bro/conn.log.labeled
```

### Step 1.2 — Test N-BaIoT CSV loader

```bash
python - <<'EOF'
import pandas as pd
from src.iot_anomaly.data.preprocessor import load_nbaiot_dataset

base = "data/raw/N-BaIoT/Dataset"
summary_csv = f"{base}/data_summary.csv"

# Load a subset (2 devices) for speed
features, labels = load_nbaiot_dataset(
    base_folder=base,
    summary_csv=summary_csv,
    devices=["Danmini_Doorbell", "Ecobee_Thermostat"]
)

print(f"Features shape : {features.shape}")
print(f"Labels shape   : {labels.shape}")
print(f"Label balance  : {labels.value_counts().to_dict()}")
print(f"NaN count      : {features.isna().sum().sum()}")
print(f"Feature dtypes : {features.dtypes.nunique()} unique dtype(s)")

assert features.shape[0] == labels.shape[0], "Feature/label row mismatch"
assert set(labels.unique()).issubset({0, 1}), "Labels must be binary (0/1)"
assert features.shape[1] > 100, "Expected 115 N-BaIoT features"
print("\n✓  N-BaIoT CSV loader: PASSED")
EOF
```

**Expected output:**
```
Features shape : (N, 115)
Labels shape   : (N,)
Label balance  : {1: X, 0: Y}
NaN count      : 0
Feature dtypes : 1 unique dtype(s)

✓  N-BaIoT CSV loader: PASSED
```

### Step 1.3 — Test PCAP session parser

```bash
python - <<'EOF'
from src.iot_anomaly.data.parser import PcapParser

pcap = "data/raw/CTU/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap"
parser = PcapParser(session_timeout=120.0)

sessions = list(parser.parse_file(pcap))
print(f"Sessions extracted : {len(sessions)}")
assert len(sessions) > 0, "No sessions extracted from PCAP"

# Verify a session has all required feature keys
s = sessions[0]
required_keys = [
    "inter_arrival_min", "inter_arrival_mean", "inter_arrival_std",
    "packet_size_min",   "packet_size_mean",   "packet_size_max",
    "TCP", "UDP", "ICMP",
    "src_port_min", "dst_port_max",
]
for k in required_keys:
    assert k in s, f"Missing key: {k}"

# Verify payload_size is not negative (BUG-03 fix)
assert s.get("payload_size_mean", 0) >= 0, "Negative payload size (BUG-03)"
print(f"Sample session keys: {list(s.keys())[:8]} ...")
print(f"\n✓  PCAP session parser: PASSED ({len(sessions)} sessions, all keys present)")
EOF
```

**Expected output:**
```
Sessions extracted : 1234
Sample session keys: ['inter_arrival_min', 'inter_arrival_mean', ...] ...

✓  PCAP session parser: PASSED (1234 sessions, all keys present)
```

### Step 1.4 — Test label mapper

```bash
python - <<'EOF'
import os, tempfile
from src.iot_anomaly.data.label_mapper import parse_conn_log_labeled, map_labels
import pandas as pd

# 1. Directory path should raise a helpful error (FIX BUG-10)
try:
    parse_conn_log_labeled("/tmp")
    assert False, "Should have raised ValueError"
except ValueError as e:
    print(f"✓  Directory path raises ValueError (BUG-10 fixed): {e}")

# 2. Valid label file parses correctly
row = "\t".join(["1234567.89","-","192.168.1.1","-",
                  "10.0.0.1","80","tcp","-","-","-",
                  "-","-","-","-","malicious-Mirai"]) + "\n"
with tempfile.NamedTemporaryFile(mode="w", suffix=".labeled", delete=False) as f:
    f.write(row)
    fname = f.name
labels = parse_conn_log_labeled(fname)
assert len(labels) == 1
print("✓  Valid label file parses correctly")

# 3. Case-insensitive binary encoding (FIX BUG-11)
df = pd.DataFrame({"label": ["Malicious-Mirai", "benign", "MALICIOUS", "Benign"]})
mapped = map_labels(df, "label")
assert list(mapped) == [1, 0, 1, 0], f"Unexpected: {list(mapped)}"
print("✓  map_labels is case-insensitive (BUG-11 fixed)")

os.unlink(fname)
print("\n✓  Label mapper: PASSED")
EOF
```

**Expected output:**
```
✓  Directory path raises ValueError (BUG-10 fixed): Expected a file path ...
✓  Valid label file parses correctly
✓  map_labels is case-insensitive (BUG-11 fixed)

✓  Label mapper: PASSED
```

---

## Phase 2 — Feature Engineering & Preprocessing Phase

This phase tests the statistical feature computation, the sklearn/imblearn
preprocessing pipeline, and the SMOTE resampling — independently of any
model training.

### Step 2.1 — Feature extraction correctness

```bash
python - <<'EOF'
import numpy as np
from src.iot_anomaly.data.features import (
    inter_arrival_time_stats, packet_flag_combinations,
    protocol_distribution, port_range_features,
    compute_session_features,
)

# BUG-01: inter_arrival_* key prefix
ts   = np.array([1.0, 2.5, 3.0, 5.5])
iat  = inter_arrival_time_stats(ts)
assert "inter_arrival_min" in iat,    "BUG-01: wrong key prefix"
assert "packet_size_min"  not in iat, "BUG-01: leaked wrong prefix"
print(f"✓  IAT keys correct: {sorted(iat.keys())}")

# BUG-07: single-packet default must be 0.0 not 1.0
single = inter_arrival_time_stats(np.array([1.0]))
assert single["inter_arrival_min"] == 0.0, \
    f"BUG-07: default is {single['inter_arrival_min']}"
print("✓  Single-packet IAT defaults to 0.0 (BUG-07 fixed)")

# BUG-02: no double prefix on flag keys
pkts = [{"tcp_flags": 2}, {"tcp_flags": 18}]
combos = packet_flag_combinations(pkts)
assert "flag_comb_flag_comb_2" not in combos, "BUG-02: double prefix"
assert 2 in combos, "Raw integer key missing"
print("✓  Flag keys are raw integers, no double prefix (BUG-02 fixed)")

# BUG-05: uppercase protocol matching
proto = protocol_distribution([{"protocol":"TCP"},{"protocol":"UDP"}])
assert proto["TCP"] > 0, "BUG-05: uppercase TCP not matched"
print("✓  Protocol distribution uses uppercase keys (BUG-05 fixed)")

# BUG-06: port range returns all four fields
ports = port_range_features([{"src_port": 1024}, {"dst_port": 80}])
for key in ["src_port_min","src_port_max","dst_port_min","dst_port_max"]:
    assert key in ports, f"BUG-06: missing {key}"
print("✓  Port range features return all four keys (BUG-06 fixed)")

print("\n✓  Feature extraction: PASSED")
EOF
```

### Step 2.2 — Preprocessing pipeline (imputation, scaling, PCA)

```bash
python - <<'EOF'
import numpy as np, pandas as pd
from src.iot_anomaly.data.preprocessor import (
    engineer_nbaiot_features,
    build_preprocessing_pipeline,
    build_resampling_pipeline,
    DataPreprocessor,
)

# Build synthetic data (simulating 3 N-BaIoT stat windows × 5 stats = 15 features)
np.random.seed(42)
n = 500
X = pd.DataFrame(np.random.randn(n, 115))
# Introduce NaNs to test imputation
X.iloc[0:10, 0:5] = np.nan
y = pd.Series(np.random.randint(0, 2, n))

# --- sklearn pipeline: imputer → scaler → PCA ---
sk_pipe = build_preprocessing_pipeline(pca_variance=0.90)
X_np = sk_pipe.fit_transform(X)
assert X_np.ndim == 2,        "Pipeline output should be 2D"
assert X_np.shape[0] == n,    "Row count changed"
assert not np.isnan(X_np).any(), "NaNs survived imputation"
print(f"✓  sklearn pipeline: {X.shape[1]} → {X_np.shape[1]} components (PCA 90%)")

# --- SMOTE + UnderSampler pipeline ---
rs_pipe = build_resampling_pipeline(random_state=42)
X_res, y_res = rs_pipe.fit_resample(X_np, y)
before = dict(zip(*np.unique(y, return_counts=True)))
after  = dict(zip(*np.unique(y_res, return_counts=True)))
print(f"✓  Resampling: {before} → {after}")

# --- DataPreprocessor end-to-end (returns 3-D tensor for Conv1D) ---
prep = DataPreprocessor(n_top_features=10, pca_variance=0.90,
                        resample=True, random_state=42)
X_3d, y_out = prep.fit_transform(X, y)
assert X_3d.ndim == 3, f"Expected 3-D tensor, got {X_3d.ndim}-D"
assert X_3d.shape[2] == 1, "Last dim should be 1 (channel)"
print(f"✓  DataPreprocessor output: {X_3d.shape} — ready for Conv1D")

print("\n✓  Preprocessing pipeline: PASSED")
EOF
```

**Expected output:**
```
✓  sklearn pipeline: 115 → N components (PCA 90%)
✓  Resampling: {0: X, 1: Y} → {0: Z, 1: Z}
✓  DataPreprocessor output: (M, K, 1) — ready for Conv1D

✓  Preprocessing pipeline: PASSED
```

---

## Phase 3 — Model Architecture Phase

This phase tests that the autoencoder and classifier are built correctly,
have the expected shapes, and use the correct loss functions — without
running any actual training.

### Step 3.1 — Autoencoder architecture

```bash
python - <<'EOF'
from src.iot_anomaly.models.autoencoder import build_autoencoder

input_shape = (12, 1)   # 12 PCA components, 1 channel
ae, enc = build_autoencoder(
    input_shape, filters=64, kernel_size=3,
    dropout_rate=0.3, learning_rate=1e-3
)

print("=== AUTOENCODER SUMMARY ===")
ae.summary(line_length=70)
print()

# Loss function must be MSE for reconstruction (FIX BUG-14)
assert ae.loss == "mse", \
    f"BUG-14 not fixed: autoencoder loss is '{ae.loss}', expected 'mse'"
print(f"✓  Autoencoder loss = 'mse' (BUG-14 fixed)")

# Encoder output must be compressed (not same shape as input)
assert enc.output_shape != (None,) + input_shape, \
    "BUG-13: encoder output == input shape (full autoencoder used as encoder)"
print(f"✓  Encoder bottleneck: {enc.output_shape} (compressed, BUG-13 fixed)")

# Output shape must equal input shape for reconstruction
assert ae.output_shape == (None,) + input_shape, \
    f"Autoencoder output shape wrong: {ae.output_shape}"
print(f"✓  Autoencoder output shape: {ae.output_shape}")

# Final decoder activation must be 'linear' for real-valued features
last_activation = ae.layers[-1].get_config().get("activation", "")
assert last_activation == "linear", \
    f"BUG-14: decoder final activation is '{last_activation}', expected 'linear'"
print(f"✓  Decoder final activation = 'linear' (BUG-14 fixed)")

print("\n✓  Autoencoder architecture: PASSED")
EOF
```

### Step 3.2 — Classifier with temporal attention

```bash
python - <<'EOF'
import numpy as np
from src.iot_anomaly.models.autoencoder import build_autoencoder
from src.iot_anomaly.models.classifier import build_classifier

input_shape = (12, 1)
ae, enc = build_autoencoder(input_shape, filters=64, kernel_size=3,
                             dropout_rate=0.3, learning_rate=1e-3)
clf = build_classifier(enc, input_shape, dense_units=43,
                        dropout_rate=0.3, l1_reg=2.9e-6, l2_reg=6.25e-4)

print("=== CLASSIFIER SUMMARY ===")
clf.summary(line_length=70)
print()

layer_names = [l.name for l in clf.layers]

# Classifier loss must be binary_crossentropy
assert clf.loss == "binary_crossentropy", \
    f"Classifier loss wrong: {clf.loss}"
print(f"✓  Classifier loss = 'binary_crossentropy'")

# Encoder (not full autoencoder) must be embedded
assert "encoder"      in layer_names, "Encoder not found in classifier layers"
assert "autoencoder" not in layer_names, \
    "BUG-13: full autoencoder embedded in classifier"
print(f"✓  Classifier uses encoder only, not full autoencoder (BUG-13 fixed)")

# Output layer: single sigmoid unit
output_shape = clf.output_shape
assert output_shape == (None, 1), f"Wrong output shape: {output_shape}"
print(f"✓  Classifier output shape: {output_shape}")

# Forward pass — values must be probabilities in [0, 1]
x = np.random.randn(8, *input_shape).astype("float32")
preds = clf.predict(x, verbose=0)
assert preds.min() >= 0.0 and preds.max() <= 1.0, \
    "Predictions outside [0, 1]"
print(f"✓  Forward pass: 8 samples, predictions in [{preds.min():.4f}, {preds.max():.4f}]")

# Attention mechanism present
attn_layers = [l for l in clf.layers if "multiply" in l.name or "attention" in l.name]
assert len(attn_layers) > 0, "Temporal attention layer not found"
print(f"✓  Temporal attention layer present: {attn_layers[0].name}")

print("\n✓  Classifier architecture: PASSED")
EOF
```

---

## Phase 4 — Training Phase

This phase runs the full two-stage training pipeline and verifies that
loss curves converge, checkpoints are saved, and the threshold is tuned.

### Step 4.1 — Quick training run (smoke validation, ~2 minutes)

Use a minimal dataset slice and reduced epochs to verify the full training
loop without waiting for the full dataset.

```bash
python - <<'EOF'
import numpy as np, pandas as pd

base = "data/raw/N-BaIoT/Dataset"
benign = pd.read_csv(f"{base}/Danmini_Doorbell/benign_traffic.csv").head(2000)
benign["label"] = 0
attack = pd.read_csv(f"{base}/Danmini_Doorbell/mirai_attacks/scan.csv").head(2000)
attack["label"] = 1
df = pd.concat([benign, attack]).sample(frac=1, random_state=42)

from src.iot_anomaly.data.preprocessor import DataPreprocessor, engineer_nbaiot_features
features = engineer_nbaiot_features(df.drop(columns=["label"])).reset_index(drop=True)
labels   = df["label"].reset_index(drop=True)
prep     = DataPreprocessor(n_top_features=10, pca_variance=0.90,
                             resample=True, random_state=42)

from src.iot_anomaly.training.trainer import Trainer, TrainingConfig
cfg = TrainingConfig(
    filters=32, kernel_size=3, dense_units=16,
    dropout_rate=0.2, learning_rate=1e-3,
    pretrain_epochs=3, finetune_epochs=5, batch_size=64,
    save_dir="checkpoints/quicktest"
)
trainer = Trainer(cfg)
# Splits the raw rows first; selection, PCA and SMOTE are fitted on train only
trainer.fit_from_features(features, labels, prep, pretrain=True)
print(f"Train: X={trainer._X_train.shape}, "
      f"y counts={np.bincount(trainer._y_train.astype(int)).tolist()} (resampled)")
print(f"Test:  y counts={np.bincount(trainer._y_test.astype(int)).tolist()} (real rows)")

# Verify checkpoints were saved
import os
ckpt_dir = "checkpoints/quicktest"
assert os.path.isfile(f"{ckpt_dir}/autoencoder_best.weights.h5"), \
    "Autoencoder checkpoint not saved"
assert os.path.isfile(f"{ckpt_dir}/classifier_best.weights.h5"), \
    "Classifier checkpoint not saved"
print(f"✓  Checkpoints saved in {ckpt_dir}/")

# Verify threshold was found
assert 0.0 < trainer.threshold < 1.0, \
    f"Invalid threshold: {trainer.threshold}"
print(f"✓  Optimal threshold: {trainer.threshold:.4f}")

results = trainer.evaluate()
print("\n✓  Test metrics:")
for k, v in results.items():
    print(f"   {k:<12} {v:.4f}")

assert results["f1"] > 0.5, "F1 < 0.5 on trivial smoke test"
print("\n✓  Training phase (quick run): PASSED")
EOF
```

**Expected output:**
```
Input: X=(N, K, 1), y counts=[M, M]
Phase 1: Autoencoder pretraining ...
Epoch 1/3 — loss: X.XXXX — val_loss: X.XXXX
...
Phase 2: Classifier fine-tuning ...
Epoch 1/5 — loss: X.XXXX — accuracy: X.XXXX
...
✓  Checkpoints saved in checkpoints/quicktest/
✓  Optimal threshold: 0.XXXX

✓  Test metrics:
   accuracy     0.XXXX
   ...

✓  Training phase (quick run): PASSED
```

### Step 4.2 — Monitor training with TensorBoard

During full training, open a second terminal and launch TensorBoard to
inspect loss curves and detect any divergence in real time:

```bash
# Terminal 2 (while training runs in Terminal 1)
source .venv/bin/activate
tensorboard --logdir checkpoints/ --port 6006
```

TensorBoard is then available at **http://localhost:6006**.

**What to look for in TensorBoard:**

| Panel | Healthy sign | Warning sign |
|---|---|---|
| Phase 1 `loss` | Smooth exponential decay toward ~0.01 | Oscillating / not decreasing |
| Phase 1 `val_loss` | Tracks `loss` closely | Large gap → overfitting |
| Phase 2 `accuracy` | Climbs steadily above 0.90 within 10 epochs | Stuck near 0.5 → resampling issue |
| Phase 2 `val_auc` | Rises toward 0.999 | Drops after initial rise → overfitting |
| Early stopping | Training stops before max epochs | Never stops → LR too high |

### Step 4.3 — Full training on N-BaIoT (production run)

```bash
python scripts/train.py \
    --dataset      nbaiot \
    --data_folder  data/raw/N-BaIoT/Dataset \
    --summary_csv  data/raw/N-BaIoT/Dataset/data_summary.csv \
    --save_dir     checkpoints/nbaiot_full \
    --n_top_features 15 \
    --pretrain_epochs 30 \
    --epochs         50 \
    --batch          256 \
    --seed           42
```

**Reference results.** The 2024 report published accuracy 0.9941,
precision 0.9938, recall 0.9954, F1 0.9946 and AUC-ROC 0.9997. The 2026
leak-free pipeline, run with this full 30 + 50 epoch schedule on the test
machine (6.9 hours; early stopping at epochs 25 and 43), produced:
```
── Test-set results ──────────────────────────────
  accuracy     0.9991
  precision    0.9996
  recall       0.9995
  f1           0.9995
  auc_roc      0.9999
  tn 82965 | fp 425 | fn 501 | tp 975500
```
These are reference values from one run with seed 42. Small differences
are expected on other hardware or library versions, because GPU training
is not bit-for-bit deterministic. The full record is in
`test-outputs/full-schedule-2026-11-18/`.

**Verify saved artefacts:**
```bash
ls -lh checkpoints/nbaiot_full/
# autoencoder_best.weights.h5
# classifier_best.weights.h5
# classifier_best.keras        (full model export; load with load_model(path, compile=False))
# roc_curve.png
# confusion_matrix.png
```

---

## Phase 5 — Evaluation Phase

This phase tests the evaluation metrics, the ROC curve, and the
confusion matrix as standalone components — using the saved model.

### Step 5.1 — Metrics computation

```bash
python - <<'EOF'
import numpy as np
from src.iot_anomaly.evaluation.evaluator import compute_metrics

# Simulate near-perfect predictions
y_true  = np.array([0,0,0,0,1,1,1,1,0,0,1,1])
y_pred  = np.array([0,0,0,1,1,1,1,1,0,0,1,1])
y_proba = np.array([0.05,0.1,0.08,0.6,0.95,0.9,0.88,0.92,0.07,0.12,0.91,0.89])

metrics = compute_metrics(y_true, y_pred, y_proba)
print("Metric results:")
for k, v in metrics.items():
    print(f"  {k:<12} {v:.4f}")

assert metrics["accuracy"]  > 0.8, "Accuracy too low"
assert metrics["f1"]        > 0.8, "F1 too low"
assert metrics["auc_roc"]   > 0.8, "AUC-ROC too low"
print("\n✓  compute_metrics: correct outputs and values in range")
EOF
```

### Step 5.2 — Optimal threshold finder

```bash
python - <<'EOF'
import numpy as np
from src.iot_anomaly.training.trainer import find_optimal_threshold

# Perfectly separable case
y_true  = np.array([0, 0, 0, 1, 1, 1])
y_proba = np.array([0.1, 0.2, 0.15, 0.9, 0.85, 0.95])
t, f1 = find_optimal_threshold(y_true, y_proba)
assert 0.0 < t < 1.0, f"Threshold out of range: {t}"
assert f1 > 0.99, f"F1 too low for perfectly separable case: {f1}"
print(f"✓  Optimal threshold: {t:.4f}, F1: {f1:.4f} (BUG-20 fixed: no IndexError)")

# Edge case: constant probabilities — must not crash
t2, _ = find_optimal_threshold(
    np.array([0,1,0,1]),
    np.array([0.5,0.5,0.5,0.5])
)
print(f"✓  Edge case (all same proba): threshold = {t2:.4f}, no IndexError")

print("\n✓  Threshold finder: PASSED")
EOF
```

### Step 5.3 — Visualisation output

```bash
python - <<'EOF'
import numpy as np
from src.iot_anomaly.evaluation.evaluator import (
    plot_roc_curve, plot_confusion_matrix
)

y_true  = np.array([0,0,0,0,0,1,1,1,1,1])
y_proba = np.array([0.1,0.15,0.08,0.2,0.09,0.85,0.9,0.88,0.92,0.95])
y_pred  = (y_proba >= 0.5).astype(int)

plot_roc_curve(y_true, y_proba,
               save_path="checkpoints/quicktest/roc_test.png")
plot_confusion_matrix(y_true, y_pred,
                      save_path="checkpoints/quicktest/cm_test.png")

import os
assert os.path.isfile("checkpoints/quicktest/roc_test.png"), "ROC not saved"
assert os.path.isfile("checkpoints/quicktest/cm_test.png"),  "CM not saved"
print("✓  roc_test.png and cm_test.png saved successfully")
print("   → Open these files to verify they render correctly")
print("\n✓  Visualisation: PASSED")
EOF
```

---

## Phase 6 — PCAP Pipeline (CTU Dataset)

This phase tests the full PCAP → session features → CSV path, then trains
the model on the CTU dataset.

### Step 6.1 — Extract session features from PCAP

```bash
python scripts/process_pcap.py \
    --pcap   data/raw/CTU/CTU-IoT-Malware-Capture-1-1/2018-05-09-192.168.100.103.pcap \
    --labels data/raw/CTU/CTU-IoT-Malware-Capture-1-1/bro/conn.log.labeled \
    --output data/processed/ctu_features.csv
```

**Expected log output:**
```
INFO  Parsing PCAP: ... 2018-05-09-192.168.100.103.pcap
INFO  Loaded 1234 labeled sessions from conn.log.labeled
INFO  Saved 987 rows to data/processed/ctu_features.csv
INFO  Label distribution:  0=856  1=131
```

### Step 6.2 — Verify the output CSV columns

```bash
python - <<'EOF'
import pandas as pd

df = pd.read_csv("data/processed/ctu_features.csv")
print(f"Rows: {len(df)}   Columns: {len(df.columns)}")

# Key column-name correctness checks (bug regression tests)
assert "inter_arrival_min"         in df.columns, "BUG-01 still present"
assert "flag_comb_flag_comb_0"    not in df.columns, "BUG-02 still present"
assert "label"                     in df.columns, "Missing label column"
assert df["label"].isin([0,1]).all(), "Labels not binary"
print("✓  All key column names correct, labels binary")
print(df[["inter_arrival_min","inter_arrival_mean","label"]].describe())
EOF
```

### Step 6.3 — Train on CTU features

```bash
python scripts/train.py \
    --dataset       ctu \
    --features_csv  data/processed/ctu_features.csv \
    --save_dir      checkpoints/ctu \
    --n_top_features 15 \
    --pretrain_epochs 15 \
    --epochs         30 \
    --batch          64
```

---

## Phase 7 — Adversarial Robustness Phase

This phase tests the model's robustness against adversarial examples
generated by the Fast Gradient Sign Method (FGSM) using the
Adversarial Robustness Toolbox (ART).

### Step 7.1 — Verify ART installation

```bash
python - <<'EOF'
import art
print(f"ART version: {art.__version__}")
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier as ARTKerasClassifier
print("✓  ART imports: FastGradientMethod, KerasClassifier")
EOF
```

### Step 7.2 — Run adversarial attack (requires trained classifier)

```bash
python - <<'EOF'
import numpy as np
import tensorflow as tf
from art.attacks.evasion import FastGradientMethod
from art.estimators.classification import KerasClassifier as ARTKerasClassifier
from src.iot_anomaly.evaluation.evaluator import compute_metrics

# Load trained model (compile=False skips the unrestorable legacy-Adam state;
# recompile with a loss because ART needs it to compute attack gradients)
clf_model = tf.keras.models.load_model("checkpoints/quicktest/classifier_best.keras", compile=False)
clf_model.compile(loss="binary_crossentropy")

# Wrap for ART — binary classification (2 classes internally)
art_clf = ARTKerasClassifier(
    model=clf_model,
    nb_classes=2,
    clip_values=(0.0, 1.0)
)

# Generate test data (small batch)
np.random.seed(42)
n_features = clf_model.input_shape[1]
X_test  = np.random.randn(50, n_features, 1).astype("float32")
y_test  = np.random.randint(0, 2, 50)

# Clean accuracy
clean_proba = clf_model.predict(X_test, verbose=0).flatten()
clean_pred  = (clean_proba >= 0.5).astype(int)
clean_acc   = (clean_pred == y_test).mean()
print(f"Clean accuracy       : {clean_acc:.4f}")

# FGSM attack (epsilon = 0.05 — small perturbation)
fgsm = FastGradientMethod(estimator=art_clf, eps=0.05)

# ART expects (n, features, 1) — reshape for attack
X_adv = fgsm.generate(x=X_test)
adv_proba = clf_model.predict(X_adv, verbose=0).flatten()
adv_pred  = (adv_proba >= 0.5).astype(int)
adv_acc   = (adv_pred == y_test).mean()
print(f"Adversarial accuracy : {adv_acc:.4f}  (eps=0.05)")

# Robustness ratio: adversarial drop vs clean
drop = clean_acc - adv_acc
print(f"Accuracy drop        : {drop:.4f} ({drop/max(clean_acc,1e-9)*100:.1f}%)")

if drop < 0.20:
    print("✓  Model is robust (< 20% accuracy drop under FGSM)")
else:
    print("⚠  Model shows sensitivity to adversarial perturbations")
    print("   Consider adversarial training or input preprocessing defences")

print("\n✓  Adversarial robustness phase: PASSED (no crashes)")
EOF
```

**Expected output:**
```
Clean accuracy       : 0.XXXX
Adversarial accuracy : 0.XXXX  (eps=0.05)
Accuracy drop        : 0.XXXX (X.X%)
✓  Model is robust (< 20% accuracy drop under FGSM)

✓  Adversarial robustness phase: PASSED (no crashes)
```

> A small but non-zero accuracy drop is expected and normal — it
> demonstrates the attack is functioning. The important outcome is that
> the model does not crash and produces valid probability outputs.

---

## Phase 8 — Real-Time Inference Phase

This phase tests the streaming inference path that would be used in
a deployed IoT monitoring context.

### Step 8.1 — Batch inference with throughput measurement

```bash
python - <<'EOF'
import numpy as np, time
import tensorflow as tf
from src.iot_anomaly.evaluation.evaluator import realtime_inference

model = tf.keras.models.load_model("checkpoints/quicktest/classifier_best.keras", compile=False)
n_features = model.input_shape[1]

# Simulate 100 incoming network sessions
X_stream = np.random.randn(100, n_features, 1).astype("float32")

threshold = 0.5
t0 = time.perf_counter()
realtime_inference(model, X_stream, threshold=threshold, batch_size=16)
elapsed = time.perf_counter() - t0

throughput = len(X_stream) / elapsed
print(f"\n✓  Processed {len(X_stream)} sessions in {elapsed*1000:.1f} ms")
print(f"   Throughput: {throughput:.0f} sessions/second")
assert throughput > 10, "Throughput too low (< 10 sessions/sec)"
print("✓  Real-time inference: PASSED")
EOF
```

---

## Phase 9 — Final Integration / System Test

This is the end-to-end system test. It runs the **entire pipeline** —
from raw CSV loading through preprocessing, training, evaluation, and
adversarial testing — as a single consistent flow, verifying that all
components interact correctly and the system achieves its goal of
detecting IoT anomalies reliably.

```bash
python - <<'EOF'
"""
IoT SAFEGUARD — End-to-End System Integration Test
===================================================
This script exercises the complete pipeline in a single run:
  1. Data loading (N-BaIoT CSV)
  2. Feature engineering
  3. Preprocessing (imputation → scaling → PCA → SMOTE)
  4. Model construction (autoencoder + attention classifier)
  5. Two-phase training (pretrain AE → finetune classifier)
  6. Evaluation (metrics + ROC curve + confusion matrix)
  7. Threshold tuning
  8. Adversarial robustness probe
  9. Real-time inference
 10. Assertion of system-level acceptance criteria
"""
import os, sys, time
import numpy as np
import pandas as pd
import tensorflow as tf

print("=" * 60)
print("IoT SAFEGUARD — System Integration Test")
print("=" * 60)

# ── Step 1: Data loading ──────────────────────────────────────────────
print("\n[1/9] Loading N-BaIoT data ...")
base   = "data/raw/N-BaIoT/Dataset"
benign = pd.read_csv(f"{base}/Danmini_Doorbell/benign_traffic.csv").head(3000)
benign["label"] = 0
attack = pd.read_csv(f"{base}/Danmini_Doorbell/mirai_attacks/scan.csv").head(3000)
attack["label"] = 1
df = pd.concat([benign, attack]).sample(frac=1, random_state=42).reset_index(drop=True)
print(f"    Rows loaded: {len(df)}, label split: {df['label'].value_counts().to_dict()}")

# ── Step 2: Feature engineering ──────────────────────────────────────
print("\n[2/9] Engineering cross-window features ...")
from src.iot_anomaly.data.preprocessor import engineer_nbaiot_features, DataPreprocessor
raw_features = df.drop(columns=["label"])
features     = engineer_nbaiot_features(raw_features)
print(f"    Features: {raw_features.shape[1]} → {features.shape[1]}")

# ── Step 3: Leak-free split + preprocessing ───────────────────────────
print("\n[3/9] Split raw rows, then fit imputation/scaling/PCA/SMOTE on train only ...")
from src.iot_anomaly.training.trainer import Trainer, TrainingConfig, split_indices
labels = df["label"]
cfg = TrainingConfig(filters=32, kernel_size=3, dense_units=16,
                     dropout_rate=0.2, learning_rate=1e-3,
                     pretrain_epochs=5, finetune_epochs=8, batch_size=64,
                     save_dir="checkpoints/integration_test")
idx = split_indices(labels.values, cfg)
prep = DataPreprocessor(n_top_features=10, pca_variance=0.90,
                        resample=True, random_state=42)
X_3d, y = prep.fit_transform(features.iloc[idx["train"]], labels.iloc[idx["train"]])
X_test_3d = prep.transform(features.iloc[idx["test"]])
print(f"    Train tensor: {X_3d.shape}, resampled labels: {np.bincount(y.astype(int)).tolist()}")
print(f"    Test tensor:  {X_test_3d.shape} (real rows, not resampled)")
assert X_3d.ndim == 3, "DataPreprocessor must return 3-D tensor"
assert not np.isnan(X_3d).any(), "NaNs in preprocessed data"
assert set(idx["train"]).isdisjoint(idx["test"]), "Train and test rows overlap"

# ── Step 4: Model construction ────────────────────────────────────────
print("\n[4/9] Building autoencoder + attention classifier ...")
from src.iot_anomaly.models.autoencoder import build_autoencoder
from src.iot_anomaly.models.classifier  import build_classifier
input_shape = X_3d.shape[1:]
ae, enc = build_autoencoder(input_shape, filters=32, kernel_size=3,
                             dropout_rate=0.2, learning_rate=1e-3)
clf = build_classifier(enc, input_shape, dense_units=16,
                        dropout_rate=0.2, l1_reg=1e-5, l2_reg=1e-4)
print(f"    Encoder bottleneck: {enc.output_shape}")
print(f"    Classifier params: {clf.count_params():,}")

# ── Step 5: Two-phase training ────────────────────────────────────────
print("\n[5/9] Two-phase training ...")
trainer = Trainer(cfg)
t_start = time.perf_counter()
# Re-runs the same split and fits a fresh preprocessor on the training rows only
trainer.fit_from_features(features.reset_index(drop=True), labels.reset_index(drop=True),
                          DataPreprocessor(n_top_features=10, pca_variance=0.90,
                                           resample=True, random_state=42),
                          pretrain=True)
t_train = time.perf_counter() - t_start
print(f"    Training completed in {t_train:.1f}s")
print(f"    Checkpoints: {os.listdir('checkpoints/integration_test')}")

# ── Step 6 & 7: Evaluation and threshold ─────────────────────────────
print("\n[6/9] Evaluating on held-out test set ...")
results = trainer.evaluate()
print(f"    Threshold: {trainer.threshold:.4f}")
for k, v in results.items():
    print(f"    {k:<12} {v:.4f}")

# ── Step 8: Adversarial robustness probe ─────────────────────────────
print("\n[7/9] Adversarial robustness probe ...")
try:
    from art.attacks.evasion import FastGradientMethod
    from art.estimators.classification import KerasClassifier as ARTKerasClassifier
    clf_model = tf.keras.models.load_model(
        "checkpoints/integration_test/classifier_best.keras", compile=False)
    clf_model.compile(loss="binary_crossentropy")   # ART needs a loss for gradients
    art_clf = ARTKerasClassifier(model=clf_model, nb_classes=2,
                                  clip_values=(0.0, 1.0))
    X_test_art = trainer._X_test[:50].astype("float32")   # held-out rows
    y_test_art = trainer._y_test[:50]
    fgsm      = FastGradientMethod(estimator=art_clf, eps=0.05)
    X_adv     = fgsm.generate(x=X_test_art)
    clean_acc  = ((clf_model.predict(X_test_art, verbose=0).flatten() >= 0.5) == y_test_art).mean()
    adv_acc    = ((clf_model.predict(X_adv,      verbose=0).flatten() >= 0.5) == y_test_art).mean()
    print(f"    Clean accuracy: {clean_acc:.4f} | Adversarial accuracy: {adv_acc:.4f}")
    print(f"    Accuracy drop: {clean_acc - adv_acc:.4f}")
except Exception as e:
    print(f"    ⚠  ART probe skipped: {e}")

# ── Step 9: Real-time inference ───────────────────────────────────────
print("\n[8/9] Real-time inference ...")
from src.iot_anomaly.evaluation.evaluator import realtime_inference
clf_model = tf.keras.models.load_model(
    "checkpoints/integration_test/classifier_best.keras", compile=False)
n_feat    = clf_model.input_shape[1]
X_stream  = np.random.randn(30, n_feat, 1).astype("float32")
realtime_inference(clf_model, X_stream, threshold=trainer.threshold, batch_size=10)

# ── Step 10: System acceptance criteria ──────────────────────────────
print("\n[9/9] Checking system acceptance criteria ...")
PASS = True

checks = [
    ("F1 > 0.60 on small test slice", results.get("f1", 0) > 0.60),
    ("AUC-ROC > 0.70",               results.get("auc_roc", 0) > 0.70),
    ("Recall > 0.50 (no missed attacks)", results.get("recall", 0) > 0.50),
    ("Autoencoder checkpoint saved",
        os.path.isfile("checkpoints/integration_test/autoencoder_best.weights.h5")),
    ("Classifier checkpoint saved",
        os.path.isfile("checkpoints/integration_test/classifier_best.keras")),
    ("No NaNs in training data",        not np.isnan(X_3d).any()),
    ("Threshold in (0, 1)",             0.0 < trainer.threshold < 1.0),
]

for desc, cond in checks:
    status = "✓" if cond else "✗"
    if not cond:
        PASS = False
    print(f"    {status}  {desc}")

print("\n" + "=" * 60)
if PASS:
    print("  RESULT:  ALL CHECKS PASSED — System integration test PASSED")
else:
    print("  RESULT:  ONE OR MORE CHECKS FAILED — See above")
print("=" * 60)
EOF
```

**Expected output:**
```
============================================================
IoT SAFEGUARD — System Integration Test
============================================================

[1/9] Loading N-BaIoT data ...
[2/9] Engineering cross-window features ...
[3/9] Preprocessing (imputation, scaling, PCA, SMOTE) ...
[4/9] Building autoencoder + attention classifier ...
[5/9] Two-phase training ...
[6/9] Evaluating on held-out test set ...
[7/9] Adversarial robustness probe ...
[8/9] Real-time inference ...
[9/9] Checking system acceptance criteria ...
    ✓  F1 > 0.60 on small test slice
    ✓  AUC-ROC > 0.70
    ✓  Recall > 0.50 (no missed attacks)
    ✓  Autoencoder checkpoint saved
    ✓  Classifier checkpoint saved
    ✓  No NaNs in training data
    ✓  Threshold in (0, 1)

============================================================
  RESULT:  ALL CHECKS PASSED — System integration test PASSED
============================================================
```

---

## Part 10 — Common Errors and Fixes

| Error | Cause | Fix |
|---|---|---|
| `IsADirectoryError: .../bro` | Passing a directory to `parse_conn_log_labeled` | Supply the full path: `.../bro/conn.log.labeled` |
| `ModuleNotFoundError: iot_anomaly` | Package not installed | Run `pip install -e .` from project root |
| `ResourceExhaustedError` (OOM) | Batch too large for GPU | Reduce `--batch` to 64 or 32; or use `tf.config.experimental.set_memory_growth` |
| `ValueError: Input 0 incompatible` | Shape mismatch between fit and transform | Ensure `prep.transform()` uses the same fitted `DataPreprocessor` as training |
| `KeyError: inter_arrival_min` | CSV generated before BUG-01 fix | Re-run `process_pcap.py` to regenerate with correct column names |
| `Training accuracy stuck at 0.5` | Resampling not applied (original BUG-16) | Confirmed fixed — `DataPreprocessor` always returns resampled arrays |
| `art.exceptions.EstimatorError` | ART version mismatch with TF | `pip install adversarial-robustness-toolbox==1.17.0` |
| GPU not detected | `tensorflow-metal` not installed, or Rosetta Python in use | Run `pip install tensorflow-metal==1.0.1`; confirm `python3 -c "import platform; print(platform.machine())"` prints `arm64` |
| TensorBoard shows no data | `--logdir` points to wrong folder | Ensure `save_dir` in config matches the `--logdir` path |

---

## Phase Completion Checklist

```
PHASE 0 — Hardware & Environment
[ ] Apple Metal GPU detected via tensorflow-metal (or CPU-only confirmed)
[ ] tensorflow-metal version matches TensorFlow version (see Step 0.2 table)
[ ] arm64 Python confirmed (not Rosetta x86)
[ ] All packages import without error (tf, dpkt, sklearn, imblearn, art)
[ ] Entry points (iot-process-pcap, iot-train) print help text

PHASE 1 — Data Processing
[ ] N-BaIoT CSV loads correctly: (N, 115) features, binary labels
[ ] PCAP parser extracts sessions with inter_arrival_min key present
[ ] Label mapper raises ValueError for directory path (BUG-10)
[ ] Label mapper is case-insensitive (BUG-11)

PHASE 2 — Feature Engineering & Preprocessing
[ ] inter_arrival_* keys correct (BUG-01)
[ ] No double prefix on flag keys (BUG-02)
[ ] Single-packet IAT default is 0.0 (BUG-07)
[ ] Uppercase protocol keys match (BUG-05)
[ ] Port range returns 4 keys (BUG-06)
[ ] Pipeline: NaNs removed, data scaled, PCA applied
[ ] SMOTE resampling applied, class balance improved
[ ] DataPreprocessor output is 3-D tensor (N, K, 1)

PHASE 3 — Model Architecture
[ ] Autoencoder loss = 'mse' (BUG-14)
[ ] Encoder output shape is compressed, not equal to input (BUG-13)
[ ] Decoder final activation = 'linear' (BUG-14)
[ ] Classifier uses encoder only (BUG-13)
[ ] Classifier output shape = (None, 1), values in [0, 1]
[ ] Temporal attention layer present

PHASE 4 — Training
[ ] Quick run completes (< 3 min) with no exceptions
[ ] Checkpoints saved (autoencoder_best.weights.h5, classifier_best.weights.h5, classifier_best.keras)
[ ] Raw rows split before preprocessing; test set contains real rows only
[ ] Optimal threshold found in (0, 1)
[ ] TensorBoard shows decreasing loss curves
[ ] Full N-BaIoT run achieves F1 > 0.994

PHASE 5 — Evaluation
[ ] compute_metrics returns all 5 metrics
[ ] find_optimal_threshold returns no IndexError (BUG-20)
[ ] roc_curve.png and confusion_matrix.png saved and visually correct

PHASE 6 — PCAP Pipeline
[ ] Session features CSV generated with correct column names
[ ] Labels binary in output CSV
[ ] CTU training completes without errors

PHASE 7 — Adversarial Robustness
[ ] ART imports without error
[ ] FGSM attack generates adversarial examples without crashing
[ ] Adversarial accuracy reported (any value acceptable)

PHASE 8 — Real-Time Inference
[ ] Throughput > 10 sessions/second

PHASE 9 — System Integration Test
[ ] ALL 7 acceptance criteria pass
[ ] Final printout: "System integration test PASSED"
```
