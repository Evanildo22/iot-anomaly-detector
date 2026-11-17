# IoT Anomaly Detector — Bug & Architecture Analysis

> Comprehensive audit of `data_processing.py` and `model.py`
> Severity: 🔴 Critical · 🟠 Major · 🟡 Minor · 🔵 Architectural

---

## Table of Contents

1. [Data Processing Script (`data_processing.py`)](#1-data-processing-script)
2. [Model Script (`model.py`)](#2-model-script)
3. [Cross-Cutting Architectural Problems](#3-cross-cutting-architectural-problems)
4. [Summary Table](#4-summary-table)

---

## 1. Data Processing Script

### BUG-01 🔴 `calculate_inter_arrival_time_statistics` returns wrong keys

**File:** `data_processing.py`
**Function:** `calculate_inter_arrival_time_statistics`

The fallback branch (fewer than 2 timestamps) returns keys prefixed with `inter_arrival_*`.
But when there *are* enough timestamps the function delegates to `calculate_packet_statistics`,
which returns keys prefixed with `packet_size_*`.

```python
# BUG — returns {'packet_size_min': …, 'packet_size_max': …, …}
def calculate_inter_arrival_time_statistics(timestamps):
    if len(timestamps) < 2:
        return {
            'inter_arrival_min': 1,   # ← correct key, wrong default (should be 0)
            ...
        }
    inter_arrival_times = np.diff(np.sort(timestamps))
    return calculate_packet_statistics(inter_arrival_times)  # ← wrong keys!
```

`process_session_features` merges this dict with `**`, so the session row ends up with
`packet_size_min / packet_size_max / …` columns twice (overwriting the real packet-size ones)
and *no* `inter_arrival_*` columns at all.  The `selected_features` list then tries to read
`inter_arrival_min`, etc., and gets 0 from the fill step—**silently corrupting all
inter-arrival statistics**.

**Fix:** rename keys inside the helper or write a dedicated wrapper:

```python
def calculate_inter_arrival_time_statistics(timestamps):
    key_map = {
        'packet_size_min':    'inter_arrival_min',
        'packet_size_max':    'inter_arrival_max',
        'packet_size_median': 'inter_arrival_median',
        'packet_size_q1':     'inter_arrival_q1',
        'packet_size_q3':     'inter_arrival_q3',
        'packet_size_mean':   'inter_arrival_mean',
        'packet_size_std':    'inter_arrival_std',
    }
    if len(timestamps) < 2:
        return {v: 0.0 for v in key_map.values()}   # default 0, not 1
    iat = np.diff(np.sort(timestamps))
    raw = calculate_packet_statistics(iat)
    return {key_map[k]: v for k, v in raw.items()}
```

---

### BUG-02 🔴 Double-prefixed flag combination keys

**File:** `data_processing.py`
**Functions:** `calculate_packet_flag_combinations`, `encode_packet_flag_combinations`

`calculate_packet_flag_combinations` already adds the `flag_comb_` prefix:

```python
return {f'flag_comb_{flags}': count / total_tcp_packets …}
# → {'flag_comb_0': 0.5, 'flag_comb_24': 0.3, …}
```

Then `encode_packet_flag_combinations` adds *another* `flag_comb_` prefix:

```python
flag_comb_df = pd.DataFrame([{f'flag_comb_{k}': v …}])
# k is already 'flag_comb_0' → final column = 'flag_comb_flag_comb_0'
```

The `selected_features` list was then manually updated to match these broken names
(`'flag_comb_flag_comb_0'`, `'flag_comb_flag_comb_24'`, …), baking the bug into the
feature schema. Any model trained on this schema cannot be reused without also carrying the bug.

**Fix:** strip the prefix before re-adding it in the encoder, *or* (simpler) do not add it in the extractor:

```python
# In calculate_packet_flag_combinations — return raw flag integers as keys
return {flags: count / total_tcp_packets for flags, count in flag_combinations.items()}

# In encode_packet_flag_combinations — add the prefix once
flag_comb_df = pd.DataFrame([{f'flag_comb_{k}': v for k, v in d.items()} …])
```

---

### BUG-03 🔴 `payload_size` is computed incorrectly

**File:** `data_processing.py`
**Line:** `payload_size = len(ip.data) - (ip.hl * 4)`

In dpkt, `ip.data` is *already* the IP payload (everything after the IP header).
Subtracting `ip.hl * 4` (header length in bytes) again produces **negative values** for
small packets and for all ICMP echo replies.

```python
# BUG
payload_size = len(ip.data) - (ip.hl * 4)   # double-subtracts header

# FIX — ip.data is already the transport payload
payload_size = len(ip.data)
```

---

### BUG-04 🔴 Session chunking splits live sessions

**File:** `data_processing.py`
**Function:** `extract_sessions_and_features_from_pcap`

```python
if packet_count >= chunk_size:
    session_features_list.extend(process_session_features(sessions))
    sessions.clear()           # ← destroys in-progress sessions
    packet_count = 0
```

When the counter hits `chunk_size`, *all* sessions are flushed and the dict is cleared.
Any long-lived TCP session that straddles two chunks is **split into two separate feature
rows**, both with incomplete statistics.  The result: duplicate / truncated session records
and corrupted flow-level features (packet counts, timing statistics, flag ratios, etc.).

**Fix:** only flush sessions that have been idle (no new packets) for a configurable timeout,
or buffer by time window rather than raw packet count.

```python
# Flush only sessions older than timeout_seconds
cutoff = current_time - timeout_seconds
idle_sessions = {sid: pkts for sid, pkts in sessions.items()
                 if float(pkts[-1]['timestamp']) < cutoff}
session_features_list.extend(process_session_features(idle_sessions))
for sid in idle_sessions:
    del sessions[sid]
```

---

### BUG-05 🔴 Protocol distribution uses wrong case

**File:** `data_processing.py`
**Function:** `process_session_features`

```python
protocol_mapping = {'tcp': 6, 'udp': 17, 'icmp': 1}   # lowercase keys

protocol_counts = {
    protocol: sum(1 for p in session_packets if p['protocol'] == protocol)
    for protocol in protocol_mapping.keys()            # iterates lowercase
}
```

But packets are stored with uppercase protocol strings (`'TCP'`, `'UDP'`, `'ICMP'`) because:

```python
protocol = ip.get_proto(ip.p).__name__.upper()
```

`protocol_counts` is therefore always `{'tcp': 0, 'udp': 0, 'icmp': 0}`, and
`protocol_distribution` is always zeros.  After `encode_protocol`, columns
`protocol_TCP / protocol_UDP / protocol_ICMP` are always 0 — the model never sees
which transport protocol a flow uses.

**Fix:** normalise the case at one point, or fix the mapping:

```python
protocol_mapping = {'TCP': 6, 'UDP': 17, 'ICMP': 1}
```

---

### BUG-06 🟠 Port range stored as a tuple (unpicklable column)

**File:** `data_processing.py`
**Function:** `process_session_features`

```python
'src_port_range': calculate_port_range(session_packets, 'src_port'),
# returns (min_port, max_port) — a Python tuple
```

Pandas stores this as an `object` dtype column containing tuples.  The later numeric
normalisation step (`df_selected[numeric_columns]`) skips it, so the column is never
properly typed.  The `selected_features` list doesn't include it either, so the work is
wasted.

**Fix:** split into two scalar columns:

```python
src_min, src_max = calculate_port_range(session_packets, 'src_port')
'src_port_min': src_min,
'src_port_max': src_max,
```

---

### BUG-07 🟠 `calculate_inter_arrival_time_statistics` defaults to 1 instead of 0

When `len(timestamps) < 2` the function returns all defaults as `1`.  A session with a
single packet should have inter-arrival statistics of `0` (no intervals exist), not `1`.
Returning `1` injects a false inter-arrival time of 1 second, biasing the model.

---

### BUG-08 🟠 `Decimal` timestamps cause severe performance and precision issues

**File:** `data_processing.py`

```python
'timestamp': Decimal(timestamp)
```

`Decimal` objects are 30–100× slower than `float64` for arithmetic.  They also can't be
passed directly to `numpy` functions, causing silent coercions and occasional
`InvalidOperation` exceptions on edge-case values.  The claimed benefit (precision) is
negated because timestamps are later cast back to `float`:

```python
timestamps = np.array([float(p['timestamp']) for p in session_packets])
```

**Fix:** store timestamps as `float64` throughout.

---

### BUG-09 🟠 `aggregate_packet_payload_byte_distribution` reads non-existent key

**File:** `data_processing.py`

```python
payload_bytes = [byte for packet in packets for byte in packet.get('payload', b'')]
```

No packet dict ever has a `'payload'` key (the code extracts `payload_size` but not the
raw bytes).  The result is always the empty-payload path, so `byte_entropy` is always 0
and the feature provides no signal.

---

### BUG-10 🟡 `parse_conn_log_labeled` called with a directory path

**File:** `data_processing.py` (bottom of script)

```python
labeled_file_path = '…/CTU-IoT-Malware-Capture-1-1/bro'
parse_conn_log_labeled(labeled_file_path)
```

`labeled_file_path` is a *directory* (`bro/`), not a file.  `open(file_path, 'r')`
immediately raises `IsADirectoryError`.  The intended file is `bro/conn.log.labeled`.

---

### BUG-11 🟡 Label encoding misses actual label values

**File:** `data_processing.py`

```python
df_selected['Label'] = df_selected['Label'].apply(
    lambda x: 1 if 'Malicious' in x else 0
)
```

The CTU dataset labels use lowercase (`'malicious'`, `'benign'`), not title-case
`'Malicious'`.  Every row will be encoded as `0` (benign), making supervised training
impossible.  Use `str.lower()` before checking:

```python
lambda x: 1 if 'malicious' in str(x).lower() else 0
```

---

### BUG-12 🟡 `SettingWithCopyWarning` throughout label-mapping

`map_labels_to_dataframe` assigns columns on a slice of a DataFrame (`df['Label'] = …`).
This triggers pandas `SettingWithCopyWarning` and can silently drop the assignment.  Use
`.loc` or work on an explicit copy.

---

## 2. Model Script

### BUG-13 🔴 Attention applied to *decoded* output, not the *encoded* bottleneck

**File:** `model.py`
**Function:** `add_attention_layer`

```python
def add_attention_layer(autoencoder_model, input_shape, …):
    inputs = Input(shape=input_shape)
    encoded = autoencoder_model(inputs)   # ← calls full autoencoder → shape == input
    attention_mul = attention_block(encoded)
    …
```

`autoencoder_model` is the *full* autoencoder (encoder + decoder), so `encoded` here is
actually the **reconstruction** of shape `(n_features, 1)` — identical to the input shape.
Attention is therefore applied to a noisy copy of the input, not to the compressed
bottleneck representation.  The entire purpose of the CAE (learning a compact, anomaly-
sensitive embedding) is defeated.

**Fix:** expose the encoder as a separate `Model`, and build the classifier on top of it:

```python
# In build_cae_model, return both models
encoder = Model(inputs, encoded)          # bottleneck output
autoencoder = Model(inputs, decoded)      # full reconstruction

# In add_attention_layer, accept encoder only
def build_classifier(encoder, input_shape, dense_units, …):
    inputs = Input(shape=input_shape)
    bottleneck = encoder(inputs)           # compressed embedding
    attended = attention_block(bottleneck)
    flat = Flatten()(attended)
    out = Dense(dense_units, activation='relu', …)(flat)
    out = Dense(1, activation='sigmoid')(out)
    return Model(inputs, out)
```

---

### BUG-14 🔴 Autoencoder uses binary cross-entropy on continuous inputs

**File:** `model.py`

```python
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
```

The input data is StandardScaler-normalised continuous features (range ≈ −3 to +3), and
the decoder uses `sigmoid` activation (range 0–1).  `binary_crossentropy` treats the
output as a probability distribution over binary values, which is **mathematically
undefined** for inputs outside [0, 1] and produces gradients that actively push
reconstructions toward 0 or 1.

**Fix:**

```python
# Decoder final activation: linear or tanh (not sigmoid)
decoded = Conv1D(1, kernel_size, activation='linear', padding='same')(x)
autoencoder.compile(optimizer='adam', loss='mse')
```

---

### BUG-15 🔴 Training data leakage via inconsistent splits

**File:** `model.py`

The script performs three independent dataset splits:

1. 4-way split: `X_train / X_validation / X_test / X_real_time` (lines 90–95)
2. SMOTE resampling of `X_train` → `X_resampled` (line 97)
3. **New** train/val split from the **entire** `features_preprocessed` array
   (`X_train_full`, line 139): this includes `X_test` and `X_real_time`.

The final model is trained on `X_train_full` which contains held-out test data, and then
evaluated on `X_test`.  This is a **data leakage** bug.

> **Update (2026-11-17):** the April 2026 refactor fixed this only partially. Model
> training no longer sees the test set, but feature selection, scaling, PCA and SMOTE
> were still fitted on the full dataset before the split. See [BUG-25](#bug-25).

---

### BUG-16 🔴 SMOTE resampling result is never used

**File:** `model.py`

```python
X_resampled, y_resampled = resampling_pipeline.fit_resample(X_train, y_train)
```

The resampled arrays are never used.  The model is trained on:

```python
X_train_subset = X_train[:10000]   # original imbalanced data
```

Class imbalance is therefore not corrected despite the explicit resampling step.

---

### BUG-17 🔴 Import conflict: `imblearn.Pipeline` silently overwrites `sklearn.Pipeline`

**File:** `model.py`

```python
from sklearn.pipeline import Pipeline      # line ~15
…
from imblearn.pipeline import Pipeline     # line ~30  ← overwrites!
```

`preprocessing_pipeline = ColumnTransformer([('num', Pipeline([…]), selected_features)])` now
uses `imblearn.Pipeline` (which requires a final estimator step and cannot be used as a
pure transformer), **breaking the preprocessing pipeline silently** unless `imblearn` and
`sklearn` happen to accept the same interface for the particular call path.

**Fix:** import with an alias:

```python
from imblearn.pipeline import Pipeline as ImbPipeline
```

---

### BUG-18 🟠 `KerasClassifier` import is immediately overridden

**File:** `model.py`

```python
from art.estimators.classification import KerasClassifier   # ART class

class KerasClassifier(BaseEstimator):                       # overwrites ART class
    …
```

The ART `KerasClassifier` (needed for adversarial robustness testing via
`FastGradientMethod`) is completely shadowed.  `FastGradientMethod` is also imported but
never used.

---

### BUG-19 🟠 Split proportions don't match documented intent

**File:** `model.py`

The comment says "65% training, 15% validation, 15% test, 5% real-time". Actual splits:

| Set | Comment | Actual |
|---|---|---|
| Train | 65% | 65.0% |
| Validation | 15% | 17.5% |
| Test | 15% | 13.1% |
| Real-time | 5% | 4.4% |

The second `train_test_split` has `test_size=0.5`, splitting the remaining 35% evenly
(17.5% + 17.5%) instead of 15% + 20%.

---

### BUG-20 🟠 `find_optimal_threshold` can return index beyond `thresholds` array

**File:** `model.py`

`precision_recall_curve` returns arrays of length `n + 1` for `precision` and `recall`
but only `n` for `thresholds`.  If `np.nanargmax(f1_scores)` returns index `n`
(the last element), `thresholds[n]` is an **IndexError**.

**Fix:**

```python
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
precision, recall = precision[:-1], recall[:-1]   # align lengths
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
return thresholds[np.argmax(f1_scores)], f1_scores.max()
```

---

### BUG-21 🟡 Model saved/loaded under different filenames

**File:** `model.py`

```python
model_checkpoint = ModelCheckpoint('best_model.keras', …)  # saved by callback
model_with_attention.save('trained_model.keras')            # saved manually
loaded_model = tf.keras.models.load_model('trained_model.keras')  # loaded
```

Three different file names are in play.  The `ModelCheckpoint` saves
`best_model.keras` (the best validation-loss checkpoint), but `save()` saves
`trained_model.keras` (the final epoch).  Evaluation loads `trained_model.keras`, which
may not be the best checkpoint.

---

### BUG-22 🟡 Duplicate model definition / dead commented-out code

A second, slightly different training block is commented out with `"""..."""` (triple-
quoted string, not a real comment).  Python parses this as a string literal on every
import, adding startup overhead.  It also creates confusion about which configuration
was actually used.

---

### BUG-23 🟡 Unused imports

The following symbols are imported but never used:

| Import | File |
|--------|------|
| `shap` | `model.py` |
| `FastGradientMethod` | `model.py` |
| `BayesSearchCV`, `Real`, `Categorical`, `Integer` | `model.py` |
| `StratifiedKFold` | `model.py` |
| `clone_model` | `model.py` |
| `LSTM`, `Bidirectional`, `RepeatVector` | `model.py` |
| `custom_accuracy_score` | `model.py` |

---

## 3. Cross-Cutting Architectural Problems

### ARCH-01 🔵 Two independent datasets with no shared pipeline

The project uses **CTU-IoT-Malware** (raw PCAP → session features) and **N-BaIoT**
(pre-extracted CSV statistics) with completely separate preprocessing code.  There is no
unified `Dataset` abstraction, making it impossible to train on a combined corpus or swap
datasets.

### ARCH-02 🔵 Feature schema is hard-coded and fragile

`selected_features` lists 120+ column names literally.  TTL distribution columns
(`ttl_52`, `ttl_97`, etc.) are specific to one capture scenario.  Any new pcap file will
produce different TTL distributions, silently padding missing columns with zeros.

### ARCH-03 🔵 Hyperparameters are hard-coded magic numbers

Best hyperparameters (`best_filters = 112`, `best_learning_rate = 0.000301…`) are pasted
inline with no provenance.  There is no configuration file, no CLI argument, no way to
reproduce the search.

### ARCH-04 🔵 No reproducibility controls

No global random seed is set for `numpy`, `python`, or `tensorflow`.  Results are not
reproducible across runs.

### ARCH-05 🔵 No unit tests

Both scripts run as top-level code with hard-coded file paths, making them untestable
and non-portable.  Any function can silently break without detection.

---

## 3b. Findings from the November 2026 Validation

These defects were in the refactored package itself and were found when the
validation guide was run end to end for the first time
(`test-outputs/VALIDATION_2026-11-16.md`).

### BUG-24 🔴 Autoencoder output length differs from input length

**File:** `models/autoencoder.py`

The encoder has two `MaxPooling1D(padding="same")` stages and the decoder two
`UpSampling1D(2)` stages, so the reconstruction has `4*ceil(ceil(L/2)/2)` time-steps.
That exceeds the input length `L` whenever `L % 4 != 0`. PCA chooses `L`
(11 on the N-BaIoT subset), so training crashed on the MSE loss
(`Dimensions must be equal, but are 12 and 11`).

**Fix:** a `Cropping1D` layer, sized from the actual tensor shape, trims the excess.

---

<a id="bug-25"></a>
### BUG-25 🔴 Preprocessing and SMOTE fitted before the train/test split

**File:** `cli/train.py`, `training/trainer.py`

The training entry point called `DataPreprocessor.fit_transform` on the **whole**
dataset and only then let `Trainer.fit` split the result. As a consequence:

1. Correlation pruning and Random-Forest feature selection saw the test labels.
2. The imputer, scaler and PCA were fitted on test rows.
3. SMOTE and random under-sampling ran on all rows, so the "held-out" test set
   contained synthetic rows interpolated from training neighbours, and its class
   balance was artificial (on the single-device subset, 12,375 of the 19,807
   benign test rows were synthetic).

The README nonetheless stated that the test split was "held out before any
preprocessing or resampling is applied".

**Fix:** `Trainer.fit_from_features` splits the raw rows first (stratified
65/15/15/5), fits selection, imputation, scaling, PCA and resampling on the training
rows only, and applies the fitted pipeline with `transform` to the validation, test
and real-time rows. Those keep their natural class balance and contain no synthetic
rows.

---

### BUG-26 🟠 Autoencoder pretrained on attack and synthetic traffic

**File:** `training/trainer.py`

Phase 1 is meant to learn what *normal* traffic looks like, but the autoencoder was
pretrained on every training row, including malicious and SMOTE-synthesised rows.

**Fix:** pretraining uses real benign training rows only, and its validation loss
is measured on benign validation rows.

---

## 4. Summary Table

| ID | Severity | Location | Description |
|----|----------|----------|-------------|
| BUG-01 | 🔴 Critical | `data/features.py` | Inter-arrival keys wrong — all IAT features are zeros |
| BUG-02 | 🔴 Critical | `data/features.py` | Flag combination keys double-prefixed |
| BUG-03 | 🔴 Critical | `data/parser.py` | `payload_size` subtracts IP header twice |
| BUG-04 | 🔴 Critical | `data/parser.py` | Session chunking splits live sessions |
| BUG-05 | 🔴 Critical | `data/features.py` | Protocol distribution always zero (case mismatch) |
| BUG-06 | 🟠 Major | `data/features.py` | Port range returned as tuple, not split scalars |
| BUG-07 | 🟠 Major | `data/features.py` | IAT defaults are 1 not 0 |
| BUG-08 | 🟠 Major | `data/parser.py` | `Decimal` timestamps: slow + unnecessary |
| BUG-09 | 🟠 Major | `data/features.py` | Payload byte distribution always empty |
| BUG-10 | 🟡 Minor | `scripts/process_data.py` | Labeled file path points to directory |
| BUG-11 | 🟡 Minor | `data/label_mapper.py` | Label case mismatch — all labels encode as 0 |
| BUG-12 | 🟡 Minor | `data/label_mapper.py` | `SettingWithCopyWarning` on slice assignment |
| BUG-13 | 🔴 Critical | `models/classifier.py` | Attention on decoder output, not encoder bottleneck |
| BUG-14 | 🔴 Critical | `models/autoencoder.py` | `binary_crossentropy` on continuous normalised input |
| BUG-15 | 🔴 Critical | `training/trainer.py` | Test data leaks into training set |
| BUG-16 | 🔴 Critical | `training/trainer.py` | SMOTE result never used — class imbalance uncorrected |
| BUG-17 | 🔴 Critical | `training/trainer.py` | `imblearn.Pipeline` overwrites `sklearn.Pipeline` |
| BUG-18 | 🟠 Major | `models/classifier.py` | ART `KerasClassifier` import shadowed by local class |
| BUG-19 | 🟠 Major | `training/trainer.py` | Split proportions don't match stated intent |
| BUG-20 | 🟠 Major | `evaluation/evaluator.py` | `thresholds[n]` IndexError in optimal threshold search |
| BUG-21 | 🟡 Minor | `training/trainer.py` | Three different model file names, loads wrong checkpoint |
| BUG-22 | 🟡 Minor | `training/trainer.py` | Dead code in triple-quoted string |
| BUG-23 | 🟡 Minor | all | 10+ unused imports |
| BUG-24 | 🔴 Critical | `models/autoencoder.py` | Reconstruction longer than input when PCA length % 4 ≠ 0 |
| BUG-25 | 🔴 Critical | `cli/train.py`, `training/trainer.py` | Selection, PCA and SMOTE fitted before the split — test set leaked |
| BUG-26 | 🟠 Major | `training/trainer.py` | Autoencoder pretrained on attack and synthetic rows, not benign only |
| ARCH-01 | 🔵 Arch | all | No unified dataset abstraction |
| ARCH-02 | 🔵 Arch | all | Feature schema hard-coded and fragile |
| ARCH-03 | 🔵 Arch | all | Hyperparameters as magic numbers |
| ARCH-04 | 🔵 Arch | all | No reproducibility controls |
| ARCH-05 | 🔵 Arch | all | No unit tests |
