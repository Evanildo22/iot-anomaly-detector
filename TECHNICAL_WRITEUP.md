# IoT SAFEGUARD — Technical Write-Up
## Architecture Review, Bug Analysis, and System Redesign

**Project:** IoT SAFEGUARD — Anomaly Detection for IoT Networks
**Original submission:** 26 May 2024, University of West London
**Refactored system:** IoT Anomaly Detector v1.0.0

---

## 1. Introduction

The IoT SAFEGUARD project established a sound theoretical basis for
network-level anomaly detection, selecting a Hybrid Convolutional
Autoencoder with Attention Mechanism as its core architecture, the
N-BaIoT dataset as its primary evaluation corpus, and Bayesian
hyperparameter optimisation to identify the best model configuration.
These design choices were well-motivated by the literature and
resulted in near-perfect cross-validation scores during testing.

However, a subsequent systematic audit of the implementation revealed
twenty-five distinct bugs and architectural deficiencies — eleven of
them critical, in the sense that they silently corrupted data or model
behaviour without producing errors or warnings. This write-up reports
those findings in discursive form, explains the reasoning behind each
correction, and documents the redesigned system that resolves them.
It is intended both as an honest account of the original
implementation's shortcomings and as a reference for future
development of the project.

The bugs are grouped into three areas — data processing, model
architecture, and cross-cutting design — corresponding to the two
original source files (`data_processing.py` and `model.py`) and the
system's overall structure.

---

## 2. Data Processing: Silent Feature Corruption

The data processing script was responsible for parsing raw PCAP
files, grouping packets into bidirectional network sessions, computing
over a hundred per-session statistical features, and joining
ground-truth labels from Zeek `conn.log.labeled` files. Despite
running without runtime exceptions, the script produced systematically
corrupted feature vectors due to several key implementation errors.

### 2.1 Inter-Arrival Time Statistics Returned the Wrong Keys

One of the most consequential bugs concerned the inter-arrival time
(IAT) feature family, which captures the temporal regularity of a
network session. These features are particularly important for
detecting scan-based and denial-of-service attacks, which produce
characteristic timing patterns. The function
`calculate_inter_arrival_time_statistics` was intended to return seven
statistics — minimum, maximum, median, first quartile, third quartile,
mean, and standard deviation — under the keys `inter_arrival_min`,
`inter_arrival_max`, and so on.

In practice, however, the function delegated to a generic helper,
`calculate_packet_statistics`, whose output used the prefix
`packet_size_*` rather than `inter_arrival_*`. Because Python
dictionary unpacking (`**`) merges keys silently, the session feature
dictionaries ended up with two copies of the packet-size columns —
one from the legitimate call to `calculate_packet_statistics(sizes)`
and one from the IAT call — while the `inter_arrival_*` columns
were never populated. The downstream code that filtered for
`selected_features` then filled these missing columns with zeros, so
every session in the dataset appeared to have identical, zeroed
inter-arrival statistics. The model was consequently denied one of
its most informative feature families for the full duration of
training.

The fix was to rename the keys at the point of return, mapping
`packet_size_min` → `inter_arrival_min`, and so on. Additionally,
the default values returned when fewer than two timestamps are
available were corrected from `1` to `0.0`: a single-packet session
has no inter-arrival intervals, so zero is the semantically correct
default, whereas one implies a one-second inter-arrival time that
would skew statistics.

### 2.2 TCP Flag Combination Columns Were Double-Prefixed

The flag-combination feature was designed to capture the distribution
of TCP control-flag patterns within a session, which is a
well-established indicator of scanning and exploitation activity.
The extraction function `calculate_packet_flag_combinations` returned
a dictionary whose keys already carried the prefix `flag_comb_` (for
example, `flag_comb_0`, `flag_comb_24`). The encoding function
`encode_packet_flag_combinations` then applied the same prefix again
during DataFrame assembly, producing columns such as
`flag_comb_flag_comb_0`. Because the entire pipeline produced these
doubled names consistently, the error was not caught by the code — it
simply meant that the feature schema was effectively arbitrary, tied
to a particular bug rather than to any meaningful naming convention.
The `selected_features` list had been manually adjusted to match the
doubled names, effectively baking the bug into the model's interface
contract. Any deployment or extension of the code would have had to
silently preserve this error.

The fix removed the prefix from the extraction function, so that keys
are raw integer flag values, and retained the prefix only in the
encoding step. This produces clean column names such as `flag_comb_2`
and `flag_comb_18`, which directly correspond to the numeric TCP flag
field values from the packet headers.

### 2.3 Payload Size Was Computed Incorrectly

In the dpkt library, `ip.data` refers to the IP packet's payload —
everything after the IP header. The original code computed
`payload_size = len(ip.data) - (ip.hl * 4)`, subtracting the IP
header length a second time. Since `ip.hl * 4` is the number of bytes
already consumed by the IP header, and `ip.data` excludes those bytes,
the subtraction was redundant and harmful: for any packet whose
transport layer header was smaller than the IP header (which is common
for ICMP and for small TCP packets), this produced a negative payload
size. The fix was simply to use `len(ip.data)` directly.

### 2.4 Session Chunking Split Live Flows

The parser was designed to process PCAP files in chunks of 10,000
packets to limit memory consumption, flushing completed session
features after each chunk. The flushing logic, however, called
`sessions.clear()` unconditionally at the chunk boundary, which meant
that any TCP session spanning the boundary was split into two separate
rows. Long-lived flows — which are precisely the kind of traffic most
likely to be associated with data exfiltration or command-and-control
communication — were therefore systematically truncated, and their
features computed from incomplete packet sequences. The duplicate
rows also inflated the dataset size and could mislead the model.

The corrected implementation uses idle-timeout flushing: a session is
only emitted when no new packets have been observed from it for a
configurable number of seconds (defaulting to 120 seconds, consistent
with common network flow export standards). This ensures that sessions
are always processed as complete units.

### 2.5 Protocol Distribution Always Returned Zeros

The protocol distribution feature, intended to record the fraction of
a session's packets that used TCP, UDP, or ICMP, was computed by
iterating over the keys of `protocol_mapping`, which used lowercase
strings (`'tcp'`, `'udp'`, `'icmp'`). The actual protocol strings
stored in packet dictionaries, however, came from
`ip.get_proto(ip.p).__name__.upper()`, which produced uppercase
strings (`'TCP'`, `'UDP'`, `'ICMP'`). The case mismatch meant that
every comparison evaluated to `False`, and all three protocol counts
were permanently zero. After one-hot encoding, the columns
`protocol_TCP`, `protocol_UDP`, and `protocol_ICMP` — which the model
depended upon to distinguish transport-layer behaviour — provided no
discriminative information whatsoever. The fix aligned the mapping
dictionary to use uppercase keys.

### 2.6 Label Encoding Used the Wrong Case

The ground-truth labels from CTU-IoT-Malware `conn.log.labeled` files
are lowercase strings such as `'malicious'` and `'benign'`. The binary
encoding logic, however, tested for the string `'Malicious'` with a
capital M. Because Python string comparison is case-sensitive, this
condition was never satisfied, and every session — including genuine
attack sessions — was encoded as benign (label `0`). The resulting
training dataset was therefore fully one-class, making supervised
learning impossible. The fix applied `str.lower()` before the
membership test.

### 2.7 Additional Data Processing Issues

Beyond the six critical bugs above, the original script exhibited
several further problems. The `parse_conn_log_labeled` function was
called with a directory path (`bro/`) rather than the specific file
path (`bro/conn.log.labeled`), which would immediately raise an
`IsADirectoryError` in production. Timestamps were stored as Python
`Decimal` objects — a high-precision type designed for financial
arithmetic — rather than as `float64`, causing a 30-to-100-fold
slowdown in all numerical operations without any benefit, since the
values were cast back to `float` immediately before use. The payload
byte distribution feature iterated over a `'payload'` dictionary key
that was never populated, meaning the feature always returned zeros.
Port range values were stored as Python tuples rather than as separate
scalar columns, preventing numeric type coercions from operating on
them. DataFrame column assignments on filtered slices used direct
indexing rather than `.loc`, triggering the pandas `SettingWithCopyWarning`
and risking silent no-ops in certain pandas versions.

---

## 3. Model Architecture: Structural and Mathematical Errors

The model script contained five critical bugs that fundamentally
undermined the intended architecture, in addition to several
lesser issues.

### 3.1 Attention Was Applied to the Reconstructed Output, Not the Bottleneck

This was the most consequential architectural error in the entire
project. The stated design intent, as described in Section 6.2 of the
original report, was for the attention mechanism to enhance the
"encoded representations" — that is, to focus the classification head
on the most anomaly-relevant features in the compressed bottleneck.
This is a well-established technique in deep learning for anomaly
detection, grounded in the principle that the encoder's compressed
representation should learn a compact description of normal behaviour,
and that attention can highlight which dimensions of that description
deviate most under attack.

The implementation, however, passed the full autoencoder model (not
just the encoder) into `add_attention_layer`. Because the full
autoencoder applies both the encoder and the decoder, its output is
the reconstruction — a tensor of the same shape as the input.
Attention was therefore computed over a noisy copy of the original
input rather than over the bottleneck embedding. The downstream
Dense layers then operated on this reconstruction rather than on a
compressed feature representation, negating the purpose of the
autoencoder entirely. The system was, in effect, a slightly unusual
direct classifier with an intermediate reconstruction detour.

The fix separated the encoder into its own independent `Model` object,
returned alongside the full autoencoder from `build_autoencoder`. The
classifier was then built on top of the encoder only. This is the
architecturally correct arrangement: the encoder maps the input to a
compressed bottleneck, the attention mechanism weights the most
informative temporal positions within that bottleneck, and the
classification head maps the attended representation to a binary
label.

### 3.2 The Autoencoder Used the Wrong Loss Function

The autoencoder's final decoder layer used a sigmoid activation, and
the model was compiled with `binary_crossentropy` as the
reconstruction loss. Binary cross-entropy is appropriate when the
target values are genuine probabilities or binary flags in the range
[0, 1]. The N-BaIoT features, however, are continuous statistics —
weights, means, variances, covariances, Pearson correlation
coefficients — that are normalised to a StandardScaler range of
approximately [-3, 3]. Binary cross-entropy applied to values outside
[0, 1] is mathematically undefined (the logarithm of a negative
number), and the sigmoid activation at the decoder output would
aggressively saturate the reconstruction towards 0 and 1 regardless
of the true feature values, making high-fidelity reconstruction
impossible.

The correct loss function for a continuous-valued autoencoder is
mean-squared error (MSE), which penalises the Euclidean distance
between the input and its reconstruction without making assumptions
about the range of the values. The decoder's final activation was
correspondingly changed from sigmoid to linear, allowing the network
to reconstruct values across the full continuous range of the
normalised features.

It is important to note that binary cross-entropy remains the correct
loss for the classifier head, which produces a single sigmoid output
representing the probability of the malicious class. The two losses
serve different purposes and operate on different outputs; the error
was in using the classification loss for the reconstruction objective.

### 3.3 Training Data Leaked into the Test Set

The script performed three independent dataset splitting operations.
The first split divided the preprocessed features into training,
validation, test, and real-time evaluation subsets. Later in the
script, however, a second split was performed on the full
`features_preprocessed` array — which included the previously
held-out test and real-time subsets — to produce a new `X_train_full`
for the final training run. The model was thus trained on data that
it would subsequently be evaluated against, a fundamental form of
data leakage that renders performance metrics untrustworthy. The
near-perfect reported metrics cannot be fully credited to
generalisation ability when test contamination of this kind is present.

The refactored system performs a single canonical split of the **raw**
rows before anything is fitted, using stratified sampling to keep the
natural class balance in every subset. The split is performed in three
stages — first removing the real-time evaluation set (5% of the total),
then the test set (15%), then the validation set (15%), leaving 65% for
training. Feature selection, imputation, scaling, PCA and SMOTE are
fitted on the training rows only; the validation, test and real-time
rows are passed through the fitted pipeline with `transform`, are never
resampled, and are not used for anything except validation, threshold
tuning (validation set) and final evaluation.

This guarantee was only partly true in the April 2026 refactor: model
training was isolated from the test set, but preprocessing and SMOTE
were still fitted on the whole dataset before the split. That residual
leakage (BUG-25) was found and fixed in November 2026, together with
pretraining the autoencoder on benign rows only (BUG-26).

### 3.4 SMOTE Resampling Was Computed But Never Used

The N-BaIoT dataset is severely imbalanced: as noted in the original
report, 6,506,674 of the 7,062,606 samples are labelled as malicious
(92.1%), while only 555,932 are benign (7.9%). The original code
correctly identified this as a problem and constructed an
`imblearn.Pipeline` applying SMOTE oversampling followed by random
under-sampling to produce a balanced training set. The resampled
arrays `X_resampled` and `y_resampled` were computed but then never
passed to the model. Instead, the model was trained on
`X_train_subset = X_train[:10000]` — a direct slice of the original,
imbalanced training data. The imbalance correction was therefore
entirely ineffective, and the model would have learned a strong prior
towards the majority (malicious) class, inflating recall and
suppressing precision.

The refactored `DataPreprocessor.fit_transform` method returns the
resampled arrays directly, making it impossible to accidentally bypass
the resampling step, since the same arrays are immediately used for
training. Resampling is applied to the training split only; evaluation
sets keep their real class balance and contain no synthetic rows.

### 3.5 The imblearn Pipeline Silently Replaced the sklearn Pipeline

Python's import system allows a name to be rebound at any point in a
module. The script imported `sklearn.pipeline.Pipeline` for use in
the preprocessing `ColumnTransformer`, and then later imported
`imblearn.pipeline.Pipeline` under the same unqualified name
`Pipeline`. From that point onward, `Pipeline` referred to the
imblearn version, which requires a final estimator step and behaves
differently as a transformer. This silent name collision could cause
the preprocessing pipeline to raise unexpected errors or to behave
incorrectly depending on how the affected code path was reached.
The fix imported the imblearn pipeline under the explicit alias
`ImbPipeline`, preserving both symbols throughout the module.

### 3.6 Additional Model Issues

The ART `KerasClassifier` imported from
`adversarial_robustness_toolbox` was immediately overridden by a
locally defined class of the same name, making adversarial robustness
testing via `FastGradientMethod` impossible despite both being
imported. The `find_optimal_threshold` function contained an
off-by-one indexing error: `sklearn.metrics.precision_recall_curve`
returns arrays of length `n+1` for precision and recall but only `n`
for thresholds, so applying `np.nanargmax` to the F1 array and using
the resulting index to access `thresholds` would raise an `IndexError`
whenever the maximum F1 occurred at the last element. Model checkpoints
were saved under three different filenames across the training
callbacks and manual save calls, causing the evaluation code to load
the final epoch's weights rather than the best-validation checkpoint.
The split proportions stated in comments (65/15/15/5) did not match
the actual code (65/17.5/13.1/4.4). Ten or more symbols were imported
but never used, including `shap`, `FastGradientMethod`, `BayesSearchCV`,
`StratifiedKFold`, `clone_model`, and `LSTM`.

---

## 4. Cross-Cutting Architectural Deficiencies

Beyond individual bugs, the original implementation exhibited several
structural properties that would impede any future development,
reproduction, or deployment of the system.

The two source files were written as monolithic scripts, with hard-
coded file paths referencing the developer's local machine
(`/Users/boss/Desktop/...`), making them non-portable by design.
Feature column names — including 120 explicitly listed TTL columns
specific to one capture scenario — were embedded as literals, meaning
that processing a different PCAP file would produce a different TTL
distribution, silently padding the missing columns with zeros and
providing misleading features to the model. The optimal hyperparameters
discovered by Bayesian search were pasted as magic numbers with no
record of when or on what data the search was run, making
reproducibility impossible without re-running the full search. No
random seeds were set for Python, NumPy, or TensorFlow, so results
varied between runs. There were no unit tests for any of the feature
extraction functions, meaning that the bugs described above could only
be discovered by careful manual code review rather than by automated
regression detection.

---

## 5. The Refactored System

The refactored system, IoT Anomaly Detector v1.0.0, resolves all
twenty-five identified issues through a modular Python package with
the following structure.

The `data` subpackage separates concerns into four modules: `parser.py`
handles PCAP reading and session assembly with idle-timeout flushing;
`features.py` contains all session-level statistical functions with
correct key naming and defaults; `label_mapper.py` parses label files
and maps them to DataFrames with case-insensitive matching;
`preprocessor.py` provides a stateful `DataPreprocessor` class
that wraps feature selection, PCA, and SMOTE in a single object:
`fit_transform` is fitted on training rows and returns the resampled
arrays, and `transform` applies the same fitted pipeline to held-out
rows without resampling.

The `models` subpackage separates the encoder, autoencoder, and
classifier into distinct constructors. `build_autoencoder` returns
both the full autoencoder (for Phase 1 pretraining with MSE loss) and
the encoder as a standalone model. `build_classifier` accepts the
encoder model and builds the attention head on top of it, ensuring
that attention is always applied to the bottleneck embedding.

The `training` subpackage encapsulates the two-phase training loop in
a `Trainer` class. A single call to
`Trainer.fit_from_features(features, labels, preprocessor)` performs
the canonical stratified split on the raw rows, fits the preprocessor
on the training rows only, pretrains the autoencoder on benign training
rows, fine-tunes the classifier, loads the best validation checkpoint,
and tunes the decision threshold on the validation set. All hyperparameters are
provided through a `TrainingConfig` dataclass, which can be
instantiated programmatically or loaded from the `configs/default.yaml`
configuration file. Global random seeds for Python, NumPy, and
TensorFlow are set at construction time.

---

## 6. Comparative Analysis: Original Versus Refactored

The practical effect of the fixes can be understood by considering
what the original pipeline was actually computing. With inter-arrival
time features zeroed, flag combinations double-prefixed to
meaningless names, protocol distributions uniformly zero, and labels
encoding every session as benign, the model was trained on a dataset
whose labels were fully uninformative and whose feature vector was
substantially corrupted. That the cross-validation scores were
nonetheless very high suggests that the remaining features — packet
size statistics, payload statistics, and TTL distributions — were
themselves highly discriminative in the N-BaIoT dataset, and that the
model was, perhaps inadvertently, learning from them effectively.

However, the high scores should be interpreted with caution for a
second reason: the test data leakage described in Section 3.3 means
that the evaluation was not strictly held-out. The refactored system
enforces a clean separation between training and evaluation, uses all
feature families correctly, and corrects the class imbalance handling.

These figures were checked in November 2026 by running both the leaky
pipeline and the corrected, split-first pipeline on all nine N-BaIoT
devices (7,062,606 rows) with a shortened 2 + 3 epoch schedule. The
corrected pipeline reached accuracy 0.9988, precision 0.9995, recall
0.9992, F1 0.9993 and AUC-ROC 0.9997 on 1,059,391 real held-out rows,
with a false-positive rate of 0.57%; the leaky pipeline scored slightly
lower. The leakage therefore did not inflate performance on this dataset,
and the 2024 report's level of performance holds under a clean
evaluation. Two limits remain: with 92.1% of rows malicious, accuracy
alone is a weak measure, and a random row split keeps test traffic from
the same devices as the training traffic, so generalisation to unseen
devices is not yet measured.

On the model side, the architectural separation of the encoder from
the full autoencoder is the most consequential change. In the original
system the attention mechanism, however correctly implemented
mathematically, was operating on the wrong representation. In the
refactored system it operates on the compressed bottleneck, focusing
the classification head on the dimensions of the learned embedding
most associated with anomalous behaviour. This is consistent with the
design intent described in the original report and with the broader
literature on attention-augmented autoencoders for anomaly detection.

---

## 7. Conclusion

The original IoT SAFEGUARD project demonstrated sound architectural
thinking and produced a well-structured academic report. The Hybrid
Convolutional Autoencoder with Attention Mechanism is an appropriate
and well-motivated choice for network-level IoT anomaly detection, and
the use of Bayesian hyperparameter optimisation, SMOTE resampling,
and adaptive threshold tuning reflected genuine engagement with the
challenges of imbalanced, high-dimensional IoT traffic data.

The refactoring exercise revealed that the gap between an
architecturally sound design and a correctly functioning implementation
can be substantial, and that numerical machine learning systems are
particularly susceptible to silent failures — bugs that produce
plausible-looking output without errors or warnings. All eleven
critical bugs identified in this audit would have been undetectable
by conventional testing unless the tester knew precisely what output
to expect from each function. This underscores the importance of
unit tests for every data transformation function, explicit
verification of column names and types at pipeline boundaries, and
rigorous separation of training and evaluation data from the earliest
stages of development.

The refactored system, with its modular package structure, stateful
preprocessor, leak-free two-phase training loop, and end-to-end
validation record (`test-outputs/`), provides a foundation from which the project's intended
contributions — real-time IoT anomaly detection, incremental learning,
and adversarial robustness testing — can be pursued with confidence
that the underlying implementation correctly realises the stated
design.

---

## Appendix: Summary of All Changes

| Issue | Original behaviour | Corrected behaviour |
|---|---|---|
| IAT key names (BUG-01) | Returned `packet_size_*` keys | Returns `inter_arrival_*` keys |
| IAT defaults (BUG-07) | Defaulted to 1 | Defaults to 0.0 |
| Flag combination keys (BUG-02) | `flag_comb_flag_comb_N` double-prefix | Single `flag_comb_N` prefix |
| Payload size (BUG-03) | Subtracted IP header twice; negative values | Uses `len(ip.data)` directly |
| Session chunking (BUG-04) | Flushed live sessions at packet count | Idle-timeout flushing only |
| Protocol distribution (BUG-05) | Always zero (case mismatch) | Correct uppercase keys |
| Port range (BUG-06) | Stored as tuple | Separate `_min` / `_max` scalars |
| `Decimal` timestamps (BUG-08) | 30–100× slower, silent cast | `float64` throughout |
| Payload bytes (BUG-09) | Always empty | Reads `packet['payload']` bytes |
| Label file path (BUG-10) | Passed directory, raised `IsADirectoryError` | Raises clear `ValueError` with hint |
| Label encoding (BUG-11) | Searched `'Malicious'`, missed all attacks | Case-insensitive `'malicious'` |
| DataFrame slice assignment (BUG-12) | `SettingWithCopyWarning` | Uses `.loc` throughout |
| Attention target (BUG-13) | Applied to full autoencoder reconstruction | Applied to encoder bottleneck |
| Autoencoder loss (BUG-14) | `binary_crossentropy` on continuous data | `mse` with linear decoder activation |
| Data leakage (BUG-15) | Test set included in `X_train_full` | Single canonical stratified split |
| Reconstruction length (BUG-24) | Decoder output longer than input when PCA length % 4 ≠ 0 | `Cropping1D` trims to input length |
| Preprocessing leakage (BUG-25) | Selection, PCA and SMOTE fitted on all rows before the split | Raw rows split first; fitted on training rows only |
| Pretraining data (BUG-26) | Autoencoder pretrained on attack and synthetic rows | Pretrained on real benign training rows only |
| SMOTE unused (BUG-16) | Resampled arrays discarded | `DataPreprocessor` returns resampled data |
| Pipeline collision (BUG-17) | `imblearn.Pipeline` overwrote `sklearn.Pipeline` | Explicit `ImbPipeline` alias |
| ART import shadowed (BUG-18) | Local class hid ART `KerasClassifier` | Renamed to `KerasAnomalyClassifier` |
| Split proportions (BUG-19) | 65/17.5/13.1/4.4%, not stated 65/15/15/5% | Three-stage split yields exact proportions |
| Threshold IndexError (BUG-20) | `thresholds[n]` off-by-one | Array lengths aligned before indexing |
| Checkpoint filename (BUG-21) | Three different names; loads wrong epoch | One canonical path per phase |
| Dead code (BUG-22) | Triple-quoted string containing old training block | Removed entirely |
| Unused imports (BUG-23) | 10+ unused symbols | Removed; optional imports documented |
| Monolithic scripts (ARCH-01–05) | Hard-coded paths, no tests, magic numbers | Modular package, config file, smoke tests |
