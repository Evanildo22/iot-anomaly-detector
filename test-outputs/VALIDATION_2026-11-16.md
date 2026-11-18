# Validation Record — 2026-11-16 to 2026-11-18

> **Notice (2026-11-18).** The test suite added afterwards exposed two model
> bugs that affect every model-evaluation figure in this record: encoder
> dropout stayed active at prediction time (BUG-28), and on the Apple GPU
> the classifier's hidden ReLU was dropped (BUG-29). See
> `BUG_ANALYSIS.md` section 3c. The figures below are kept as the record of
> what was run, but they are not reliable measurements; the full-schedule
> training is being rerun with both fixes.

First full execution of the TESTING_GUIDE_M2Max.md plan (the 2026-04 session
created the phase folders but never ran the tests). Hardware: MacBook Pro
M2 Max, 64 GB, tensorflow-metal (GPU active). Stack: Python 3.11.5,
TensorFlow 2.13.0, numpy 1.24.3, scikit-learn 1.5.2, imbalanced-learn 0.12.4.

The first run (2026-11-16) made the pipeline work end to end. A review on
2026-11-17 found that preprocessing and SMOTE were fitted before the
train/test split (BUG-25), so the training flow was corrected and every
phase's evidence was regenerated with the corrected pipeline.

## Results by phase (corrected pipeline)

| Phase | Scope | Result |
|---|---|---|
| 0 | Environment, Metal GPU, package imports, CLI entry points | PASS (after fixes 1–2) |
| 1 | N-BaIoT CSV loader, PCAP parser, label mapper (BUG-10/11 guards) | PASS |
| 2 | Feature correctness (BUG-01/02/05/07); split-first preprocessing | PASS — 307 features/session; splits disjoint; test set keeps natural balance |
| 3 | Model architecture build | PASS (after fix 3) for input lengths 8/11/13/15/23 |
| 4 | Two-phase training (2 pretrain + 3 finetune epochs) | PASS (after fixes 4, 6, 7) |
| 5 | Held-out test-set evaluation (single-device subset) | PASS — acc 0.9990, prec 0.9990, rec 0.9995, F1 0.9993, AUC 1.0000 (rounded); 19 false alarms of 7,432 benign rows, 9 missed attacks |
| 6 | PCAP → features → labels → CSV (CTU-IoT-Malware-Capture-34-1 irc.pcap) | PASS — 4,055 sessions, labels joined |
| 7 | Adversarial robustness (ART) | NOT RUN — optional extra, ART not installed |
| 8 | Real-time inference | PASS — 9,080 real rows, F1 0.9990 |
| 9 | Integration (load → split → preprocess → train → eval → realtime) | PASS |

The phase 3–5, 8 and 9 evidence was regenerated on 2026-11-18 with the
BUG-27/28/29 fixes, so this table reflects the fixed model.

Phases 2–5, 8 and 9 use the single-device subset (1.benign / 1.mirai.ack /
1.gafgyt.scan, 181,592 rows → PCA 11 components) at smoke scale.

## Old and corrected pipelines on the full dataset

Both pipelines were run on all nine N-BaIoT devices (7,062,606 rows:
555,932 benign, 6,506,674 malicious) with the report's hyperparameters,
seed 42, and a shortened schedule of 2 pretraining + 3 fine-tuning epochs.

**These two columns are not a controlled before/after comparison.** The
two test sets have different class distributions (the old one is roughly
balanced, the corrected one is 92.13% malicious), and the old one is
mostly synthetic SMOTE rows. Accuracy, precision, recall and F1 depend on
the class distribution, so their values cannot be compared across the two
columns.

| Metric | Old pipeline (leaky) | Corrected pipeline |
|---|---|---|
| Accuracy | 0.9983 | 0.9988 |
| Precision | 0.9983 | 0.9995 |
| Recall | 0.9982 | 0.9992 |
| F1 | 0.9983 | 0.9993 |
| AUC-ROC | 0.9997 | 0.9997 |
| Test set | 1,952,003 rows, 976,002 benign of which ~893,000 synthetic | 1,059,391 real rows (83,390 benign, 976,001 malicious) |

Checks on the corrected run: the four splits are disjoint and cover every
row; the test set's malicious share (0.9213) equals the dataset's; PCA kept
11 components; the validation-tuned threshold is 0.1569.

Corrected confusion counts on the test set, taken directly from
`confusion_matrix(y_test, y_pred).ravel()`:

| | Predicted benign | Predicted malicious |
|---|---|---|
| **Actually benign** (83,390) | 82,915 (true negatives) | 475 (false alarms) |
| **Actually malicious** (976,001) | 803 (missed attacks) | 975,198 (true positives) |

False-positive rate: 475 / 83,390 = 0.570% (99.43% of benign rows correctly
passed). The counts were extracted by rebuilding the run's deterministic
split and fitted preprocessing, loading its saved best-classifier weights
and re-tuning the threshold on the validation set (0.15695); the reproduced
predictions match all five of the run's metrics exactly.

**Interpretation.** After removing the leakage, performance remained strong
on a real, naturally distributed hold-out set. Because the old and
corrected test sets have different class distributions, and the old set
includes synthetic samples, the metric values should not be treated as a
controlled before/after comparison; this experiment cannot show how much,
if at all, the leakage inflated the old figures. The nearly identical
AUC-ROC (which does not depend on the decision threshold) suggests that the
model's ability to separate the classes stayed strong. Further caveats:

- Because 92.1% of rows are malicious, a model that flags everything would
  reach 92.1% accuracy. The false-positive rate is the more telling figure
  for the benign class.
- Rows are split at random, so the test set comes from the same devices and
  capture sessions as the training set. Per-device hold-out evaluation
  (roadmap Phase 3) is needed to measure generalisation to unseen devices.

## Full-schedule run with the corrected pipeline (2026-11-17/18)

The corrected pipeline was then trained with the report's full schedule
(30 pretraining + 50 fine-tuning epochs, early-stopping patience 10) on all
nine devices, with the same hyperparameters, seed 42 and therefore the same
split and test set as the shortened run above. Wall-clock time was 6.98
hours (6.92 hours of training) on the MacBook Pro M2 Max.

- Pretraining stopped at epoch 25 (best validation loss at epoch 15);
  fine-tuning stopped at epoch 43 (best at epoch 33). The best weights were
  used for evaluation.
- PCA kept 11 components; the validation-tuned threshold is 0.0956.

Because the two runs share the same test set, this comparison is a direct
one: only the training schedule changed.

| Metric | Shortened run (2 + 3 epochs) | Full schedule (30 + 50 epochs) |
|---|---|---|
| Accuracy | 0.99879 | 0.99913 |
| Precision | 0.99951 | 0.99956 |
| Recall | 0.99918 | 0.99949 |
| F1 | 0.99935 | 0.99953 |
| AUC-ROC | 0.99973 | 0.99988 |
| False alarms (FP) | 475 | 425 |
| Missed attacks (FN) | 803 | 501 |
| False-positive rate | 0.570% | 0.510% |

Full-schedule confusion counts on the test set (from
`confusion_matrix(y_test, y_pred).ravel()` in the run itself):

| | Predicted benign | Predicted malicious |
|---|---|---|
| **Actually benign** (83,390) | 82,965 (true negatives) | 425 (false alarms) |
| **Actually malicious** (976,001) | 501 (missed attacks) | 975,500 (true positives) |

On the separate real-time split (353,131 rows: 27,797 benign, 325,334
malicious) the same model gave accuracy 0.99915, F1 0.99954, AUC-ROC
0.99988, with 124 false alarms (FPR 0.446%) and 176 missed attacks.

The longer schedule reduced missed attacks by 37.6% (803 → 501) and false
alarms by 10.5% (475 → 425). Each figure comes from a single seed, so the
size of these differences has no confidence interval yet; repeated seeds or
k-fold evaluation (roadmap Phase 3) would provide one. Metrics, plots and a
condensed training log are in `full-schedule-2026-11-18/`.

## Bugs found and fixed

1. **Broken package import** — `src/iot_anomaly/data/__init__.py` exported
   `extract_session_features`, which does not exist (renamed to
   `compute_session_features` in the refactor). Every `from iot_anomaly.data
   import …` failed.
2. **Broken console scripts** — `setup.py` entry points referenced
   `scripts.train:main` / `scripts.process_pcap:main`, but `scripts/` is not
   an installed package, so `iot-train` and `iot-process-pcap` crashed with
   ModuleNotFoundError. CLI code moved into `iot_anomaly.cli.*`; `scripts/`
   kept as thin wrappers.
3. **BUG-24: autoencoder shape mismatch** — the decoder emits
   `4*ceil(ceil(L/2)/2)` time-steps, which exceeds the input length whenever
   L % 4 != 0 (here PCA produced L=11 → decoder emitted 12). Training crashed
   on the MSE loss. Fixed with a `Cropping1D` layer sized from the actual
   tensor shape.
4. **Checkpoint save crash** — TF 2.13's `ModelCheckpoint` passes an
   `options` kwarg that the native `.keras` format rejects
   (`ValueError: … not supported with the native Keras format`). Switched to
   weights-only checkpoints (`*.weights.h5`, `save_weights_only=True`,
   `load_weights`), which is also Keras-3-compatible for the planned upgrade.
   The best classifier is additionally exported as `classifier_best.keras`;
   load it with `load_model(path, compile=False)`, because TF 2.13 on Apple
   Silicon saves legacy-Adam optimizer state that cannot be restored.
5. **CTU training branch crash** — the PCAP feature CSV keeps the raw
   `protocol` string, which crashed feature selection
   (`could not convert string to float: 'TCP'`). The column is now dropped;
   the numeric TCP/UDP/ICMP columns already encode it.
6. **BUG-25: preprocessing leakage** — feature selection, imputation,
   scaling, PCA and SMOTE were fitted on the whole dataset before the split,
   so the test set contained synthetic rows and influenced every fitted
   step. `Trainer.fit_from_features` now splits the raw rows first and fits
   everything on the training rows only.
7. **BUG-26: pretraining data** — the autoencoder was pretrained on attack
   and synthetic rows. It now uses real benign training rows only.
8. **Training command hang** — `plot_roc_curve` and `plot_confusion_matrix`
   called `plt.show()` even after saving to a file, which blocks a terminal
   run on macOS until the plot windows are closed. They now close the figure
   when a save path is given.

## Conclusion

The pipeline is functional end to end on both dataset families and its
evaluation is now leak-free. The corrected full-schedule run achieves
performance at the same general level as the 2024 report, but it is not a
direct reproduction: the hyperparameters and epoch budget match, but the
evaluation protocol differs. Phase 7 (adversarial, optional) is the only guide item not
exercised.
