# Validation Record — 2026-11-16 / 2026-11-17

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
| 5 | Held-out test-set evaluation (single-device subset) | PASS — acc 0.9988, prec 0.9989, rec 0.9995, F1 0.9992, AUC 0.9996 |
| 6 | PCAP → features → labels → CSV (CTU-IoT-Malware-Capture-34-1 irc.pcap) | PASS — 4,055 sessions, labels joined |
| 7 | Adversarial robustness (ART) | NOT RUN — optional extra, ART not installed |
| 8 | Real-time inference | PASS — 9,080 real rows, F1 0.9989 |
| 9 | Integration (load → split → preprocess → train → eval → realtime) | PASS |

Phases 2–5, 8 and 9 use the single-device subset (1.benign / 1.mirai.ack /
1.gafgyt.scan, 181,592 rows → PCA 11 components) at smoke scale.

## Old versus corrected pipeline on the full dataset

Both pipelines were run on all nine N-BaIoT devices (7,062,606 rows:
555,932 benign, 6,506,674 malicious) with the report's hyperparameters,
seed 42, and a shortened schedule of 2 pretraining + 3 fine-tuning epochs.

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

Corrected confusion counts on the test set: 975,198 true positives, 803
missed attacks, 475 false alarms, 82,915 true negatives. That is a
false-positive rate of 0.57% (99.43% of benign rows correctly passed).

**Interpretation.** The leakage did not inflate the scores on N-BaIoT: the
corrected pipeline scores as high or higher, so the 2024 report's level of
performance is not an artefact of leakage. Two caveats remain:

- Because 92.1% of rows are malicious, a model that flags everything would
  reach 92.1% accuracy. The false-positive rate is the more telling figure
  for the benign class.
- Rows are split at random, so the test set comes from the same devices and
  capture sessions as the training set. Per-device hold-out evaluation
  (roadmap Phase 3) is needed to measure generalisation to unseen devices.

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

The pipeline is functional end to end on both dataset families, its
evaluation is now leak-free, and on the full N-BaIoT dataset it matches or
exceeds the 2024 report's figures. Phase 7 (adversarial, optional) is the
only guide item not exercised.
