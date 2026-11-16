# Validation Run — 2026-11-16

First full execution of the TESTING_GUIDE_M2Max.md plan (the 2026-04 session
created the phase folders but never ran the tests). Hardware: MacBook Pro
M2 Max, 64 GB, tensorflow-metal (GPU active). Stack: Python 3.11.5,
TensorFlow 2.13.0, numpy 1.24.3, scikit-learn 1.5.2, imbalanced-learn 0.12.4.

## Results by phase

| Phase | Scope | Result |
|---|---|---|
| 0 | Environment, Metal GPU, package imports, CLI entry points | PASS (after fixes 1–2) |
| 1 | N-BaIoT CSV loader, PCAP parser, label mapper (BUG-10/11 guards) | PASS |
| 2 | Feature-extraction correctness (BUG-01/02/05/07 fixes) | PASS — 307 features/session, no negatives |
| 3 | Model architecture build | PASS (after fix 3) for input lengths 8/11/13/15/23 |
| 4 | Two-phase training (2 pretrain + 3 finetune epochs, 60k smoke subset) | PASS (after fix 4) |
| 5 | Test-set evaluation | PASS — acc 0.9986, prec 0.9978, rec 0.9993, F1 0.9985, AUC 0.9995 |
| 6 | PCAP → features → labels → CSV (CTU-IoT-Malware-Capture-34-1 irc.pcap) | PASS — 4,055 sessions, labels joined |
| 7 | Adversarial robustness (ART) | NOT RUN — optional extra, ART not installed |
| 8 | Real-time inference | PASS — 3,000 samples, F1 0.9963 |
| 9 | Integration (load → preprocess → train → eval → realtime in one run) | PASS |

Smoke-run metrics are consistent with the report's full-run numbers
(acc 99.41%, AUC 99.97%) despite the reduced epochs and 3-file subset
(1.benign / 1.mirai.ack / 1.gafgyt.scan, 181,592 rows → PCA 11 components,
SMOTE-balanced 264,088 → 60,000 sampled).

## Bugs found and fixed in this run

1. **Broken package import** — `src/iot_anomaly/data/__init__.py` exported
   `extract_session_features`, which does not exist (renamed to
   `compute_session_features` in the refactor). Every `from iot_anomaly.data
   import …` failed.
2. **Broken console scripts** — `setup.py` entry points referenced
   `scripts.train:main` / `scripts.process_pcap:main`, but `scripts/` is not
   an installed package, so `iot-train` and `iot-process-pcap` crashed with
   ModuleNotFoundError. CLI code moved into `iot_anomaly.cli.*`; `scripts/`
   kept as thin wrappers.
3. **BUG-23: autoencoder shape mismatch** — the decoder emits
   `4*ceil(ceil(L/2)/2)` time-steps, which exceeds the input length whenever
   L % 4 != 0 (here PCA produced L=11 → decoder emitted 12). Training crashed
   on the MSE loss. Fixed with a `Cropping1D` layer sized from the actual
   tensor shape.
4. **Checkpoint save crash** — TF 2.13's `ModelCheckpoint` passes an
   `options` kwarg that the native `.keras` format rejects
   (`ValueError: … not supported with the native Keras format`). Switched to
   weights-only checkpoints (`*.weights.h5`, `save_weights_only=True`,
   `load_weights`), which is also Keras-3-compatible for the planned upgrade.

## Conclusion

The pipeline is functional end-to-end on both dataset families and its
metrics are consistent with the 2024 report. Phase 7 (adversarial, optional)
is the only guide item not exercised.
