# Corrected full-schedule run — verified 2026-11-18

This is the full-schedule N-BaIoT run trained after the BUG-28 and BUG-29
fixes. It passed the acceptance gate (`scripts/acceptance_gate.sh`). It
replaces the earlier run in `../full-schedule-2026-11-18/`, which is kept
only as a superseded record.

## Provenance

| Item | Value |
|---|---|
| Training commit | `71bf719` (recorded by the launcher; see `run_log_summary.txt`, first line) |
| Training | 2026-11-18, 10:31 to 15:11 GMT; exit code 0; MacBook Pro M2 Max, tensorflow-metal 1.0.1 |
| Gate commit | `4e60127` (clean working tree, unchanged during the gate) |
| Gate script SHA-256 | `362d3f54…30fd47` (full value in `gate_provenance.txt`) |
| Gate verdict | training exit 0, pytest exit 0, verify exit 0: **ACCEPTANCE GATE PASSED** |

The run was launched with an earlier copy of `scripts/full_schedule_run.py`
that did not yet write a `provenance` block, so `results.json` has none. The
verifier therefore takes the training commit from the launcher log. Between
the training commit and the gate commit, the only change under `src/` is
the BUG-30 fix in `cli/process_pcap.py`, which the N-BaIoT path does not
use.

## Results (held-out test split, threshold 0.1176)

1,059,391 real rows: 83,390 benign and 976,001 malicious (92.13% malicious).
The counts come from `confusion_matrix(...).ravel()`. The freshly loaded
model reproduced them exactly.

| | Predicted benign | Predicted malicious |
|---|---|---|
| **Actually benign** (83,390) | 83,007 | 383 (false alarms) |
| **Actually malicious** (976,001) | 322 (missed attacks) | 975,679 |

| Metric | Test split | Real-time split |
|---|---|---|
| Rows (benign / malicious) | 83,390 / 976,001 | 27,797 / 325,334 |
| Accuracy | 0.99933 | 0.99935 |
| Precision | 0.99961 | 0.99964 |
| Recall | 0.99967 | 0.99966 |
| F1 | 0.99964 | 0.99965 |
| AUC-ROC | 0.99991 | 0.99994 |
| False-positive rate | 0.459% (383) | 0.421% (117) |
| Missed attacks | 322 | 112 |

Early stopping (patience 10, monitoring validation loss) ended training as
follows:
- Pretraining stopped at epoch 25 of 30 (best epoch 15). It used 361,355
  real benign training rows.
- Fine-tuning stopped at epoch 28 of 50 (best epoch 18).

The best weights were used. Wall-clock time was 4.66 h. PCA kept 11
components.

## Fresh-process integrity checks (`verification.json`)

All 16 checks passed. The key measured differences are:

| Check | Result |
|---|---|
| `predict()` vs eager call | max abs diff 1.2e-7 |
| `predict()` vs NumPy forward pass **with** ReLU | 1.2e-7 |
| `predict()` vs NumPy forward pass **without** ReLU (control) | 0.955, so the ReLU is active |
| GPU graph mode vs CPU | 2.5e-6 |
| Threshold re-tuned on validation split | 0.11759631, identical to the run |

The NumPy overflow warning in `verify.log` comes from the no-ReLU control
computation, where some logits are very large. It does not affect any check.

## Files

- `results.json`: the run's configuration, split sizes and metrics.
- `verification.json`, `verify.log`: the fresh-process verification.
- `gpu_regression_tests.txt`: the local GPU regression tests, 5 passed.
- `gate_provenance.txt`: the gate's commit, script hash, tree state and verdict.
- `roc_curve.png`, `confusion_matrix.png`, `roc_points_test.csv`: plots and
  ROC points regenerated from the freshly loaded model. The training
  process's own plots were byte-identical.
- `run_log_summary.txt`: the condensed training log, with local paths removed.

## Caveats

- The result comes from a single seed (42), so it has no confidence
  interval.
- Rows are split at random within devices, so the test traffic comes from
  the same devices as the training traffic.
- Flagging everything would already reach 92.1% accuracy. The false-positive
  rate is the more telling figure for the benign class.
- The figures are at the same general level as the 2024 report, but this is
  not a direct reproduction, because the evaluation protocol differs.
