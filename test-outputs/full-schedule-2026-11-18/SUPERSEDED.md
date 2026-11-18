# Superseded — kept as a historical record

The files in this folder come from the full-schedule run that finished on
2026-11-18 at 06:51 (`results.json`, `roc_curve.png`, `confusion_matrix.png`,
`run_log_summary.txt`). They are left unchanged so the history stays
auditable, but **their figures are not reliable measurements** and must not
be quoted as results.

The run was trained and evaluated with two model bugs found later the same
day by the test suite (see `BUG_ANALYSIS.md`, section 3c):

- **BUG-28** — the encoder was called with `training=True` stored in the
  graph, so its dropout (rate 0.43) stayed active during prediction. The
  reported metrics depended on dropout masks: re-evaluating the same
  weights in a fresh process gave 4,550 test false alarms instead of the
  425 recorded here.
- **BUG-29** — on the Apple GPU (tensorflow-metal 1.0.1) the classifier's
  hidden ReLU was dropped in graph mode, so the model trained here had a
  linear hidden layer, unlike the documented architecture.

This run has been replaced by the corrected full-schedule run (training
commit `71bf719`). That run passed fresh-process verification and the
acceptance gate on 2026-11-18; see `../full-schedule-2026-11-18-fixed/`.
