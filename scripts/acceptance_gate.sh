#!/usr/bin/env bash
# Acceptance gate for a full-schedule training run.
#
# A run is accepted only if all three hold:
#   1. training finished with exit code 0 (from <run_dir>/launcher.log, or
#      --skip_training_check when results.json was produced another way);
#   2. the model regression tests pass on this machine (on an Apple GPU this
#      includes the GPU-vs-CPU check for BUG-29);
#   3. scripts/verify_run.py passes every fresh-process check.
#
# Logs go to <run_dir>/verification/. Prints ACCEPTANCE GATE PASSED and exits
# 0 only when all three hold; otherwise prints ACCEPTANCE GATE FAILED and
# exits 1.
#
# Usage:
#   scripts/acceptance_gate.sh --run_dir runs/full_schedule \
#       --data_folder /path/to/N-BaIoT/Dataset \
#       --summary_csv /path/to/N-BaIoT/Dataset/data_summary.csv \
#       [--python .venv/bin/python] [--skip_training_check]

set -u

usage() { sed -n '2,20p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

run_dir="" data_folder="" summary_csv="" check_training=1
repo="$(cd "$(dirname "$0")/.." && pwd)"
python="$repo/.venv/bin/python"
[[ -x "$python" ]] || python="python"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run_dir)             run_dir="$2"; shift 2 ;;
        --data_folder)         data_folder="$2"; shift 2 ;;
        --summary_csv)         summary_csv="$2"; shift 2 ;;
        --python)              python="$2"; shift 2 ;;
        --skip_training_check) check_training=0; shift ;;
        -h|--help)             usage ;;
        *) echo "unknown argument: $1" >&2; usage ;;
    esac
done
[[ -n "$run_dir" && -n "$data_folder" && -n "$summary_csv" ]] || usage

run_dir="$(cd "$run_dir" && pwd)" || { echo "ACCEPTANCE GATE FAILED"; exit 1; }
mkdir -p "$run_dir/verification"
cd "$repo" || exit 1

if (( check_training )); then
    train_status="$(grep -o 'finished with exit code [0-9]*' "$run_dir/launcher.log" 2>/dev/null \
                    | tail -1 | awk '{print $NF}')"
    if [[ "$train_status" != "0" ]]; then
        echo "training exit ${train_status:-missing}"
        echo "ACCEPTANCE GATE FAILED"
        exit 1
    fi
fi

pytest_status=0
verify_status=0

"$python" -m pytest tests/test_models.py -v -p no:cacheprovider \
    -k "gpu or relu or deterministic or dropout" \
    > "$run_dir/verification/gpu_regression_tests.txt" 2>&1 || pytest_status=$?

"$python" scripts/verify_run.py --run_dir "$run_dir" \
    --data_folder "$data_folder" --summary_csv "$summary_csv" \
    > "$run_dir/verification/verify.log" 2>&1 || verify_status=$?

printf 'pytest exit %d\n' "$pytest_status" >> "$run_dir/verification/gpu_regression_tests.txt"
printf 'verify exit %d\n' "$verify_status" >> "$run_dir/verification/verify.log"
echo "training exit $( (( check_training )) && echo 0 || echo skipped) | pytest exit $pytest_status | verify exit $verify_status"

if (( pytest_status != 0 || verify_status != 0 )); then
    echo "ACCEPTANCE GATE FAILED"
    exit 1
fi
echo "ACCEPTANCE GATE PASSED"
