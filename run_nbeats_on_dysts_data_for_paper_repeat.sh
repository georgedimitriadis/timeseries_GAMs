#!/usr/bin/env bash
# Run run_nbeats_dysts.py multiple times, saving versioned outfiles.
#
# Usage:
#   ./run_nbeats_repeat.sh N [args passed through to run_nbeats_dysts.py]
#
# N = number of runs.
# All args after N are forwarded to the python script.
# If --outfile is passed through, it is ignored (this script sets it per run).
#
# Examples:
#   ./run_nbeats_repeat.sh 10 --systems all --compare
#   ./run_nbeats_repeat.sh 5 --systems Lorenz

set -euo pipefail

# ---- config ----
PYTHON="${PYTHON:-python}"
SCRIPT="${SCRIPT:-run_nbeats_on_dysts_data_for_paper.py}"
DATANAME="multivariate__pts_per_period_100__periods_12"
OUTDIR="${OUTDIR:-dysts_data/results}"
OUTBASE="${OUTBASE:-results_${DATANAME}_mine}"

# ---- parse N ----
if [ "$#" -lt 1 ]; then
  echo "usage: $0 N [args...]" >&2
  exit 1
fi
N="$1"
shift

# ---- strip any user-supplied --outfile (and its value) from passthrough args ----
PASS_ARGS=()
skip_next=0
for arg in "$@"; do
  if [ "$skip_next" -eq 1 ]; then
    skip_next=0
    continue
  fi
  case "$arg" in
    --outfile)
      skip_next=1
      ;;
    --outfile=*)
      ;;
    *)
      PASS_ARGS+=("$arg")
      ;;
  esac
done

mkdir -p "$OUTDIR"

# ---- run loop ----
for i in $(seq 1 "$N"); do
  OUTFILE="${OUTDIR}/${OUTBASE}_v${i}.json"
  echo "=== run ${i}/${N} -> ${OUTFILE} ==="
  "$PYTHON" "$SCRIPT" "${PASS_ARGS[@]}" --outfile "$OUTFILE"
done