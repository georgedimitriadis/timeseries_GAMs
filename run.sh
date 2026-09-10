#!/usr/bin/env bash
set -euo pipefail

# ============================================================================
# Launcher that runs the ProbTS benchmark with a model defined in THIS repo,
# without modifying the benchmark repo at all.
#
# It works by putting two directories on PYTHONPATH:
#   1. the ProbTS benchmark repo root  -> so `import probts...` works
#   2. this repo's root                -> so `import src.models...` works
# and then calling the benchmark's own run.py with OUR config file.
# ============================================================================

# --- paths you set once ------------------------------------------------------
# Root of THIS repo (the folder that contains `src/`). Resolved automatically
# as the directory this script lives in.
THIS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute path to the cloned ProbTS benchmark repo (the folder that CONTAINS
# the `probts/` package directory and `run.py`).
PROBTS_REPO="${THIS_REPO}/ptsbenchmark"
# ----------------------------------------------------------------------------

# --- experiment settings -----------------------------------------------------
MODEL=dlinear_autoreg          # just used to pick the config file name below
DATASET=electricity_ltsf
CTX_LEN=96
PRED_LEN=720

DATA_DIR=./datasets
LOG_DIR=./log_dir

# Our config lives in THIS repo, not in the benchmark repo.
CONFIG="${THIS_REPO}/src/configs/${MODEL}.yaml"
# ----------------------------------------------------------------------------

# Make both packages importable. Prepend so our code wins on any name clash.
export PYTHONPATH="${THIS_REPO}:${PROBTS_REPO}:${PYTHONPATH:-}"

# Call the benchmark's own run.py by path. Relative paths in the args below
# (./datasets, ./log_dir) are resolved against the benchmark repo root, so we
# cd there first to match how the benchmark normally runs. Adjust if you keep
# datasets elsewhere (use absolute DATA_DIR/LOG_DIR to avoid ambiguity).
cd "${PROBTS_REPO}"

python run.py --config "${CONFIG}" --seed_everything 0 \
    --data.data_manager.init_args.path "${DATA_DIR}" \
    --trainer.default_root_dir "${LOG_DIR}" \
    --data.data_manager.init_args.dataset "${DATASET}" \
    --data.data_manager.init_args.split_val true \
    --trainer.max_epochs 50 \
    --data.data_manager.init_args.context_length "${CTX_LEN}" \
    --data.data_manager.init_args.prediction_length "${PRED_LEN}"
    # --trainer.accelerator=cpu --trainer.devices=1