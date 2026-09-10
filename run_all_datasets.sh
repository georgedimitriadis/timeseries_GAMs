#!/usr/bin/env bash
#set -euo pipefail

#MODEL=$1
#if [ -z "$MODEL" ]; then
#    echo "Usage: $0 <model_name>"
#    echo "e.g., $0 dlinear"
#    exit 1
#fi

MODEL=dlinear_autoreg

# --- paths you set once ------------------------------------------------------
# Root of THIS repo (the folder that contains `src/`). Resolved automatically
# as the directory this script lives in.
THIS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Absolute path to the cloned ProbTS benchmark repo (the folder that CONTAINS
# the `probts/` package directory and `run.py`).
PROBTS_REPO="${THIS_REPO}/ptsbenchmark"
# ----------------------------------------------------------------------------

# --- experiment settings ----------------------------------------------------
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

# Multivariate short-term datasets (configs under config/stsf)
MULTIVARIATE_DATASETS=(
    'exchange' 'solar' 'electricity' 'traffic' 'wiki'
)

# Long-term forecasting datasets (configs under config/ltsf)
LONG_TERM_DATASETS=(
    'etth1' 'etth2' 'ettm1' 'ettm2' 'traffic_ltsf' 'electricity_ltsf'
    'exchange_ltsf' 'traffic_ltsf' 'weather_ltsf'
)

CTX_LEN=96
PRED_LENS=(96 132 336 720)

# Multivariate short-term: default lengths, config/stsf
for DATASET in "${MULTIVARIATE_DATASETS[@]}"; do
    echo "=== Running ${MODEL} on ${DATASET} (multivariate, default lengths) ==="
    python run.py --config "${CONFIG}" --seed_everything 0  \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR} \
            --data.data_manager.init_args.split_val true
done

# Long-term: CTX_LEN=96, sweep PRED_LEN, config/ltsf
for DATASET in "${LONG_TERM_DATASETS[@]}"; do
    for PRED_LEN in "${PRED_LENS[@]}"; do
        echo "=== Running ${MODEL} on ${DATASET} (ctx=${CTX_LEN}, pred=${PRED_LEN}) ==="
        python run.py --config "${CONFIG}" --seed_everything 0 \
            --data.data_manager.init_args.path ${DATA_DIR} \
            --trainer.default_root_dir ${LOG_DIR} \
            --data.data_manager.init_args.dataset ${DATASET} \
            --data.data_manager.init_args.split_val true \
            --trainer.max_epochs 50 \
            --data.data_manager.init_args.context_length ${CTX_LEN} \
            --data.data_manager.init_args.prediction_length ${PRED_LEN}
            # --trainer.accelerator=cpu --trainer.devices=1
    done
done