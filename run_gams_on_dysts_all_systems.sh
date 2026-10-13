#!/usr/bin/env bash
# Experiment 6: GAM (ec), autoregressive, no teacher forcing, on all dysts systems
# (train_multivariate__pts_per_period_100__periods_12.json.gz, 133 systems), in parallel on CPU.
# One run of src/run_gams_on_dysts.py per system.
#
# Usage:
#   ./run_gams_dysts_all_systems.sh [gam_config.yaml] [extra args for run_gams_on_dysts.py]
#   e.g. ./run_gams_dysts_all_systems.sh src/configs/gams/gams_autoreg_dysts.yaml --predict_increment
#
# Environment variables (optional):
#   TOTAL_CPUS=40      CPUs to use
#   CORES_PER_JOB=1    cores per run  -> number of parallel workers = TOTAL_CPUS / CORES_PER_JOB
#   FIRST_CPU=0        first CPU id (workers use FIRST_CPU .. FIRST_CPU+TOTAL_CPUS-1)
#   SYSTEMS="Lorenz Rossler"   subset of systems (default: all systems in the train file)
#   ROOT_DIR=dysts_data/results/gams
#   SEED=0
#   DRY_RUN=1          only print the commands and the worker/core assignment
#
# Scheduling: systems are ordered by dimension D (largest first: D GAMs per system);
# one shared queue; each worker is pinned to its own cores (taskset) and, when free,
# takes the next system (counter file + flock). Thread pools (XLA, BLAS) are limited
# to CORES_PER_JOB.
#
# Output: <ROOT_DIR>/<system>/{result.json, summary.json, gam_state.pkl}
#         <ROOT_DIR>/job_logs/<system>.log            (stdout + stderr of each run)
#         <ROOT_DIR>/results_<DATANAME>_GAMAutoReg.json  all result.json merged:
#             {system: {"GAMAutoReg": {...}}}, the format of the N-BEATS results file

THIS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GAM_CONFIG="${1:-${THIS_REPO}/src/configs/gams/gams_autoreg_dysts.yaml}"
shift || true
EXTRA_ARGS=("$@")

TOTAL_CPUS="${TOTAL_CPUS:-40}"
CORES_PER_JOB="${CORES_PER_JOB:-1}"
FIRST_CPU="${FIRST_CPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"
N_WORKERS=$(( TOTAL_CPUS / CORES_PER_JOB ))

DATANAME="multivariate__pts_per_period_100__periods_12"
DATA_DIR="${THIS_REPO}/dysts_data/dysts_data/data"
TRAIN_FILE="${DATA_DIR}/train_${DATANAME}.json.gz"
ROOT_DIR="${ROOT_DIR:-${THIS_REPO}/dysts_data/results/gams}"
JOB_LOG_DIR="${ROOT_DIR}/job_logs"
SCRIPT="${THIS_REPO}/src/run_gams_on_dysts.py"
MERGED="${ROOT_DIR}/results_${DATANAME}_GAMAutoReg.json"

export PYTHONPATH="${THIS_REPO}:${THIS_REPO}/src:${THIS_REPO}/ptsbenchmark:${PYTHONPATH:-}"
export KERAS_BACKEND=jax
export JAX_PLATFORMS=cpu          # CPU only
export CUDA_VISIBLE_DEVICES=""
# limit thread pools to the cores of one worker
export OMP_NUM_THREADS="${CORES_PER_JOB}"
export MKL_NUM_THREADS="${CORES_PER_JOB}"
export OPENBLAS_NUM_THREADS="${CORES_PER_JOB}"
if [ "${CORES_PER_JOB}" -eq 1 ]; then
    export XLA_FLAGS="--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
fi

if [ $(( FIRST_CPU + TOTAL_CPUS )) -gt "$(nproc --all)" ]; then
    echo "CPUs ${FIRST_CPU}-$(( FIRST_CPU + TOTAL_CPUS - 1 )) requested, machine has $(nproc --all)"; exit 1
fi
mkdir -p "${JOB_LOG_DIR}"

# ---------------------------------------------------------------------------
# Job list: systems ordered by dimension, largest first (SYSTEMS overrides)
# ---------------------------------------------------------------------------
if [ -n "${SYSTEMS:-}" ]; then
    read -r -a JOBS <<< "${SYSTEMS}"
else
    mapfile -t JOBS < <(python - "${TRAIN_FILE}" <<'EOF'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f:
    d = json.load(f)
dims = {k: len(v["values"][0]) for k, v in d.items()}
for k in sorted(dims, key=lambda k: (-dims[k], k)):
    print(k)
EOF
)
fi
[ "${#JOBS[@]}" -eq 0 ] && { echo "No systems found in ${TRAIN_FILE}"; exit 1; }

job_cmd() {   # prints the python command for one system
    local system=$1
    echo python "${SCRIPT}" --system "${system}" --gam_config "${GAM_CONFIG}" \
        --data_dir "${DATA_DIR}" --root_dir "${ROOT_DIR}" --seed "${SEED}" "${EXTRA_ARGS[@]}"
}

QUEUE="${JOB_LOG_DIR}/.queue_counter"

next_job() {  # atomically take the next job index from the shared counter
    (
        flock -x 9
        local i
        i=$(cat "${QUEUE}")
        echo $(( i + 1 )) > "${QUEUE}"
        echo "${i}"
    ) 9>"${QUEUE}.lock"
}

worker() {    # worker w: pinned to its own cores, takes systems until the queue is empty
    local w=$1
    local first=$(( FIRST_CPU + w * CORES_PER_JOB ))
    local last=$(( first + CORES_PER_JOB - 1 ))
    local i system log cmd
    while true; do
        i=$(next_job)
        [ "${i}" -ge "${#JOBS[@]}" ] && break
        system="${JOBS[$i]}"
        log="${JOB_LOG_DIR}/${system}.log"
        cmd=$(job_cmd "${system}")
        if [ "${DRY_RUN}" == "1" ]; then
            echo "[worker ${w} | cpus ${first}-${last}] job ${i} ${system}: ${cmd}"
            continue
        fi
        echo "[$(date '+%F %T')] worker ${w} (cpus ${first}-${last}) START ${system}"
        taskset -c "${first}-${last}" ${cmd} > "${log}" 2>&1
        local rc=$?
        if [ "${rc}" -eq 0 ]; then
            echo "[$(date '+%F %T')] worker ${w} DONE  ${system}"
        else
            echo "[$(date '+%F %T')] worker ${w} FAIL  ${system} (exit ${rc}, see ${log})"
            echo "${system} ${rc}" >> "${JOB_LOG_DIR}/failed.txt"
        fi
    done
}

echo "${#JOBS[@]} systems, ${N_WORKERS} workers x ${CORES_PER_JOB} cores (cpus ${FIRST_CPU}-$(( FIRST_CPU + N_WORKERS * CORES_PER_JOB - 1 )))"
echo "GAM config: ${GAM_CONFIG}  extra args: ${EXTRA_ARGS[*]:-none}"
rm -f "${JOB_LOG_DIR}/failed.txt"
echo 0 > "${QUEUE}"

for (( w = 0; w < N_WORKERS && w < ${#JOBS[@]}; w++ )); do
    worker "${w}" &
done
wait
rm -f "${QUEUE}" "${QUEUE}.lock"
[ "${DRY_RUN}" == "1" ] && exit 0

# ---------------------------------------------------------------------------
# Merge <ROOT_DIR>/<system>/result.json -> one file (N-BEATS results format)
# ---------------------------------------------------------------------------
python - "${ROOT_DIR}" "${MERGED}" "${JOBS[@]}" <<'EOF'
import json, sys
from pathlib import Path
root, out, systems = Path(sys.argv[1]), sys.argv[2], sys.argv[3:]
merged, missing = {}, []
for s in systems:
    p = root / s / "result.json"
    if p.exists():
        merged.update(json.loads(p.read_text()))
    else:
        missing.append(s)
with open(out, "w") as f:
    json.dump(merged, f, indent=2)
print(f"merged {len(merged)} systems -> {out}; missing: {missing or 'none'}")
EOF

if [ -s "${JOB_LOG_DIR}/failed.txt" ]; then
    echo "Failed systems (name, exit code):"; cat "${JOB_LOG_DIR}/failed.txt"
else
    echo "All systems finished."
fi