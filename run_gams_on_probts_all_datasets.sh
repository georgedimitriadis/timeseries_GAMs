#!/usr/bin/env bash
# Experiment 3: GAM (ec), autoregressive, no teacher forcing, on all ProbTS
# datasets: 5 short-term + 9 long-term x 4 horizons = 41 runs, in parallel on CPU.
#   short-term : context = prediction = 24 (solar, electricity, traffic),
#                30 (exchange, wiki)
#   long-term  : context 96; prediction 96/192/336/720
#                (illness_ltsf: 24/36/48/60)
#
# Usage:
#   ./run_gams_probts_all_datasets.sh [gam_config.yaml]
#
# Environment variables (optional):
#   TOTAL_CPUS=40      CPUs to use
#   CORES_PER_JOB=1    cores per run  -> number of parallel workers = TOTAL_CPUS / CORES_PER_JOB
#   FIRST_CPU=0        first CPU id (workers use FIRST_CPU .. FIRST_CPU+TOTAL_CPUS-1)
#   DRY_RUN=1          only print the commands and the worker/core assignment
#
# Scheduling: jobs are listed from (roughly) largest to smallest in one shared
# queue; each worker is pinned to its own cores (taskset) and, when free, takes
# the next job from the queue (counter file + flock). Thread pools (XLA, torch,
# BLAS) are limited to CORES_PER_JOB so workers do not compete for cores.
#
# Output: ptsbenchmark/log_dir_gams/<dataset>_GAMAutoReg_CTX.._PRED.._seed0/
#         ptsbenchmark/log_dir_gams/job_logs/<dataset>_ctx<c>_pred<p>.log   (stdout + stderr of each run)

THIS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROBTS_REPO="${THIS_REPO}/ptsbenchmark"
GAM_CONFIG="${1:-${THIS_REPO}/src/configs/gams/gams_autoreg.yaml}"

TOTAL_CPUS="${TOTAL_CPUS:-45}"
CORES_PER_JOB="${CORES_PER_JOB:-1}"
FIRST_CPU="${FIRST_CPU:-0}"
DRY_RUN="${DRY_RUN:-0}"
N_WORKERS=$(( TOTAL_CPUS / CORES_PER_JOB ))

DATA_DIR="${PROBTS_REPO}/datasets"
LOG_DIR="${PROBTS_REPO}/log_dir_gams"
JOB_LOG_DIR="${LOG_DIR}/job_logs"
CONFIG_BASE="${THIS_REPO}/src/configs"
SCRIPT="${THIS_REPO}/src/run_gams_on_probts.py"
SEED=0

export PYTHONPATH="${THIS_REPO}/src:${PROBTS_REPO}:${PYTHONPATH:-}"
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
# Job list, roughly largest first. Format: "<kind> <config dir> <context> <prediction>"
#   kind = stsf (src/configs/stsf/<dir>) | ltsf (src/configs/ltsf/<dir>)
# ---------------------------------------------------------------------------
JOBS=(
    "ltsf traffic_ltsf 96 720"      "ltsf traffic_ltsf 96 336"      "ltsf electricity_ltsf 96 720"
    "ltsf traffic_ltsf 96 192"      "ltsf electricity_ltsf 96 336"  "stsf wiki 30 30"
    "ltsf traffic_ltsf 96 96"       "ltsf electricity_ltsf 96 192"  "stsf traffic 24 24"
    "ltsf weather_ltsf 96 720"      "ltsf electricity_ltsf 96 96"   "ltsf ettm1 96 720"
    "ltsf ettm2 96 720"             "stsf electricity 24 24"        "ltsf weather_ltsf 96 336"
    "ltsf ettm1 96 336"             "ltsf ettm2 96 336"             "stsf solar 24 24"
    "ltsf weather_ltsf 96 192"      "ltsf ettm1 96 192"             "ltsf ettm2 96 192"
    "ltsf weather_ltsf 96 96"       "ltsf ettm1 96 96"              "ltsf ettm2 96 96"
    "ltsf etth1 96 720"             "ltsf etth2 96 720"             "ltsf exchange_ltsf 96 720"
    "ltsf etth1 96 336"             "ltsf etth2 96 336"             "ltsf exchange_ltsf 96 336"
    "ltsf etth1 96 192"             "ltsf etth2 96 192"             "ltsf exchange_ltsf 96 192"
    "ltsf etth1 96 96"              "ltsf etth2 96 96"              "ltsf exchange_ltsf 96 96"
    "stsf exchange 30 30"
    "ltsf illness_ltsf 96 60"       "ltsf illness_ltsf 96 48"       "ltsf illness_ltsf 96 36"
    "ltsf illness_ltsf 96 24"
)

job_cmd() {   # prints the python command for one job
    local kind=$1 name=$2 ctx=$3 pred=$4
    local extra=""
    [ "${kind}" == "ltsf" ] && extra="--dataset ${name} --max_epochs 50"
    echo python "${SCRIPT}" --probts_config "${CONFIG_BASE}/${kind}/${name}/dlinear_autoreg.yaml" \
        --gam_config "${GAM_CONFIG}" --seed "${SEED}" \
        --data_path "${DATA_DIR}" --root_dir "${LOG_DIR}" --split_val true \
        --context_length "${ctx}" --prediction_length "${pred}" ${extra}
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

worker() {    # worker w: pinned to its own cores, takes jobs until the queue is empty
    local w=$1
    local first=$(( FIRST_CPU + w * CORES_PER_JOB ))
    local last=$(( first + CORES_PER_JOB - 1 ))
    local i
    while true; do
        i=$(next_job)
        [ "${i}" -ge "${#JOBS[@]}" ] && break
        read -r kind name ctx pred <<< "${JOBS[$i]}"
        local tag="${name}_ctx${ctx}_pred${pred}"
        local log="${JOB_LOG_DIR}/${tag}.log"
        local cmd
        cmd=$(job_cmd "${kind}" "${name}" "${ctx}" "${pred}")
        if [ "${DRY_RUN}" == "1" ]; then
            echo "[worker ${w} | cpus ${first}-${last}] job ${i} ${tag}: ${cmd}"
            continue
        fi
        echo "[$(date '+%F %T')] worker ${w} (cpus ${first}-${last}) START ${tag}"
        if taskset -c "${first}-${last}" ${cmd} > "${log}" 2>&1; then
            echo "[$(date '+%F %T')] worker ${w} DONE  ${tag}"
        else
            echo "[$(date '+%F %T')] worker ${w} FAIL  ${tag} (exit $?, see ${log})"
            echo "${tag}" >> "${JOB_LOG_DIR}/failed.txt"
        fi
    done
}

echo "${#JOBS[@]} jobs, ${N_WORKERS} workers x ${CORES_PER_JOB} cores (cpus ${FIRST_CPU}-$(( FIRST_CPU + N_WORKERS * CORES_PER_JOB - 1 )))"
echo "GAM config: ${GAM_CONFIG}"
rm -f "${JOB_LOG_DIR}/failed.txt"
echo 0 > "${QUEUE}"

for (( w = 0; w < N_WORKERS && w < ${#JOBS[@]}; w++ )); do
    worker "${w}" &
done
wait

if [ -s "${JOB_LOG_DIR}/failed.txt" ]; then
    echo "Failed jobs:"; cat "${JOB_LOG_DIR}/failed.txt"
else
    echo "All jobs finished."
fi
rm -f "${QUEUE}" "${QUEUE}.lock"