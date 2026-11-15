#!/usr/bin/env bash
# Experiment 6: GAM (ec), autoregressive, no teacher forcing, on all dysts systems
# (train_multivariate__pts_per_period_100__periods_12.json.gz, 133 systems), in parallel on CPU.
# One run of src/run_gams_on_dysts.py per system.
#
# Usage:
#   ./run_gams_dysts_all_systems.sh [gam_config.yaml] [extra args for run_gams_on_dysts.py]
#   e.g. ./run_gams_dysts_all_systems.sh src/configs/gams/gams_autoreg_dysts.yaml --predict_increment
#        ./run_gams_dysts_all_systems.sh --max_epochs 500     (no config given: default config)
#
# Environment variables (optional):
#   TOTAL_CPUS=40      CPUs considered (FIRST_CPU .. FIRST_CPU+TOTAL_CPUS-1)
#   CORES_PER_JOB=1    cores per run
#   FIRST_CPU=0        first CPU id considered
#   BUSY_PCT=30        a CPU whose use by other processes is above this (%) is skipped
#   SAMPLE_SECONDS=5   length of the CPU-use measurement
#   RECHECK=1          before each job, re-measure the worker's CPUs; if busy, wait
#   RECHECK_WAIT=60    seconds between re-measurements of a busy worker
#   MAX_WAIT=3600      a worker whose CPUs stay busy this long stops (its jobs go to others);
#                      systems never started are listed in job_logs/not_run.txt
#                      (rerun them with SYSTEMS="$(cat <ROOT_DIR>/job_logs/not_run.txt)")
#   SYSTEMS="Lorenz Rossler"   subset of systems (default: all systems in the train file)
#   ROOT_DIR=dysts_data/results/gams
#   SEED=0
#   DRY_RUN=1          only print the CPU measurement, the commands and the worker/core assignment
#
# CPU selection: at the start, the use of every considered CPU is measured over SAMPLE_SECONDS
# (from /proc/stat). CPUs above BUSY_PCT are left out; the free CPUs are split into workers of
# CORES_PER_JOB CPUs each (number of workers = free CPUs / CORES_PER_JOB). With RECHECK=1 a
# worker measures its own CPUs again before taking each job (between jobs, so its own run is
# not counted) and waits while they are busy; after MAX_WAIT seconds it stops and the other
# workers take the remaining jobs.
# Note: taskset only keeps OUR runs on their CPUs. Other processes that are not pinned can still
# be moved onto these CPUs by the kernel; the measurement only avoids CPUs busy at that time.
#
# Scheduling: systems in alphabetical order (mixes small and large D, so the large-D
# systems, which need the most memory, do not all run at the same time);
# one shared queue; each worker is pinned to its own CPUs (taskset) and, when free,
# takes the next system (counter file + flock). Thread pools (XLA, BLAS) are limited
# to CORES_PER_JOB.
#
# Output: <ROOT_DIR>/<system>/{result.json, summary.json, gam_state.pkl}
#         <ROOT_DIR>/job_logs/<system>.log            (stdout + stderr of each run)
#         <ROOT_DIR>/results_<DATANAME>_GAMAutoReg.json  all result.json merged:
#             {system: {"GAMAutoReg": {...}}}, the format of the N-BEATS results file

THIS_REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# first argument = config file only if it is not an option (--...); GAM_CONFIG env var also works
GAM_CONFIG="${GAM_CONFIG:-${THIS_REPO}/src/configs/gams/gams_autoreg_dysts.yaml}"
if [ $# -gt 0 ] && [[ "$1" != --* ]]; then
    GAM_CONFIG="$1"
    shift
fi
EXTRA_ARGS=("$@")
[ -f "${GAM_CONFIG}" ] || { echo "GAM config not found: ${GAM_CONFIG}"; exit 1; }

TOTAL_CPUS="${TOTAL_CPUS:-40}"
CORES_PER_JOB="${CORES_PER_JOB:-1}"
FIRST_CPU="${FIRST_CPU:-0}"
BUSY_PCT="${BUSY_PCT:-30}"
SAMPLE_SECONDS="${SAMPLE_SECONDS:-5}"
RECHECK="${RECHECK:-1}"
RECHECK_WAIT="${RECHECK_WAIT:-60}"
MAX_WAIT="${MAX_WAIT:-3600}"
DRY_RUN="${DRY_RUN:-0}"
SEED="${SEED:-0}"

DATANAME="multivariate__pts_per_period_100__periods_12"
DATA_DIR="${THIS_REPO}/dysts_data/dysts_data/data"
TRAIN_FILE="${DATA_DIR}/train_${DATANAME}.json.gz"
ROOT_DIR="${ROOT_DIR:-${THIS_REPO}/dysts_data/results/gams}"
JOB_LOG_DIR="${ROOT_DIR}/job_logs"
SCRIPT="${THIS_REPO}/src/run_gams_on_dysts.py"
MERGED="${ROOT_DIR}/results_${DATANAME}_GAMAutoReg.json"

SYSTEMS="IkedaDelay MacArthur MackeyGlass PiecewiseCircuit ScrollDelay SprottB SprottC SprottD SprottDelay SprottE SprottF SwingingAtwood VossDelay"

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
# CPU use from /proc/stat: prints "<cpu id> <% busy>" per CPU over <seconds>
# busy = 1 - (idle + iowait) / total, from the difference of two readings
# ---------------------------------------------------------------------------
cpu_usage() {
    local a b
    a=$(grep '^cpu[0-9]' /proc/stat)
    sleep "$1"
    b=$(grep '^cpu[0-9]' /proc/stat)
    paste -d' ' <(echo "${a}") <(echo "${b}") | awk '{
        n = NF / 2; ta = 0; tb = 0
        for (i = 2; i <= n; i++) ta += $i
        for (i = n + 2; i <= NF; i++) tb += $i
        ia = $5 + $6; ib = $(n + 5) + $(n + 6); d = tb - ta
        p = (d > 0) ? 100 * (1 - (ib - ia) / d) : 0
        sub("cpu", "", $1); printf "%s %.0f\n", $1, p }'
}

cpus_busy() {  # cpus_busy "3,7" -> exit 0 if any of these CPUs is above BUSY_PCT
    local list=",$1,"
    cpu_usage "${SAMPLE_SECONDS}" | awk -v list="${list}" -v lim="${BUSY_PCT}" \
        'index(list, "," $1 ",") && $2 > lim { busy = 1 } END { exit busy ? 0 : 1 }'
}

# ---------------------------------------------------------------------------
# Free CPUs at the start -> worker CPU groups
# ---------------------------------------------------------------------------
echo "Measuring CPU use over ${SAMPLE_SECONDS}s (busy above ${BUSY_PCT}%) ..."
FREE_CPUS=()
BUSY_LIST=()
while read -r cpu pct; do
    (( cpu < FIRST_CPU || cpu >= FIRST_CPU + TOTAL_CPUS )) && continue
    if (( pct > BUSY_PCT )); then BUSY_LIST+=("${cpu}:${pct}%"); else FREE_CPUS+=("${cpu}"); fi
done < <(cpu_usage "${SAMPLE_SECONDS}")
echo "busy CPUs (skipped): ${BUSY_LIST[*]:-none}"
echo "free CPUs: ${#FREE_CPUS[@]}"

N_WORKERS=$(( ${#FREE_CPUS[@]} / CORES_PER_JOB ))
[ "${N_WORKERS}" -eq 0 ] && { echo "No free CPUs (need ${CORES_PER_JOB} per job)"; exit 1; }
WORKER_CPUS=()               # WORKER_CPUS[w] = "c1,c2,..." (CORES_PER_JOB CPUs)
for (( w = 0; w < N_WORKERS; w++ )); do
    group=("${FREE_CPUS[@]:$(( w * CORES_PER_JOB )):${CORES_PER_JOB}}")
    WORKER_CPUS+=("$(IFS=,; echo "${group[*]}")")
done

# ---------------------------------------------------------------------------
# Job list: systems in alphabetical order (SYSTEMS overrides)
# ---------------------------------------------------------------------------
if [ -n "${SYSTEMS:-}" ]; then
    read -r -a JOBS <<< "${SYSTEMS}"
else
    mapfile -t JOBS < <(python - "${TRAIN_FILE}" <<'EOF'
import gzip, json, sys
with gzip.open(sys.argv[1], "rt") as f:
    d = json.load(f)
for k in sorted(d):
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

wait_until_free() {  # worker w: wait while its CPUs are busy; exit 1 after MAX_WAIT seconds
    local w=$1 cpus=$2 waited=0
    [ "${RECHECK}" == "1" ] || return 0
    while cpus_busy "${cpus}"; do
        if (( waited >= MAX_WAIT )); then
            echo "[$(date '+%F %T')] worker ${w} (cpus ${cpus}) STOP: CPUs busy for ${waited}s"
            return 1
        fi
        echo "[$(date '+%F %T')] worker ${w} (cpus ${cpus}) WAIT: CPUs busy, recheck in ${RECHECK_WAIT}s"
        sleep "${RECHECK_WAIT}"
        waited=$(( waited + RECHECK_WAIT + SAMPLE_SECONDS ))
    done
    return 0
}

worker() {    # worker w: pinned to its own CPUs, takes systems until the queue is empty
    local w=$1
    local cpus="${WORKER_CPUS[$w]}"
    local i system log cmd
    while true; do
        if [ "${DRY_RUN}" != "1" ]; then
            wait_until_free "${w}" "${cpus}" || break        # before taking a job: none is lost
        fi
        i=$(next_job)
        [ "${i}" -ge "${#JOBS[@]}" ] && break
        system="${JOBS[$i]}"
        log="${JOB_LOG_DIR}/${system}.log"
        cmd=$(job_cmd "${system}")
        if [ "${DRY_RUN}" == "1" ]; then
            echo "[worker ${w} | cpus ${cpus}] job ${i} ${system}: ${cmd}"
            continue
        fi
        echo "[$(date '+%F %T')] worker ${w} (cpus ${cpus}) START ${system}"
        echo "${system}" >> "${JOB_LOG_DIR}/started.txt"
        taskset -c "${cpus}" ${cmd} > "${log}" 2>&1
        local rc=$?
        if [ "${rc}" -eq 0 ]; then
            echo "[$(date '+%F %T')] worker ${w} DONE  ${system}"
        else
            echo "[$(date '+%F %T')] worker ${w} FAIL  ${system} (exit ${rc}, see ${log})"
            echo "${system} ${rc}" >> "${JOB_LOG_DIR}/failed.txt"
        fi
    done
}

echo "${#JOBS[@]} systems, ${N_WORKERS} workers x ${CORES_PER_JOB} cores"
for (( w = 0; w < N_WORKERS; w++ )); do echo "  worker ${w}: cpus ${WORKER_CPUS[$w]}"; done
echo "GAM config: ${GAM_CONFIG}  extra args: ${EXTRA_ARGS[*]:-none}"
rm -f "${JOB_LOG_DIR}/failed.txt" "${JOB_LOG_DIR}/started.txt"
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

NOT_RUN=()
for s in "${JOBS[@]}"; do
    grep -qxF "${s}" "${JOB_LOG_DIR}/started.txt" 2>/dev/null || NOT_RUN+=("${s}")
done
if [ "${#NOT_RUN[@]}" -gt 0 ]; then
    echo "Not run (all workers stopped on busy CPUs): ${NOT_RUN[*]}"
    printf '%s\n' "${NOT_RUN[@]}" > "${JOB_LOG_DIR}/not_run.txt"
fi
if [ -s "${JOB_LOG_DIR}/failed.txt" ]; then
    echo "Failed systems (name, exit code):"; cat "${JOB_LOG_DIR}/failed.txt"
elif [ "${#NOT_RUN[@]}" -eq 0 ]; then
    echo "All systems finished."
fi