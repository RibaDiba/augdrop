#!/bin/bash

#SBATCH --job-name=dropaug-gaa-search
#SBATCH --nodes=1

## GAMMA, 4 GPUs, one child network on each at a time (a job under the cap, slurm/gate.sh:
## submit with slurm/submit.sh; GPU=a5000 makes it one of the four A5000 jobs). The
## walltime is the 24 h of faa_search: at the default budget (host.search, 2 levels of
## 250 trials) the user's estimate is about 5 h on 4 A5000s. Time one child first
## (SEARCH_ARGS="--time-trial"). A search stopped at the walltime resumes on resubmit.
#SBATCH --time=24:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --mem=96gb
#SBATCH --ntasks=16

# Absolute and flat; the body groups them into slurm/logs/<job-name>/.
#SBATCH --output=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.out
#SBATCH --error=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# gaa_search.sbatch -- Greedy AutoAugment's policy search on one dataset, before the
# greedy_autoaugment training runs on it (scripts/gaa_search.py).
#
#   [GPU=a5000] bash slurm/submit.sh <name> slurm/gaa_search.sbatch [dataset]
#
#   dataset   a configs/dataset name (default cityscapes)
#
# Writes <WORK_DIR>/gaa_search_<dataset>_<BACKBONE>[_prune-...][_<SEARCH_TAG>]/final.json
# (and level_<c>.json after each level), which train.sbatch's greedy_autoaugment runs on
# that dataset and backbone read by default (its GAA_POLICY; a pruned search's policy is
# passed with GAA_POLICY=<dir>/final.json). Resubmitting continues a stopped search; one
# with a final.json is refused.
#
# Env knobs:
#   CLUSTER=nexus        configs/cluster/<CLUSTER>.yaml: data and checkpoint paths
#   WORK_DIR=./experiments
#   BACKBONE=segformer   the child's task (gaa_search.py --backbone): the run's own
#   SEARCH_TAG           suffix for a second search of the same setup
#   PRUNE_OPS="A B"      the search leaves these ops out (--prune-ops): the search-then-train
#                        recipe's NMS prune P (scripts/nms_prune.py's PRUNE_OPS= line on the
#                        trunk's 25% probe)
#   PRUNE_RANDOM=K  PRUNE_SEED=S
#                        K more left out, drawn from the ops the probe gives a row
#                        (--random-prune / --random-prune-seed): the random-K null. The
#                        output dir gets the training run name's prune suffix
#   SEARCH_ARGS="..."    extra gaa_search.py flags and Hydra overrides, word-split,
#                        e.g. "--time-trial" or "host.search.levels=1"
#   CONDA_ENV  REPO_DIR
# ==========================================================================

DATASET="${1-cityscapes}"
case "${DATASET}" in
    ''|*[!a-zA-Z0-9_]*) echo "ERROR: dataset='${DATASET}' must be a non-empty [A-Za-z0-9_] name." >&2; exit 1 ;;
esac
CLUSTER="${CLUSTER:-nexus}"
WORK_DIR="${WORK_DIR:-./experiments}"
REPO_DIR="${REPO_DIR:-/fs/gamma-projects/pruneaug}"
BACKBONE="${BACKBONE:-segformer}"
case "${BACKBONE}" in
    *[!a-zA-Z0-9_]*) echo "ERROR: BACKBONE='${BACKBONE}' must be a [A-Za-z0-9_] name." >&2; exit 1 ;;
esac

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-gaa-search
gate_check_running || exit 1

[ -f "configs/cluster/${CLUSTER}.yaml" ] || {
    echo "ERROR: no cluster config configs/cluster/${CLUSTER}.yaml under ${REPO_DIR}" >&2; exit 1; }
mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"

read -r -a PRUNE_OP_LIST <<< "${PRUNE_OPS:-}"
for _op in "${PRUNE_OP_LIST[@]}"; do
    case "${_op}" in
        *[!a-zA-Z0-9_.]*) echo "ERROR: PRUNE_OPS entry '${_op}' must be a [A-Za-z0-9_.] op name." >&2; exit 1 ;;
    esac
done
for _pair in "PRUNE_RANDOM:${PRUNE_RANDOM:-0}" "PRUNE_SEED:${PRUNE_SEED:-0}"; do
    if ! [[ "${_pair#*:}" =~ ^(0|[1-9][0-9]*)$ ]]; then
        echo "ERROR: ${_pair%%:*}='${_pair#*:}' must be a non-negative integer without leading zeros." >&2
        exit 1
    fi
done
SEARCH_FLAGS=(--dataset "${DATASET}" --backbone "${BACKBONE}" --cluster "${CLUSTER}" --work-root "${WORK_DIR}")
[ -n "${SEARCH_TAG:-}" ] && SEARCH_FLAGS+=(--search-tag "${SEARCH_TAG}")
[ "${#PRUNE_OP_LIST[@]}" -gt 0 ] && SEARCH_FLAGS+=(--prune-ops "${PRUNE_OP_LIST[*]}")
if [ "${PRUNE_RANDOM:-0}" != "0" ]; then
    SEARCH_FLAGS+=(--random-prune "${PRUNE_RANDOM}")
    [ -n "${PRUNE_SEED:-}" ] && SEARCH_FLAGS+=(--random-prune-seed "${PRUNE_SEED}")
elif [ -n "${PRUNE_SEED:-}" ]; then
    echo "ERROR: PRUNE_SEED is for PRUNE_RANDOM." >&2; exit 1
fi

activate_env || exit 1
check_hydra || exit 1
# Each child's DataLoader workers (host.search.num_workers) share the job's CPUs.
export OMP_NUM_THREADS=2

split_args extra "${SEARCH_ARGS:-}"

# The output dir is the script's (its prune suffix is dropaug.config.prune_tag's): a dry
# run composes the config and prints it, and refuses a dir that holds a finished search.
OUT_DIR="$(python scripts/gaa_search.py "${SEARCH_FLAGS[@]}" --dry-run "${extra[@]}" | sed -n 's/^OUT_DIR=//p')"
[ -n "${OUT_DIR}" ] || { echo "ERROR: gaa_search.py --dry-run failed (above)." >&2; exit 1; }

echo "=== gaa_search: ${DATASET} (${BACKBONE}) -> ${OUT_DIR}"
echo "    args: ${SEARCH_FLAGS[*]} ${extra[*]:-} ==="

python scripts/gaa_search.py "${SEARCH_FLAGS[@]}" "${extra[@]}"
