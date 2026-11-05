#!/bin/bash

#SBATCH --job-name=dropaug-faa-search
#SBATCH --nodes=1

## GAMMA, 1 GPU (a job under the cap, slurm/gate.sh: submit with slurm/submit.sh). The
## walltime is a guess: 20 epochs over up to 4000 crops, three segmentor passes a step,
## the policy's 26 ops x 2 stages on half of each batch.
#SBATCH --time=24:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=48gb
#SBATCH --ntasks=6

# Absolute and flat; the body groups them into slurm/logs/<job-name>/.
#SBATCH --output=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.out
#SBATCH --error=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# faa_search.sbatch -- Faster AutoAugment's policy search on one dataset, before any
# faster_autoaugment training run on it (scripts/faa_search.py).
#
#   bash slurm/submit.sh <name> slurm/faa_search.sbatch [dataset]
#
#   dataset   a configs/dataset name (default cityscapes)
#
# Writes <WORK_DIR>/faa_search_<dataset>[_prune-...][_<SEARCH_TAG>]/final.npz, which
# train.sbatch's faster_autoaugment runs on that dataset read by default (its FAA_POLICY;
# a pruned search's policy is passed with FAA_POLICY=<dir>/final.npz).
#
# Env knobs:
#   CLUSTER=nexus        configs/cluster/<CLUSTER>.yaml: data and checkpoint paths
#   WORK_DIR=./experiments
#   SEARCH_TAG           suffix for a second search of the same dataset
#   PRUNE_OPS="A B"      the search leaves these ops out (--prune-ops): the search-then-train
#                        recipe's NMS prune P (scripts/nms_prune.py's PRUNE_OPS= line on the
#                        trunk's 25% probe)
#   PRUNE_RANDOM=K  PRUNE_SEED=S
#                        K more left out, drawn from the ops the probe gives a row
#                        (--random-prune / --random-prune-seed): the random-K null. The
#                        output dir gets the training run name's prune suffix
#   SEARCH_ARGS="..."    extra faa_search.py flags and Hydra overrides, word-split,
#                        e.g. "--task mmseg_pspnet host.search.epochs=10"
#   CONDA_ENV  REPO_DIR
# ==========================================================================

DATASET="${1-cityscapes}"
case "${DATASET}" in
    ''|*[!a-zA-Z0-9_]*) echo "ERROR: dataset='${DATASET}' must be a non-empty [A-Za-z0-9_] name." >&2; exit 1 ;;
esac
CLUSTER="${CLUSTER:-nexus}"
WORK_DIR="${WORK_DIR:-./experiments}"
REPO_DIR="${REPO_DIR:-/fs/gamma-projects/pruneaug}"

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-faa-search
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
SEARCH_FLAGS=(--dataset "${DATASET}" --cluster "${CLUSTER}" --work-root "${WORK_DIR}")
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
export OMP_NUM_THREADS="${SLURM_NTASKS:-6}"

split_args extra "${SEARCH_ARGS:-}"

# The output dir is the script's (its prune suffix is dropaug.config.prune_tag's): a dry
# run composes the config and prints it, and refuses a dir that holds a finished search.
OUT_DIR="$(python scripts/faa_search.py "${SEARCH_FLAGS[@]}" --dry-run "${extra[@]}" | sed -n 's/^OUT_DIR=//p')"
[ -n "${OUT_DIR}" ] || { echo "ERROR: faa_search.py --dry-run failed (above)." >&2; exit 1; }

echo "=== faa_search: ${DATASET} -> ${OUT_DIR}"
echo "    args: ${SEARCH_FLAGS[*]} ${extra[*]:-} ==="

python scripts/faa_search.py "${SEARCH_FLAGS[@]}" "${extra[@]}"
