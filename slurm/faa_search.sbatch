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
# Writes <WORK_DIR>/faa_search_<dataset>[_<SEARCH_TAG>]/final.npz, which train.sbatch's
# faster_autoaugment runs on that dataset read by default (its FAA_POLICY).
#
# Env knobs:
#   CLUSTER=nexus        configs/cluster/<CLUSTER>.yaml: data and checkpoint paths
#   WORK_DIR=./experiments
#   SEARCH_TAG           suffix for a second search of the same dataset
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
OUT_DIR="${WORK_DIR}/faa_search_${DATASET}${SEARCH_TAG:+_${SEARCH_TAG}}"

activate_env || exit 1
check_hydra || exit 1
export OMP_NUM_THREADS="${SLURM_NTASKS:-6}"

split_args extra "${SEARCH_ARGS:-}"

echo "=== faa_search: ${DATASET} -> ${OUT_DIR}"
echo "    args: ${extra[*]:-(defaults)} ==="

python scripts/faa_search.py \
    --dataset "${DATASET}" \
    --cluster "${CLUSTER}" \
    --output-dir "${OUT_DIR}" \
    "${extra[@]}"
