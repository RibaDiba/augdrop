#!/bin/bash

#SBATCH --job-name=dropaug-matrix
#SBATCH --nodes=1

## One array task = one train.sbatch run, so the same allocation.
#SBATCH --time=24:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --mem=120gb
#SBATCH --ntasks=16

## Scavenger / tron fallbacks: see slurm/train.sbatch.

# %A_%a: array job id and task index. The body groups them into
# slurm/logs/<job-name>/, so one submission's tasks share one directory.
#SBATCH --output=/gammascratch/amodak/dropaug/slurm/logs/%x.%A_%a.out
#SBATCH --error=/gammascratch/amodak/dropaug/slurm/logs/%x.%A_%a.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# matrix.sbatch -- an array job over backbone x host x dataset x selector x seed. Each task
# runs slurm/train.sbatch for its grid point (which then queues its own eval).
# Replaces Sensaug's run_all_datasets.sh / grad_corr_seed.sh /
# random_prune_all_datasets.sh drivers.
#
#   bash slurm/matrix.sbatch             list the grid and the command that submits it
#   bash slurm/matrix.sbatch --submit    submit it
#   sbatch --array=3,17 slurm/matrix.sbatch   rerun chosen points (same env as before)
#
# Grid (space-separated, env):
#   BACKBONES="segformer"        configs/task backbones (pspnet, convnext, ...)
#   HOSTS="sensaug uniform"      the DIFF32 hosts; the mmseg preprocessor can't train the others
#   DATASETS="cityscapes ade20k pascal_voc12 loveda"   the default; potsdam, synapse, a2i2haze
#                                and acdc also train (every configs/recipe pair is Sensaug's)
#   SELECTORS="none mrmr random_k kl_reweight qpfs dpp relevance_only"
#   SEEDS="0 1 2"
# Task index -> point: backbone varies slowest, then seed, then host, dataset, and
# selector fastest, so the first tasks out of a throttled array are one whole seed
# of the grid for the first backbone.
#
# Other env:
#   MAX_PARALLEL=8                --submit's array throttle (%K)
#   EXTRA_SBATCH_ARGS="..."       forwarded to sbatch by --submit
#   NAME=refactor1                --submit names the array refactor1-dropaug-matrix
#   everything train.sbatch reads (LAM, CLUSTER, DESCENDING_MA, INV_AUG, WORK_DIR, RUN_TAG,
#   TRAIN_ARGS, ALLOW_RESUME, EVAL, ...) passes through to every task
#
# --submit exports the grid it listed, so every task decodes its index against
# the same lists. Rerunning points by hand needs the same BACKBONES/HOSTS/DATASETS/
# SELECTORS/SEEDS in the environment as the original submission, or the indices mean other
# points; each task prints its point at the top of its log.
# ==========================================================================

HOSTS="${HOSTS:-sensaug uniform}"
DATASETS="${DATASETS:-cityscapes ade20k pascal_voc12 loveda}"
SELECTORS="${SELECTORS:-none mrmr random_k kl_reweight qpfs dpp relevance_only}"
SEEDS="${SEEDS:-0 1 2}"
BACKBONES="${BACKBONES:-segformer}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/dropaug}"

read -r -a H <<< "${HOSTS}"
read -r -a D <<< "${DATASETS}"
read -r -a S <<< "${SELECTORS}"
read -r -a N <<< "${SEEDS}"
read -r -a B <<< "${BACKBONES}"
for _pair in "HOSTS:${#H[@]}" "DATASETS:${#D[@]}" "SELECTORS:${#S[@]}" "SEEDS:${#N[@]}" "BACKBONES:${#B[@]}"; do
    if [ "${_pair#*:}" -eq 0 ]; then
        echo "ERROR: ${_pair%%:*} is empty." >&2
        exit 1
    fi
done
TOTAL=$(( ${#H[@]} * ${#D[@]} * ${#S[@]} * ${#N[@]} * ${#B[@]} ))

# Index -> "host dataset selector seed backbone".
point() {
    local i="$1" sel ds host
    sel="${S[i % ${#S[@]}]}";  i=$(( i / ${#S[@]} ))
    ds="${D[i % ${#D[@]}]}";   i=$(( i / ${#D[@]} ))
    host="${H[i % ${#H[@]}]}"; i=$(( i / ${#H[@]} ))
    printf '%s %s %s %s %s\n' "${host}" "${ds}" "${sel}" "${N[i % ${#N[@]}]}" "${B[i / ${#N[@]}]}"
}

# ------------------------------------------------ outside SLURM: list/submit ----
if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
    printf '%5s  %-14s %-16s %-14s %-16s %s\n' idx backbone host dataset selector seed
    for (( k = 0; k < TOTAL; k++ )); do
        read -r _h _d _s _n _b <<< "$(point "${k}")"
        printf '%5d  %-14s %-16s %-14s %-16s %s\n' "${k}" "${_b}" "${_h}" "${_d}" "${_s}" "${_n}"
    done
    ARRAY="0-$(( TOTAL - 1 ))%${MAX_PARALLEL:-8}"
    echo
    echo "${TOTAL} runs (${#H[@]} hosts x ${#D[@]} datasets x ${#S[@]} selectors x ${#N[@]} seeds x ${#B[@]} backbones)," \
         "each a 4-GPU job, plus one 1-GPU eval each."

    if [ "${1:-}" != "--submit" ]; then
        echo "submit with:  bash ${BASH_SOURCE[0]} --submit    (sbatch --array=${ARRAY})"
        exit 0
    fi
    EXTRA_SB=()
    [ -n "${EXTRA_SBATCH_ARGS:-}" ] && read -r -a EXTRA_SB <<< "${EXTRA_SBATCH_ARGS}"
    if [ -n "${NAME:-}" ]; then
        case "${NAME}" in
            *[!a-zA-Z0-9_-]*) echo "ERROR: NAME='${NAME}' must be [A-Za-z0-9_-]." >&2; exit 1 ;;
        esac
        EXTRA_SB+=(-J "${NAME}-dropaug-matrix")
    fi
    # Exported, so the tasks decode against these lists and not whatever the
    # defaults above are by the time they start.
    export HOSTS DATASETS SELECTORS SEEDS BACKBONES
    exec sbatch --array="${ARRAY}" "${EXTRA_SB[@]}" "${BASH_SOURCE[0]}"
fi

# ------------------------------------------------------- inside an array task ----
cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-matrix

if [ "${SLURM_ARRAY_TASK_ID}" -ge "${TOTAL}" ]; then
    echo "ERROR: task ${SLURM_ARRAY_TASK_ID} is outside the ${TOTAL}-point grid" \
         "(HOSTS='${HOSTS}' DATASETS='${DATASETS}' SELECTORS='${SELECTORS}' SEEDS='${SEEDS}' BACKBONES='${BACKBONES}')." >&2
    echo "       The array was submitted for a different grid." >&2
    exit 1
fi

read -r HOST DATASET SELECTOR SEED BACKBONE <<< "$(point "${SLURM_ARRAY_TASK_ID}")"
echo "=== matrix task ${SLURM_ARRAY_TASK_ID}/${TOTAL}: host=${HOST} dataset=${DATASET}" \
     "selector=${SELECTOR} seed=${SEED} backbone=${BACKBONE} ==="

# From the checkout, not next to this file: under SLURM that is the spooled copy.
export BACKBONE
exec bash "${REPO_DIR}/slurm/train.sbatch" "${HOST}" "${DATASET}" "${SELECTOR}" "${SEED}"
