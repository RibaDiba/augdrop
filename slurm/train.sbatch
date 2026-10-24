#!/bin/bash

#SBATCH --job-name=dropaug-train
#SBATCH --nodes=1

## GAMMA training config (Sensaug's train_nexus_gamma.sbatch allocation)
#SBATCH --time=24:00:00
#SBATCH --qos=huge-long
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:4
#SBATCH --mem=120gb
#SBATCH --ntasks=16

## Scavenger fallback -- swap the four gamma lines above for these when gamma is full.
##SBATCH --time=48:00:00
##SBATCH --qos=scavenger
##SBATCH --account=scavenger
##SBATCH --partition=scavenger

## Tron fallback (shorter walltime, explicit GPU model)
##SBATCH --time=23:59:00
##SBATCH --qos=high
##SBATCH --account=nexus
##SBATCH --partition=tron
##SBATCH --gres=gpu:rtxa4000:4
##SBATCH --mem=64gb
##SBATCH --ntasks=15

# Absolute and flat: SLURM opens these before the body runs, so neither a path
# relative to the submit dir nor a per-job subdirectory works. The body moves them
# into slurm/logs/<job-name>/ (see common.sh). slurm/logs/ must exist.
#SBATCH --output=/gammascratch/amodak/dropaug/slurm/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/dropaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# train.sbatch -- one dropaug training run on 4 GPUs, then its robustness eval
# queued as a separate 1-GPU job. Ported from Sensaug's train_nexus_gamma.sbatch.
#
#   sbatch slurm/train.sbatch [host] [dataset] [selector] [seed]
#
#   host      a configs/host name (default sensaug); train.py's config check rejects
#             a host the task's augmentation path can't apply
#   dataset   a configs/dataset name (default cityscapes)
#   selector  a selection.SELECTORS name (default mrmr), or `none`: the lambda=0
#             control, which probes and logs R but never changes the host's q
#   seed      integer (default 0); the run's training seed, and random_k's draw
#
# slurm/matrix.sbatch runs this once per grid point.
#
# Env knobs:
#   BACKBONE=segformer   picks the task (train.py --backbone): segformer, pspnet,
#                        convnext, deeplabv3plus, swin, mae, vit, ... (configs/task/)
#   LAM=0.25             forced to 0 for selector none; defaults to 1 for nms, which
#                        prunes and refuses any other value. Normalised like
#                        dropaug.config.run_name does (%g), so LAM=1.0 names the run -l1
#   CLUSTER=nexus        configs/cluster/<CLUSTER>.yaml: data and checkpoint paths
#   WORK_DIR=./experiments          the run goes in WORK_DIR/EXP_NAME
#   EXP_NAME                        default <host>_<selector>-l<lam>_<dataset>_<backbone>_s<seed>
#                                   (what dropaug.config.run_name gives); `none` has no -l
#   RUN_TAG                         suffix for a repeat of the same config
#   TRAIN_ARGS="..."                extra train.py flags and Hydra overrides, word-split,
#                                   e.g. "task.max_iters=400 sa.ratio=0.02"
#   DESCENDING_MA=1  INV_AUG=0      host sensaug only. Sensaug's train_nexus_gamma.sbatch
#                                   passed --descending-MA --no-inv-aug on every run, so they
#                                   are on here too: the most training mass goes on each op's
#                                   most damaging level (src/sa/pdf.py), and lighter_H /
#                                   darker_H stay out of the SA curve and the pdf.
#                                   DESCENDING_MA=0 / INV_AUG=1 turn them off. (Its
#                                   --auto-scale-lr is not needed: the builder enables
#                                   auto_scale_lr wherever the task has a base_batch_size, as
#                                   Sensaug's build_config did whatever the flag said.)
#   ALLOW_RESUME=1                  continue a run that already has a checkpoint
#                                   (passes --resume)
#   EVAL=0                          don't queue slurm/eval.sbatch afterwards
#   TRAIN_PY=train.py  CONDA_ENV=sensaug  CONDA_SH  REPO_DIR
#
# The launch is preceded by a single-process `train.py --dry-run` with the same
# arguments: it composes and checks the whole config (host, selector, dataset, task,
# vocab, ...) in seconds and writes <run>/dropaug.yaml; train.py exits 2 on a config
# error. The real launch then finds an identical dropaug.yaml, which train.py accepts.
# The train.py interface this relies on:
#   [--cluster NAME] [--backbone NAME] [--dataset NAME] [--host NAME] [--selector NAME]
#   [--lam F] [--seed N] [--work-dir DIR (the run directory itself)]
#   [--launcher none|pytorch] [--resume] [--descending-MA] [--no-inv-aug] [--dry-run]
#   [overrides ...]
# ==========================================================================

# `${1-default}`, not `${1:-default}`: an explicitly empty argument is then
# rejected below instead of silently becoming the default.
HOST="${1-sensaug}"
DATASET="${2-cityscapes}"
SELECTOR="${3-sensaug}"
SEED="${4-0}"
BACKBONE="${BACKBONE:-segformer}"
if [ "${SELECTOR}" = "nms" ]; then
    LAM="${LAM:-1}"
else
    LAM="${LAM:-0.25}"
fi

for _pair in "host:${HOST}" "dataset:${DATASET}" "selector:${SELECTOR}" "BACKBONE:${BACKBONE}"; do
    case "${_pair#*:}" in
        ''|*[!a-zA-Z0-9_]*)
            echo "ERROR: ${_pair%%:*}='${_pair#*:}' must be a non-empty [A-Za-z0-9_] name." >&2
            exit 1 ;;
    esac
done
case "${SEED}" in
    ''|*[!0-9]*) echo "ERROR: seed='${SEED}' must be a non-negative integer." >&2; exit 1 ;;
esac
if [ "${SELECTOR}" = "none" ]; then
    LAM=0
fi
if ! [[ "${LAM}" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: LAM='${LAM}' must be a non-negative decimal." >&2
    exit 1
fi
# The run name carries lam as python's `{float:g}`, so 1.0 -> 1 and 0.250 -> 0.25.
LAM="$(awk -v x="${LAM}" 'BEGIN { printf "%g", x }')"

# lambda is in the name: two runs of one selector at different strengths are
# different experiments and must not share a work dir (and resume each other).
if [ "${SELECTOR}" = "none" ]; then
    _arm="none"
else
    _arm="${SELECTOR}-l${LAM}"
fi
EXP_NAME="${EXP_NAME:-${HOST}_${_arm}_${DATASET}_${BACKBONE}_s${SEED}${RUN_TAG:+_${RUN_TAG}}}"

WORK_DIR="${WORK_DIR:-./experiments}"
CLUSTER="${CLUSTER:-nexus}"
TRAIN_PY="${TRAIN_PY:-train.py}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/dropaug}"

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs "${EXP_NAME}"

[ -f "${TRAIN_PY}" ] || { echo "ERROR: no ${TRAIN_PY} under ${REPO_DIR}" >&2; exit 1; }
[ -f "configs/cluster/${CLUSTER}.yaml" ] || {
    echo "ERROR: no cluster config configs/cluster/${CLUSTER}.yaml under ${REPO_DIR}" >&2; exit 1; }

mkdir -p "${WORK_DIR}"
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
EXP_DIR="${WORK_DIR}/${EXP_NAME}"

# mmengine resumes silently when work_dir/last_checkpoint exists. Refuse instead,
# so a resubmit never continues a previous run by accident.
if [ -z "${ALLOW_RESUME:-}" ] && [ -f "${EXP_DIR}/last_checkpoint" ]; then
    echo "ERROR: ${EXP_DIR} already holds a checkpoint; training would resume it." >&2
    echo "       Set RUN_TAG=... for a fresh run, or ALLOW_RESUME=1 to continue it." >&2
    exit 1
fi

activate_env || exit 1
command -v torchrun >/dev/null 2>&1 || {
    echo "ERROR: torchrun not on PATH after activating the env. PATH=${PATH}" >&2; exit 1; }

# Hydra and omegaconf are what dropaug.config composes with.
check_hydra || exit 1

split_args extra "${TRAIN_ARGS:-}"
TRAIN_FLAGS=(
    --cluster "${CLUSTER}"
    --backbone "${BACKBONE}"
    --dataset "${DATASET}"
    --host "${HOST}"
    --selector "${SELECTOR}"
    --lam "${LAM}"
    --seed "${SEED}"
    --work-dir "${EXP_DIR}"
)
if [ "${HOST}" = "sensaug" ]; then
    [ "${DESCENDING_MA:-1}" != "0" ] && TRAIN_FLAGS+=(--descending-MA)
    [ "${INV_AUG:-0}" = "0" ] && TRAIN_FLAGS+=(--no-inv-aug)
fi
# Continue only a run that has something to continue; ALLOW_RESUME on a fresh
# directory is just a fresh start.
if [ -n "${ALLOW_RESUME:-}" ] && [ -f "${EXP_DIR}/last_checkpoint" ]; then
    TRAIN_FLAGS+=(--resume)
fi

# Check the whole composed config once, single-process. Under torchrun a bad name
# would fail on every rank, after the ranks had already spun up. This writes
# ${EXP_DIR}/dropaug.yaml; the launch below composes the same one.
echo "=== preflight: ${TRAIN_PY} --dry-run ==="
python "${TRAIN_PY}" "${TRAIN_FLAGS[@]}" --dry-run "${extra[@]}" || {
    echo "ERROR: ${TRAIN_PY} --dry-run rejected the config (exit $?); not launching." >&2; exit 1; }

# Nexus compute nodes have internet access, so pretrained backbones need no staging.
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
if [ "${NUM_GPUS}" -eq 0 ]; then
    NUM_GPUS="${SLURM_GPUS_ON_NODE:-0}"
fi
[ "${NUM_GPUS}" -gt 0 ] || { echo "ERROR: no GPUs visible" >&2; exit 1; }

# The port comes from the job id, so two jobs on one node don't both take 29500.
TORCHRUN_ARGS=(
    --nnodes="${NNODES:-1}"
    --node_rank="${NODE_RANK:-0}"
    --master_addr="${MASTER_ADDR:-127.0.0.1}"
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${PORT:-$((29500 + ${SLURM_JOB_ID:-0} % 1000))}"
)
export OMP_NUM_THREADS=16

echo "=== train: host=${HOST} dataset=${DATASET} selector=${SELECTOR} lam=${LAM}" \
     "seed=${SEED} backbone=${BACKBONE} gpus=${NUM_GPUS} -> ${EXP_DIR} ==="

torchrun "${TORCHRUN_ARGS[@]}" \
    "${TRAIN_PY}" \
    "${TRAIN_FLAGS[@]}" \
    --launcher pytorch \
    "${extra[@]}"

TRAIN_STATUS=$?
if [ "${TRAIN_STATUS}" -ne 0 ]; then
    echo "ERROR: training exited ${TRAIN_STATUS}; not queueing the eval." >&2
    exit "${TRAIN_STATUS}"
fi

# The eval is its own 1-GPU job rather than a step here: dropaug.eval has no DDP,
# so running it inside this allocation would idle three A6000s for hours.
if [ "${EVAL:-1}" != "0" ]; then
    sbatch -J "eval-${EXP_NAME}" "${REPO_DIR}/slurm/eval.sbatch" "${EXP_DIR}" \
        || echo "WARNING: could not queue slurm/eval.sbatch for ${EXP_DIR}; submit it by hand." >&2
fi

echo "All done: ${EXP_DIR}"
