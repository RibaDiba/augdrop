#!/bin/bash

#SBATCH --job-name=dropaug-eval
#SBATCH --nodes=1

## GAMMA, 1 GPU (Sensaug's test_robust.sbatch allocation). Its suite ran ~120
## conditions at ~2 min each on one L40S: ~4h on pascal_voc12, ~6h on ade20k.
## The run resumes, so a timeout costs only a resubmit.
#SBATCH --time=16:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --ntasks=4

# Absolute and flat; the body groups them into slurm/logs/<job-name>/.
#SBATCH --output=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.out
#SBATCH --error=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# eval.sbatch -- dropaug.eval's robustness suite on one checkpoint of a finished
# run. Replaces Sensaug's test_robust.sbatch / eval_all_augmentations.sbatch.
# train.sbatch queues it after every successful run.
#
#   sbatch slurm/eval.sbatch <exp_dir>
#
# One GPU, plain python: dropaug.eval has no DDP. It appends one record per
# condition to a JSONL and skips the ones already there, so resubmitting the same
# command after a timeout picks up where it stopped.
#
# Env knobs:
#   CKPT=best|last|<path>   best*.pth (default), the file last_checkpoint names, or a path
#   EVAL_OUT                default <exp_dir>/eval/<checkpoint stem>.jsonl; one file
#                           per checkpoint, as dropaug.eval.robustness.run requires
#   EVAL_ARGS="..."         extra eval flags, word-split
#   EVAL_PY=scripts/eval.py  CONDA_ENV=sensaug  CONDA_SH  REPO_DIR
#
# scripts/eval.py takes
#   --work-dir (the run, whose <work-dir>/dropaug.yaml it rebuilds) --checkpoint --out
#   [--dataset NAME for a shift target such as acdc or idd]
# so EVAL_ARGS="--dataset acdc" scores a domain-shift set. Needs hydra-core and
# omegaconf in CONDA_ENV (checked up front).
# Not ported: test.py's Cityscapes domain-shift sets (ACDC, IDD, Dark Zurich,
# Nighttime Driving).
# ==========================================================================

EXP_DIR="${1:?usage: sbatch slurm/eval.sbatch <exp_dir>}"
CKPT="${CKPT:-best}"
EVAL_PY="${EVAL_PY:-scripts/eval.py}"
REPO_DIR="${REPO_DIR:-/fs/gamma-projects/pruneaug}"

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-eval

[ -f "${EVAL_PY}" ] || { echo "ERROR: no ${EVAL_PY} under ${REPO_DIR}" >&2; exit 1; }
[ -d "${EXP_DIR}" ] || { echo "ERROR: EXP_DIR=${EXP_DIR} does not exist" >&2; exit 1; }
# Resolve here, where the message can say which run is missing a checkpoint.
CKPT_PATH="$(resolve_ckpt "${EXP_DIR}" "${CKPT}")" || exit 1
EVAL_OUT="${EVAL_OUT:-${EXP_DIR}/eval/$(basename "${CKPT_PATH}" .pth).jsonl}"
mkdir -p "$(dirname "${EVAL_OUT}")"

activate_env || exit 1
check_hydra || exit 1
export OMP_NUM_THREADS="${SLURM_NTASKS:-4}"

split_args extra "${EVAL_ARGS:-}"

echo "=== eval: ${EXP_DIR}"
echo "    ckpt: ${CKPT_PATH}"
echo "    out:  ${EVAL_OUT} ==="

python "${EVAL_PY}" \
    --work-dir "${EXP_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --out "${EVAL_OUT}" \
    "${extra[@]}"
