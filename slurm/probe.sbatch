#!/bin/bash

#SBATCH --job-name=dropaug-probe
#SBATCH --nodes=1

## GAMMA, 1 GPU. The walltime is a guess: one forward+backward per op per image,
## two forwards per op for the finite-difference ones, over the whole val set.
#SBATCH --time=08:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:1
#SBATCH --mem=32gb
#SBATCH --ntasks=4

# Absolute and flat; the body groups them into slurm/logs/<job-name>/.
#SBATCH --output=/gammascratch/amodak/dropaug/slurm/logs/%x.%j.out
#SBATCH --error=/gammascratch/amodak/dropaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# probe.sbatch -- the probe, post hoc, on one checkpoint of a finished run: one
# frozen sweep -> G -> R and its logs. Replaces Sensaug's compute_grad_corr.sbatch
# + scripts/run_grad_corr.sh.
#
#   sbatch slurm/probe.sbatch <exp_dir>
#
# One GPU, plain python: the probe's DDP sharding is not ported.
#
# Env knobs:
#   CKPT=best|last|<path>   best*.pth (default), the file last_checkpoint names, or a path
#   PROBE_OUT               default <exp_dir>/probe/<checkpoint stem>[_<PROBE_TAG>]
#   PROBE_TAG               names a second probe of the same checkpoint (another
#                           magnitude, probe set, ...)
#   PROBE_ARGS="..."        probe flags, word-split, e.g.
#                           "--magnitude 0.5 --n-probe 500 --probe-seed 0 --batch-size 1"
#   ALLOW_EXISTING=1        write into a PROBE_OUT that already holds probe logs
#   PROBE_PY=scripts/probe.py  CONDA_ENV=sensaug  CONDA_SH  REPO_DIR
#
# scripts/probe.py takes
#   --work-dir (the run, whose <work-dir>/dropaug.yaml it rebuilds) --checkpoint
#   --output-dir [--magnitude --n-probe --probe-seed --batch-size]
# and writes aug_gradient_log.txt / corr_*_log.* into --output-dir, not into the run
# directory, where the in-training probe's logs already are. Needs hydra-core and
# omegaconf in CONDA_ENV (checked up front).
#
# Not ported from run_grad_corr.sh: the KID magnitude calibration
# (calibrate_kid_magnitudes.py is not in dropaug) and the SA-snapshot magnitude modes.
# ==========================================================================

EXP_DIR="${1:?usage: sbatch slurm/probe.sbatch <exp_dir>}"
CKPT="${CKPT:-best}"
PROBE_PY="${PROBE_PY:-scripts/probe.py}"
REPO_DIR="${REPO_DIR:-/gammascratch/amodak/dropaug}"

cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-probe

[ -f "${PROBE_PY}" ] || { echo "ERROR: no ${PROBE_PY} under ${REPO_DIR}" >&2; exit 1; }
[ -d "${EXP_DIR}" ] || { echo "ERROR: EXP_DIR=${EXP_DIR} does not exist" >&2; exit 1; }
CKPT_PATH="$(resolve_ckpt "${EXP_DIR}" "${CKPT}")" || exit 1
PROBE_OUT="${PROBE_OUT:-${EXP_DIR}/probe/$(basename "${CKPT_PATH}" .pth)${PROBE_TAG:+_${PROBE_TAG}}}"

# The gradient log is appended to. A second sweep in the same directory would put
# two Gs in one file, and R recomputed from it would pool them.
if [ -z "${ALLOW_EXISTING:-}" ] \
   && compgen -G "${PROBE_OUT}/aug_gradient_log.txt" >/dev/null; then
    echo "ERROR: ${PROBE_OUT} already holds a probe; set PROBE_TAG=... for another," >&2
    echo "       or ALLOW_EXISTING=1 to write into it anyway." >&2
    exit 1
fi
mkdir -p "${PROBE_OUT}"

activate_env || exit 1
check_hydra || exit 1
export OMP_NUM_THREADS="${SLURM_NTASKS:-4}"

split_args extra "${PROBE_ARGS:-}"

echo "=== probe: ${EXP_DIR}"
echo "    ckpt: ${CKPT_PATH}"
echo "    out:  ${PROBE_OUT}"
echo "    args: ${extra[*]:-(defaults)} ==="

python "${PROBE_PY}" \
    --work-dir "${EXP_DIR}" \
    --checkpoint "${CKPT_PATH}" \
    --output-dir "${PROBE_OUT}" \
    "${extra[@]}"
