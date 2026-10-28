#!/usr/bin/env bash
# submit.sh -- sbatch with a name put in front of the script's job name, and the job cap.
#
#   bash slurm/submit.sh <name> <script> [args...]
#   GPU=a5000 bash slurm/submit.sh <name> <script> [args...]
#
#   bash slurm/submit.sh refactor1 slurm/train.sbatch sensaug cityscapes mrmr 0
#       -> job name `refactor1-dropaug-train`
#
# THE JOB CAP (slurm/gate.sh): at most ONE A6000 job or TWO A5000 jobs in flight, each
# with the GPU count its script's header gives (train 4, probe 1). GPU=a6000 (default) or
# a5000 picks the card; a job submitted past the cap waits for the ones ahead of it.
# eval.sbatch is exempt and goes through as is.
#
# The job name is read by sbatch before the script body runs (it fixes %x in the log
# paths too), so a script argument can't set it; this reads the script's own
# `#SBATCH --job-name` and passes `-J <name>-<that>`. Logs then group under
# slurm/logs/<name>-<that>/ (common.sh). Extra sbatch flags: SBATCH_ARGS="..." (not
# --gres, -G, --gpus*, --nodes, --dependency, --array: the cap owns those).
# For the matrix, use NAME=<name> bash slurm/matrix.sbatch --submit.
set -eu

[ "$#" -ge 2 ] || { sed -n '2,19p' "${BASH_SOURCE[0]}" >&2; exit 2; }
NAME="$1"; SCRIPT="$2"; shift 2
case "${NAME}" in
    ''|*[!a-zA-Z0-9_-]*) echo "ERROR: name='${NAME}' must be a non-empty [A-Za-z0-9_-] string." >&2; exit 1 ;;
esac
[ -f "${SCRIPT}" ] || { echo "ERROR: no script ${SCRIPT}" >&2; exit 1; }

BASE="$(sed -n 's/^#SBATCH --job-name=//p' "${SCRIPT}" | head -n 1)"
[ -n "${BASE}" ] || { echo "ERROR: ${SCRIPT} has no #SBATCH --job-name." >&2; exit 1; }

EXTRA=()
[ -n "${SBATCH_ARGS:-}" ] && read -r -a EXTRA <<< "${SBATCH_ARGS}"

case "$(basename "${SCRIPT}")" in
    eval.sbatch) exec sbatch -J "${NAME}-${BASE}" "${EXTRA[@]}" "${SCRIPT}" "$@" ;;
    matrix.sbatch)
        echo "ERROR: submit the matrix with  NAME=${NAME} bash slurm/matrix.sbatch --submit" >&2
        exit 1 ;;
esac

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/gate.sh"
gate_sbatch "${GPU:-a6000}" single -J "${NAME}-${BASE}" "${EXTRA[@]}" -- "${SCRIPT}" "$@"
