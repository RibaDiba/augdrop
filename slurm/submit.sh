#!/usr/bin/env bash
# submit.sh -- sbatch with a name put in front of the script's job name.
#
#   bash slurm/submit.sh <name> <script> [args...]
#
#   bash slurm/submit.sh refactor1 slurm/train.sbatch sensaug cityscapes mrmr 0
#       -> job name `refactor1-dropaug-train`
#
# The job name is read by sbatch before the script body runs (it fixes %x in the log
# paths too), so a script argument can't set it; this reads the script's own
# `#SBATCH --job-name` and passes `-J <name>-<that>`. Logs then group under
# slurm/logs/<name>-<that>/ (common.sh). Extra sbatch flags: SBATCH_ARGS="...".
# For the matrix, use NAME=<name> bash slurm/matrix.sbatch --submit.
set -eu

[ "$#" -ge 2 ] || { sed -n '2,12p' "${BASH_SOURCE[0]}" >&2; exit 2; }
NAME="$1"; SCRIPT="$2"; shift 2
case "${NAME}" in
    ''|*[!a-zA-Z0-9_-]*) echo "ERROR: name='${NAME}' must be a non-empty [A-Za-z0-9_-] string." >&2; exit 1 ;;
esac
[ -f "${SCRIPT}" ] || { echo "ERROR: no script ${SCRIPT}" >&2; exit 1; }

BASE="$(sed -n 's/^#SBATCH --job-name=//p' "${SCRIPT}" | head -n 1)"
[ -n "${BASE}" ] || { echo "ERROR: ${SCRIPT} has no #SBATCH --job-name." >&2; exit 1; }

EXTRA=()
[ -n "${SBATCH_ARGS:-}" ] && read -r -a EXTRA <<< "${SBATCH_ARGS}"
exec sbatch -J "${NAME}-${BASE}" "${EXTRA[@]}" "${SCRIPT}" "$@"
