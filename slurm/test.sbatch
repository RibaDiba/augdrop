#!/bin/bash

#SBATCH --job-name=dropaug-test
#SBATCH --nodes=1

## GAMMA, 1 GPU (a job under the cap, slurm/gate.sh: submit with slurm/submit.sh, which
## counts it like a probe). pytest on the vocab/host/augment tests, then the GPU smoke test.
#SBATCH --time=01:00:00
#SBATCH --qos=default
#SBATCH --account=gamma
#SBATCH --partition=gamma
#SBATCH --gres=gpu:rtxa6000:1
#SBATCH --mem=32gb
#SBATCH --ntasks=4

#SBATCH --output=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.out
#SBATCH --error=/fs/gamma-projects/pruneaug/slurm/logs/%x.%j.err

#SBATCH --mail-user=amodak@terpmail.umd.edu
#SBATCH --mail-type=FAIL,END

set -o pipefail

# ==========================================================================
# test.sbatch -- 1-GPU check of the *_diff hosts.
#
#   bash slurm/submit.sh <name> slurm/test.sbatch
#
# Env knobs: PYTEST_ARGS (default: the vocab, hosts, tasks, probe and config tests),
# CONDA_ENV (common.sh's default, sensaug: the only env on Nexus), REPO_DIR.
# ==========================================================================

REPO_DIR="${REPO_DIR:-/fs/gamma-projects/pruneaug}"
cd "${REPO_DIR}" || { echo "ERROR: REPO_DIR=${REPO_DIR} does not exist" >&2; exit 1; }
# shellcheck disable=SC1091
source "${REPO_DIR}/slurm/common.sh"
group_logs dropaug-test
gate_check_running || exit 1

activate_env || exit 1

rc=0
echo "== pytest"
python -m pytest ${PYTEST_ARGS:-tests/vocab tests/hosts tests/tasks tests/probe tests/config} -x -q || rc=1
echo "== smoke"
python scripts/smoke_diff_hosts.py || rc=1
exit ${rc}
