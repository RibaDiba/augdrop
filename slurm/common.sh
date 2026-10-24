# Sourced by the dropaug sbatch scripts, never run on its own:
#
#     source "${REPO_DIR}/slurm/common.sh"
#
# Sourced from REPO_DIR, not from next to the script: under SLURM, $BASH_SOURCE of
# an sbatch script is the spooled copy in /var/spool/slurmd, not the checkout.
#
# Ported from Sensaug's job_scripts/grad_corr_seed.sbatch, the one of its Nexus
# scripts whose environment setup survives a submission from a non-interactive
# shell (a wrapper, cron, an agent, or a job submitting its own follow-up).

# Must match the directory in every #SBATCH --output/--error line. SLURM opens those
# files before the script body runs, so they cannot be templated from REPO_DIR.
LOG_BASE="${LOG_BASE:-/fs/gamma-projects/pruneaug/slurm/logs}"

# Move this job's bootstrap .out/.err (flat in LOG_BASE) into LOG_BASE/<job-name>/.
# A same-filesystem rename keeps the fd SLURM already opened writing to the same
# inode, so nothing is lost and no redirect is needed. Idempotent: a second call
# finds nothing left to move. Array tasks are named %x.%A_%a, other jobs %x.%j.
# Both are tried: a job submitted from inside an array task (train.sbatch queueing
# its eval) inherits the task's SLURM_ARRAY_* variables through --export=ALL.
group_logs() {
    local name="${SLURM_JOB_NAME:-$1}" id ext boot
    [ -n "${SLURM_JOB_ID:-}" ] || return 0
    mkdir -p "${LOG_BASE}/${name}"
    for id in "${SLURM_JOB_ID}" "${SLURM_ARRAY_JOB_ID:-}_${SLURM_ARRAY_TASK_ID:-}"; do
        for ext in out err; do
            boot="${LOG_BASE}/${name}.${id}.${ext}"
            [ -f "${boot}" ] && mv "${boot}" "${LOG_BASE}/${name}/"
        done
    done
    return 0
}

# conda env + src/ on PYTHONPATH. Returns non-zero with a message on failure.
#
# `module` is a shell FUNCTION from /etc/profile.d/modules.sh. A job submitted by
# hand from a login shell inherits it through --export=ALL; one submitted from a
# script does not, and then `module`, `conda` and `torchrun` all die with a bare 127
# (Sensaug jobs 7554655/7554656). So re-source it rather than assume it.
activate_env() {
    local init conda_sh env="${CONDA_ENV:-sensaug}"
    if ! declare -F module >/dev/null 2>&1 && ! command -v module >/dev/null 2>&1; then
        for init in /etc/profile.d/modules.sh /etc/profile.d/ummodules.sh; do
            # shellcheck disable=SC1090
            [ -f "${init}" ] && source "${init}"
        done
    fi
    # The modulefile only exists where MODULEPATH carries it; the install path is
    # stable. CONDA_SH overrides both.
    module load miniforge3 2>/dev/null || true
    if ! command -v conda >/dev/null 2>&1; then
        for conda_sh in "${CONDA_SH:-}" \
                        /opt/common/miniforge3/latest/etc/profile.d/conda.sh \
                        "${HOME}/miniforge3/etc/profile.d/conda.sh"; do
            if [ -n "${conda_sh}" ] && [ -f "${conda_sh}" ]; then
                # shellcheck disable=SC1090
                source "${conda_sh}" && break
            fi
        done
    fi
    if ! command -v conda >/dev/null 2>&1; then
        echo "ERROR: conda not found. Set CONDA_SH=/path/to/etc/profile.d/conda.sh." >&2
        return 1
    fi
    eval "$(conda shell.bash hook)"
    conda activate "${env}" || {
        echo "ERROR: could not activate conda env '${env}'." >&2
        return 1
    }
    # torch ships its own CUDA runtime, so a missing module is a warning, not fatal.
    module load cuda/11.7.0 2>/dev/null \
        || echo "WARNING: could not 'module load cuda/11.7.0'; relying on torch's bundled runtime." >&2

    # vocab / probe / hosts / tasks import top-level from src/, and mmseg configs
    # load the glue with custom_imports = dict(imports=["tasks.mmseg.register"]).
    export PYTHONPATH="${REPO_DIR}/src${PYTHONPATH:+:${PYTHONPATH}}"
    echo "python:   $(command -v python)  ($(python --version 2>&1))"
}

# train.py, probe.py and eval.py all rebuild the run through dropaug.config.
check_hydra() {
    python -c "import hydra, omegaconf" 2>/dev/null || {
        echo "ERROR: hydra / omegaconf are missing from conda env '${CONDA_ENV:-sensaug}'." >&2
        echo "       pip install hydra-core omegaconf   (with that env active)" >&2
        return 1
    }
}

# Print the checkpoint a post-hoc job should load from a finished run.
#   best   the one best*.pth mmengine's CheckpointHook keeps (save_best)
#   last   the file named in last_checkpoint
#   else   a path, used as is
resolve_ckpt() {
    local exp_dir="$1" spec="$2" matches
    case "${spec}" in
        best)
            matches=("${exp_dir}"/best*.pth)
            if [ ! -f "${matches[0]}" ]; then
                echo "ERROR: no best*.pth in ${exp_dir}" >&2; return 1
            fi
            if [ "${#matches[@]}" -gt 1 ]; then
                echo "ERROR: ${#matches[@]} best*.pth in ${exp_dir}; pass CKPT=<path>:" >&2
                printf '  %s\n' "${matches[@]}" >&2
                return 1
            fi
            printf '%s\n' "${matches[0]}"
            ;;
        last)
            if [ ! -s "${exp_dir}/last_checkpoint" ]; then
                echo "ERROR: no last_checkpoint in ${exp_dir}" >&2; return 1
            fi
            spec="$(head -1 "${exp_dir}/last_checkpoint")"
            [ -f "${spec}" ] || { echo "ERROR: last_checkpoint points at missing ${spec}" >&2; return 1; }
            printf '%s\n' "${spec}"
            ;;
        *)
            [ -f "${spec}" ] || { echo "ERROR: checkpoint ${spec} does not exist" >&2; return 1; }
            printf '%s\n' "${spec}"
            ;;
    esac
}

# Word-split an env string into the named array, e.g. `split_args extra "${TRAIN_ARGS:-}"`.
split_args() {
    # shellcheck disable=SC2034
    read -r -a "$1" <<< "${2:-}"
}
