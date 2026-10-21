# Sourced, never run on its own:
#
#     source "${REPO_DIR}/slurm/gate.sh"
#
# THE JOB CAP. This repo's Slurm jobs are, at any moment, either
#
#     ONE job on A6000s (rtxa6000), or
#     up to TWO jobs on A5000s (rtxa5000),
#
# never a mix. A job takes the GPU count its script's `#SBATCH --gres=gpu:<type>:N`
# line gives (4 for train and matrix, 1 for probe; at most GATE_MAX_GPUS = 4); the gate
# sets the type. A job submitted past the cap waits for the ones ahead of it
# (--dependency=afterany), like a queue. The cap counts every train, probe and matrix
# job of this repo, pending or running. eval.sbatch is exempt: it is neither counted
# nor gated.
#
# It is a budget of GATE_CAP = 2 units per JOB, whatever its GPU count: a job on A6000s
# costs 2 and a job on A5000s costs 1, so "1 A6000 job or 2 A5000 jobs" is the only way
# to fill it. An array (the matrix) costs the whole budget, and is submitted with the
# throttle that keeps it inside.
#
# Two layers, because sbatch can be called by hand:
#   gate_sbatch         at submission. Every submit path goes through it (submit.sh,
#                       matrix.sbatch --submit). It picks the GPU type, adds the dependency
#                       on every in-flight job older than the ones that fit beside the new
#                       one, and refuses --gres / --dependency flags that would defeat it.
#   gate_check_running  at job start (train.sbatch, probe.sbatch). The backstop for a
#                       job that was submitted without the gate: it refuses to run if
#                       the running jobs already exceed the cap. The refusal comes after
#                       the allocation, so a bypass still holds its GPUs for a few seconds.
#
# Jobs are told apart as this repo's by their command path (squeue %o) under
# ${REPO_DIR}/slurm/. Other repos' jobs are not counted. A job of this repo whose
# --gres names no rtxa6000/rtxa5000 type, or more than GATE_MAX_GPUS of them, is
# counted as more than the whole budget.

GATE_CAP=2
GATE_MAX_GPUS=4

# awk helpers shared by the two squeue readers below. squeue -o '%i|%o|%b' rows are
# <job id>|<command>|<gres>, e.g. 7878654_7|/…/slurm/matrix.sbatch|gres/gpu:rtxa6000:4.
_GATE_AWK='
function weight(b,   s, n, t) {
    if (match(b, /rtxa6000(:[0-9]+)?/)) t = 2
    else if (match(b, /rtxa5000(:[0-9]+)?/)) t = 1
    else return cap + 1
    s = substr(b, RSTART, RLENGTH); n = 1
    if (match(s, /:[0-9]+$/)) n = substr(s, RSTART + 1) + 0
    return (n > maxg) ? cap + 1 : t
}
function mine(cmd) {
    gsub(/^ +| +$/, "", cmd)
    return index(cmd, repo) == 1 && cmd !~ /\/eval\.sbatch$/
}
'

_gate_repo_dir() { printf '%s' "${REPO_DIR:-/fs/gamma-projects/pruneaug}"; }

# a6000|rtxa6000 -> rtxa6000, a5000|rtxa5000 -> rtxa5000; anything else is an error.
gate_type() {
    case "${1:-}" in
        a6000|rtxa6000) printf 'rtxa6000\n' ;;
        a5000|rtxa5000) printf 'rtxa5000\n' ;;
        *) echo "ERROR: GPU='${1:-}' is not allowed: only a6000 (1 job at a time) or a5000 (2 at a time)." >&2
           return 1 ;;
    esac
}

# What one job of that type costs against GATE_CAP.
gate_weight() {
    case "$1" in rtxa6000) echo 2 ;; rtxa5000) echo 1 ;; *) return 1 ;; esac
}

# The array throttle (%K) that keeps an array inside the cap: 1 on A6000, 2 on A5000.
gate_array_throttle() {
    local w; w="$(gate_weight "$1")" || return 1
    echo $(( GATE_CAP / w ))
}

# This repo's in-flight units, oldest first, one "<unit id> <cost>" line each. An array
# is one unit (its base job id) and costs GATE_CAP. Fails if squeue does, so the gate
# never reads "no jobs" off a Slurm outage.
gate_units() {
    local rows
    rows="$(squeue -u "${USER:-$(id -un)}" -h -o '%i|%o|%b')" || {
        echo "ERROR: squeue failed; cannot tell what is in flight, so not submitting." >&2
        return 1
    }
    printf '%s\n' "${rows}" | awk -F'|' -v repo="$(_gate_repo_dir)/slurm/" -v cap="${GATE_CAP}" -v maxg="${GATE_MAX_GPUS}" \
        "${_GATE_AWK}"'
        NF < 3 || !mine($2) { next }
        {
            id = $1; w = weight($3)
            if (id ~ /_/) { sub(/_.*/, "", id); w = cap }
            if (!(id in seen)) { seen[id] = 1; print id, w }
        }' | sort -n
}

# The jobs a new one costing $1 must wait for, colon-joined (empty: none). Walking back
# from the newest in-flight unit, the units that still fit beside it may run alongside;
# the first that does not fit, and everything older, must have ended.
gate_deps() {
    local new_w="$1" units id w i kept=0 cut=-1
    local ids=() costs=()
    units="$(gate_units)" || return 1
    while read -r id w; do
        [ -n "${id}" ] && { ids+=("${id}"); costs+=("${w}"); }
    done <<< "${units}"
    for (( i = ${#ids[@]} - 1; i >= 0; i-- )); do
        if (( kept + costs[i] + new_w <= GATE_CAP )); then
            kept=$(( kept + costs[i] ))
        else
            cut=${i}
            break
        fi
    done
    (( cut >= 0 )) || return 0
    local IFS=:
    printf '%s\n' "${ids[*]:0:cut+1}"
}

# gate_sbatch <a6000|a5000> <single|array> [sbatch options...] -- <script> [args...]
# sbatch with the GPU type set, the GPU count the script's header asks for (1 to
# GATE_MAX_GPUS), and the dependency that keeps the cap. The options may not set the
# GPU or a dependency. Submissions are serialized on a lock, so two at once do not both
# read the same queue.
gate_sbatch() {
    local type kind opts=() o w n
    type="$(gate_type "${1:-}")" || return 1
    kind="${2:-}"; shift 2 || { echo "gate_sbatch: usage: <type> <single|array> [opts] -- script" >&2; return 2; }
    case "${kind}" in single|array) ;; *) echo "gate_sbatch: kind must be single or array" >&2; return 2 ;; esac
    while [ "$#" -gt 0 ] && [ "$1" != "--" ]; do opts+=("$1"); shift; done
    [ "${1:-}" = "--" ] || { echo "gate_sbatch: missing -- before the script" >&2; return 2; }
    shift
    for o in "${opts[@]}"; do
        case "${o}" in
            --gres|--gres=*|-G|-G*|--gpus|--gpus=*|--gpus-per-*|--dependency|--dependency=*|-d|-d*)
                echo "ERROR: ${o} is not allowed: the GPU and the dependency belong to the cap (slurm/gate.sh)." >&2
                return 1 ;;
        esac
    done
    w="$(gate_weight "${type}")"
    [ "${kind}" = "array" ] && w="${GATE_CAP}"
    n="$(sed -n 's/^#SBATCH --gres=gpu:[^:]*:\([0-9][0-9]*\).*/\1/p' "${1:-/dev/null}" 2>/dev/null | head -n 1)"
    if [ -z "${n}" ] || [ "${n}" -lt 1 ] || [ "${n}" -gt "${GATE_MAX_GPUS}" ]; then
        echo "ERROR: ${1:-<no script>} needs '#SBATCH --gres=gpu:<type>:N' with 1 <= N <= ${GATE_MAX_GPUS}" \
             "(got '${n:-none}')." >&2
        return 1
    fi

    local lock="${LOG_BASE:-$(_gate_repo_dir)/slurm/logs}/.gate.lock"
    mkdir -p "$(dirname "${lock}")" 2>/dev/null
    (
        if command -v flock >/dev/null 2>&1; then
            flock -x 9 || exit 1
        else
            echo "WARNING: no flock; two simultaneous submissions could read the same queue." >&2
        fi
        deps="$(gate_deps "${w}")" || exit 1
        if [ -n "${deps}" ]; then
            echo "gate: ${type} x${n} (cap: 1 A6000 job or 2 A5000 jobs) -> waits for job(s) ${deps//:/ }" >&2
        else
            echo "gate: ${type} x${n} (cap: 1 A6000 job or 2 A5000 jobs) -> nothing ahead, can start at once" >&2
        fi
        sbatch --gres="gpu:${type}:${n}" ${deps:+--dependency="afterany:${deps}"} "${opts[@]}" "$@"
    ) 9>"${lock}"
}

# The backstop. Called at the top of a job's body; non-zero means do not run. Counts
# the RUNNING rows (an array's tasks are separate rows, each a job).
gate_check_running() {
    local rows total
    rows="$(squeue -u "${USER:-$(id -un)}" -h -t RUNNING -o '%i|%o|%b')" || {
        echo "ERROR: squeue failed; cannot verify the job cap, so not running." >&2
        return 1
    }
    total="$(printf '%s\n' "${rows}" | awk -F'|' -v repo="$(_gate_repo_dir)/slurm/" -v cap="${GATE_CAP}" -v maxg="${GATE_MAX_GPUS}" \
        "${_GATE_AWK}"'NF >= 3 && mine($2) { s += weight($3) } END { print s + 0 }')"
    if (( total > GATE_CAP )); then
        echo "ERROR: job cap exceeded: the running jobs of this repo cost ${total} (cap ${GATE_CAP}:" >&2
        echo "       1 A6000 job, or 2 A5000 jobs, up to ${GATE_MAX_GPUS} GPUs each). Submit through slurm/submit.sh so" >&2
        echo "       the job queues behind the others. Not running." >&2
        printf '%s\n' "${rows}" | awk -F'|' -v repo="$(_gate_repo_dir)/slurm/" -v cap="${GATE_CAP}" -v maxg="${GATE_MAX_GPUS}" \
            "${_GATE_AWK}"'NF >= 3 && mine($2) { print "       " $1 "  " $3 }' >&2
        return 1
    fi
}
