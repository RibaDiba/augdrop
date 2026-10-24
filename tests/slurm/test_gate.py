"""slurm/gate.sh: the job cap (1 A6000 job or 2 A5000 jobs in flight, up to 4 GPUs each).

Runs the real bash against stand-in `squeue` and `sbatch` on PATH, so no Slurm is
needed. The stand-in squeue serves a fixture of `id|command|gres|state` rows.
"""
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
FAKE_REPO = "/fake/repo"
TRAIN = f"{FAKE_REPO}/slurm/train.sbatch"
MATRIX = f"{FAKE_REPO}/slurm/matrix.sbatch"
EVAL = f"{FAKE_REPO}/slurm/eval.sbatch"
OTHER = "/fs/elsewhere/run.sh"

SQUEUE = """#!/bin/bash
[ -n "$FAKE_SQUEUE_FAIL" ] && exit 1
running=0
while [ $# -gt 0 ]; do [ "$1" = "-t" ] && running=1; shift; done
while IFS='|' read -r id cmd gres state; do
    [ -z "$id" ] && continue
    if [ "$running" = 1 ] && [ "$state" != RUNNING ]; then continue; fi
    printf '%s|%s|%s\\n' "$id" "$cmd" "$gres"
done < "$FAKE_QUEUE"
"""
SBATCH = """#!/bin/bash
printf '%s\\n' "$@" > "$FAKE_SBATCH_ARGS"
echo "Submitted batch job 999"
"""


@pytest.fixture
def env(tmp_path):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("squeue", SQUEUE), ("sbatch", SBATCH)):
        p = bin_dir / name
        p.write_text(body)
        p.chmod(p.stat().st_mode | stat.S_IEXEC)
    queue = tmp_path / "queue"
    queue.write_text("")
    # The scripts gate_sbatch reads its GPU count from: train asks for 4, probe for 1.
    for name, n in (("train", 4), ("probe", 1), ("big", 8)):
        (tmp_path / f"{name}.sbatch").write_text(f"#!/bin/bash\n#SBATCH --gres=gpu:rtxa6000:{n}\n")
    (tmp_path / "bare.sbatch").write_text("#!/bin/bash\n")
    e = dict(os.environ)
    e.update(
        PATH=f"{bin_dir}:{e['PATH']}",
        USER="tester",
        REPO_DIR=FAKE_REPO,
        LOG_BASE=str(tmp_path / "logs"),
        FAKE_QUEUE=str(queue),
        FAKE_SBATCH_ARGS=str(tmp_path / "sbatch_args"),
        TRAIN=str(tmp_path / "train.sbatch"),
        PROBE=str(tmp_path / "probe.sbatch"),
        BIG=str(tmp_path / "big.sbatch"),
        BARE=str(tmp_path / "bare.sbatch"),
    )
    e.pop("FAKE_SQUEUE_FAIL", None)
    return e


def run(env, snippet, rows=()):
    Path(env["FAKE_QUEUE"]).write_text("".join(r + "\n" for r in rows))
    return subprocess.run(
        ["bash", "-c", f'source "{REPO}/slurm/gate.sh"\n{snippet}'],
        env=env, capture_output=True, text=True,
    )


def submitted(env):
    return Path(env["FAKE_SBATCH_ARGS"]).read_text().split("\n")[:-1]


def deps_of(args):
    d = [a for a in args if a.startswith("--dependency=")]
    return d[0].split("=", 1)[1] if d else None


def submit(env, gpu, rows, kind="single", opts=""):
    r = run(env, f"gate_sbatch {gpu} {kind} {opts} -- $TRAIN a b", rows)
    assert r.returncode == 0, r.stderr
    return submitted(env)


def a6(i, state="RUNNING", n=4, cmd=TRAIN):
    return f"{i}|{cmd}|gres/gpu:rtxa6000:{n}|{state}"


def a5(i, state="RUNNING", n=4, cmd=TRAIN):
    return f"{i}|{cmd}|gres/gpu:rtxa5000:{n}|{state}"


def test_empty_queue_starts_at_once(env):
    for gpu, t in (("a6000", "rtxa6000"), ("a5000", "rtxa5000")):
        args = submit(env, gpu, [])
        assert f"--gres=gpu:{t}:4" in args
        assert deps_of(args) is None
        assert args[-3:] == [env["TRAIN"], "a", "b"]


def test_the_gpu_count_comes_from_the_script_header(env):
    r = run(env, "gate_sbatch a5000 single -- $PROBE ckpt")
    assert r.returncode == 0, r.stderr
    assert "--gres=gpu:rtxa5000:1" in submitted(env)
    for script in ("$BIG", "$BARE", "/no/such.sbatch"):  # 8 GPUs; no gres line; no file
        r = run(env, f"gate_sbatch a6000 single -- {script}")
        assert r.returncode != 0, script
        assert "must" in r.stderr or "needs" in r.stderr, script


def test_a6000_waits_for_everything_in_flight(env):
    args = submit(env, "a6000", [a6(10), a6(11, "PENDING")])
    assert deps_of(args) == "afterany:10:11"


def test_a5000_alone_does_not_wait_and_a_second_fits(env):
    assert deps_of(submit(env, "a5000", [])) is None
    assert deps_of(submit(env, "a5000", [a5(10)])) is None


def test_third_a5000_waits_for_the_oldest_only(env):
    assert deps_of(submit(env, "a5000", [a5(10), a5(11)])) == "afterany:10"
    assert deps_of(submit(env, "a5000", [a5(10), a5(11), a5(12, "PENDING")])) == "afterany:10:11"


def test_the_two_cards_never_mix(env):
    assert deps_of(submit(env, "a5000", [a6(10)])) == "afterany:10"
    assert deps_of(submit(env, "a6000", [a5(10)])) == "afterany:10"
    assert deps_of(submit(env, "a6000", [a5(10), a5(11)])) == "afterany:10:11"


def test_eval_and_other_repos_are_not_counted(env):
    rows = [a6(10, cmd=EVAL), a6(11, cmd=OTHER), "12|bash|gres/gpu:l40s:2|RUNNING"]
    assert deps_of(submit(env, "a6000", rows)) is None


def test_an_array_costs_the_whole_cap_and_is_one_unit(env):
    rows = [f"20_{i}|{MATRIX}|gres/gpu:rtxa5000:1|RUNNING" for i in (0, 1)]
    rows.append(f"20_[2-9%2]|{MATRIX}|gres/gpu:rtxa5000:1|PENDING")
    assert deps_of(submit(env, "a5000", rows)) == "afterany:20"
    # and a new array waits for the single job ahead of it
    args = submit(env, "a5000", [a5(10)], kind="array", opts="--array=0-3%2")
    assert deps_of(args) == "afterany:10"
    assert "--array=0-3%2" in args


def test_an_untyped_gres_counts_as_the_whole_cap(env):
    assert deps_of(submit(env, "a5000", [f"10|{TRAIN}|gres/gpu:1|RUNNING"])) == "afterany:10"


def test_flags_that_defeat_the_cap_are_refused(env):
    for flag in ("--gres=gpu:4", "-G 4", "--gpus=2", "--dependency=afterok:1", "-d afterok:1"):
        r = run(env, f"gate_sbatch a6000 single {flag} -- $TRAIN")
        assert r.returncode != 0, flag
        assert "not allowed" in r.stderr, flag


def test_script_arguments_may_look_like_flags(env):
    r = run(env, "gate_sbatch a6000 single -- $PROBE ckpt -G --gres=x")
    assert r.returncode == 0, r.stderr
    assert submitted(env)[-4:] == [env["PROBE"], "ckpt", "-G", "--gres=x"]


def test_other_cards_are_refused(env):
    for gpu in ("a4000", "l40s", "4", ""):
        r = run(env, f"gate_sbatch '{gpu}' single -- $TRAIN")
        assert r.returncode != 0, gpu


def test_a_squeue_failure_stops_the_submission(env):
    env["FAKE_SQUEUE_FAIL"] = "1"
    r = run(env, "gate_sbatch a6000 single -- $TRAIN")
    assert r.returncode != 0
    assert not Path(env["FAKE_SBATCH_ARGS"]).exists()


def test_array_throttle(env):
    r = run(env, "gate_array_throttle rtxa6000; gate_array_throttle rtxa5000")
    assert r.stdout.split() == ["1", "2"]


@pytest.mark.parametrize(
    "rows,ok",
    [
        ([], True),
        ([a6(1)], True),
        ([a5(1), a5(2)], True),
        ([a5(1), a5(2), a5(3)], False),
        ([a6(1), a5(2)], False),
        ([a6(1, n=4), a6(2, "PENDING", n=4)], True),
        ([a6(1, n=5)], False),  # more than 4 GPUs
        ([a5(1, n=1), a5(2, n=4)], True),  # the count does not change what a job costs
        ([a6(1), a6(2, "PENDING")], True),  # pending does not run
        ([a6(1), a6(2, cmd=EVAL)], True),  # eval is exempt
        ([a6(1), a6(2, cmd=OTHER)], True),  # another repo's job is not ours
        ([f"1_0|{MATRIX}|gres/gpu:rtxa6000:1|RUNNING", f"1_1|{MATRIX}|gres/gpu:rtxa6000:1|RUNNING"], False),
    ],
)
def test_backstop_counts_running_jobs(env, rows, ok):
    r = run(env, "gate_check_running", rows)
    assert (r.returncode == 0) == ok, r.stderr
    if not ok:
        assert "job cap exceeded" in r.stderr
