#!/usr/bin/env bash
# Study XS, unattended run of the frozen protocol (docs/studies/XS_learner_followup.md): a budget
# check, the two RecipeQuest members (budget slack 2 and 12, sparse and shaped MaskablePPO in each)
# at once, 4 workers each, then their analyses and the combined analysis of the study.
#
#     export XS_SNAPSHOT=../snapshots/XS-<commit>    # a clean worktree of the frozen commit
#     bash "$XS_SNAPSHOT/scripts/run_xs_overnight.sh" # measurement, from the repository root
#     bash scripts/run_xs_overnight.sh --dev         # development members, this checkout's code
#
# Run it from the repository root, which holds runs/ and artifacts/ (study X's frozen tuning record,
# artifacts/baselines/x-rq-tune/tuning_summary.json, is read from there). The measurement needs
# XS_SNAPSHOT, a git worktree of the frozen commit without changes: the code and configs come from
# it (PYTHONPATH=$XS_SNAPSHOT/src, $XS_SNAPSHOT/configs/baselines/) and its commit is logged.
# --dev runs the development members instead (allocation dev, namespaces ending in _dev) with this
# checkout's code; there is no other path and no fallback. Before any launch or resume the driver
# checks that the ledger still covers the caps of every job without a completed run (check-budget)
# and stops otherwise: further budget only through a dated amendment. Every step is resume-safe:
# after an interruption, run the same command again (completed jobs are skipped). The first step
# that fails stops the script. PY selects the interpreter (default: the repository's .venv).

set -euo pipefail

MODE="measurement"
if [ "${1:-}" = "--dev" ]; then
    MODE="dev"
elif [ -n "${1:-}" ]; then
    echo "usage: bash run_xs_overnight.sh [--dev]" >&2
    exit 2
fi
if [ ! -d configs/baselines ] || [ ! -d src/hypergraph_agent ]; then
    echo "run this script from the repository root" >&2
    exit 2
fi
if [ "$MODE" = "measurement" ]; then
    if [ -z "${XS_SNAPSHOT:-}" ]; then
        echo "XS_SNAPSHOT is not set: the measurement runs only from a clean worktree of the frozen" \
             "commit (--dev runs the development members)" >&2
        exit 2
    fi
    if ! DIRTY="$(git -C "$XS_SNAPSHOT" status --porcelain 2>&1)"; then
        echo "XS_SNAPSHOT=$XS_SNAPSHOT is not a git worktree: $DIRTY" >&2
        exit 2
    fi
    if [ -n "$DIRTY" ]; then
        echo "XS_SNAPSHOT=$XS_SNAPSHOT has uncommitted changes:" >&2
        echo "$DIRTY" >&2
        exit 2
    fi
    SRC="$XS_SNAPSHOT/src"
    CFG="$XS_SNAPSHOT/configs/baselines"
    MEMBERS="xs_rq_s2 xs_rq_s12"
    STUDY="xs_rq"
    LOG="runs/xs_overnight.log"
else
    SRC="src"
    CFG="configs/baselines"
    MEMBERS="xs_rq_s2_dev xs_rq_s12_dev"
    STUDY="xs_rq_dev"
    LOG="runs/xs_overnight_dev.log"
fi
if [ -z "${PY:-}" ]; then
    PY=".venv/Scripts/python.exe"
    [ -x "$PY" ] || PY=".venv/bin/python"
fi
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1

mkdir -p runs
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
x() {
    say "baselines $*"
    "$PY" -m hypergraph_agent.baselines "$@"
}

say "start ($MODE): source $SRC, configs $CFG, python $PY"
if [ "$MODE" = "measurement" ]; then
    say "snapshot $XS_SNAPSHOT at commit $(git -C "$XS_SNAPSHOT" rev-parse HEAD)"
fi

# 1. A hard kill can leave the ledger lock behind; every live holder keeps it for milliseconds,
#    so a lock older than one minute is stale.
LOCK="runs/XS-ledger.lock"
if [ -e "$LOCK" ] && [ -n "$(find "$LOCK" -mmin +1 2>/dev/null)" ]; then
    say "removing stale $LOCK"
    rm -f "$LOCK"
fi

# 2. Budget: continue only if what the ledger has left covers the caps of every job of both members
#    that has no completed run (training first of all); otherwise stop.
if ! x check-budget --config "$CFG/$STUDY.yaml"; then
    say "stopping: the ledger does not cover the missing jobs; further budget only through a dated amendment"
    exit 3
fi

# 3. Both members at once, 4 workers each (8 training processes). A measurement run checks the
#    inherited tuning record before it starts and exits non-zero when any of its jobs failed.
PIDS=""
for m in $MEMBERS; do
    if [ "$MODE" = "measurement" ]; then
        x run --config "$CFG/$m.yaml" --allow-measurement --resume --workers 4 &
    else
        x run --config "$CFG/$m.yaml" --resume --workers 4 &
    fi
    PIDS="$PIDS $!"
done
STATUS=0
for pid in $PIDS; do
    wait "$pid" || STATUS=$?
done
if [ "$STATUS" -ne 0 ]; then
    say "a run failed (status $STATUS); rerun this script to resume"
    exit "$STATUS"
fi

# 4. Analyses: each member, then the combined study (selection on validation worlds, tests). The
#    development analyses go to runs/, never to artifacts/.
for c in $MEMBERS $STUDY; do
    if [ "$MODE" = "measurement" ]; then
        x analyze --config "$CFG/$c.yaml"
    else
        x analyze --config "$CFG/$c.yaml" --out "runs/xs_dev_analysis/$c"
    fi
done

say "done"
