#!/usr/bin/env bash
# Study X, unattended run of the frozen protocol (docs/studies/X_open_source_baselines.md):
# tuning, the mechanical tuning amendment, the three measurement settings and the five
# benchmark test sets, each followed by its analysis.
#
#     bash scripts/run_x_overnight.sh
#
# Run it from the repository root, which holds runs/ and artifacts/. With X_SNAPSHOT set
# (a clean worktree of the frozen commit), the code and configs come from that snapshot:
# PYTHONPATH=$X_SNAPSHOT/src and $X_SNAPSHOT/configs/baselines/; apply-tuning then writes the
# amendment into the snapshot's measurement configs. Every step is resume-safe: after an
# interruption, run the same command again. The first step that fails stops the script.
# PY selects the interpreter (default: the repository's .venv).

set -euo pipefail

if [ ! -d configs/baselines ] || [ ! -d src/hypergraph_agent ]; then
    echo "run this script from the repository root" >&2
    exit 2
fi
if [ -n "${X_SNAPSHOT:-}" ]; then
    SRC="$X_SNAPSHOT/src"
    CFG="$X_SNAPSHOT/configs/baselines"
else
    SRC="src"
    CFG="configs/baselines"
fi
if [ -z "${PY:-}" ]; then
    PY=".venv/Scripts/python.exe"
    [ -x "$PY" ] || PY=".venv/bin/python"
fi
export PYTHONPATH="$SRC"
export PYTHONUNBUFFERED=1

mkdir -p runs
LOG="runs/x_overnight.log"
exec > >(tee -a "$LOG") 2>&1

say() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }
x() {
    say "baselines $*"
    "$PY" -m hypergraph_agent.baselines "$@"
}

say "start: source $SRC, configs $CFG, python $PY"

# 1. A hard kill can leave the ledger lock behind; every live holder keeps it for milliseconds,
#    so a lock older than one minute is stale.
LOCK="runs/X-ledger.lock"
if [ -e "$LOCK" ] && [ -n "$(find "$LOCK" -mmin +1 2>/dev/null)" ]; then
    say "removing stale $LOCK"
    rm -f "$LOCK"
fi

# 2. Tuning stage (development namespaces; completed jobs are skipped on a rerun) and its analyses.
for t in tune_rq_f1 tune_tt_u_k3 tune_tt_l2_d2; do
    x run --config "$CFG/$t.yaml" --workers 8
    x analyze --config "$CFG/$t.yaml"
done

# 3. Mechanical amendment: refuses unless every tuning job completed and every group has a valid
#    candidate; an already applied record is verified instead.
for m in rq_f1_main tt_u_k3_main tt_l2_d2_main; do
    x apply-tuning --config "$CFG/$m.yaml"
done

# 4. Measurement, rq_f1 (the benchmark sets evaluate its checkpoints).
x run --config "$CFG/rq_f1_main.yaml" --allow-measurement --resume --workers 8
x analyze --config "$CFG/rq_f1_main.yaml"

# 5. Measurement, the two TechTree settings at once, 4 workers each.
x run --config "$CFG/tt_u_k3_main.yaml" --allow-measurement --resume --workers 4 &
PID_U=$!
x run --config "$CFG/tt_l2_d2_main.yaml" --allow-measurement --resume --workers 4 &
PID_L=$!
STATUS=0
wait "$PID_U" || STATUS=$?
wait "$PID_L" || STATUS=$?
if [ "$STATUS" -ne 0 ]; then
    say "a TechTree measurement run failed (status $STATUS); rerun this script to resume"
    exit "$STATUS"
fi
x analyze --config "$CFG/tt_u_k3_main.yaml"
x analyze --config "$CFG/tt_l2_d2_main.yaml"

# 6. Benchmark test sets (evaluation only, on the rq_f1 checkpoints).
for b in bench_crafter bench_psketch_craft bench_msgi_mining bench_little_alchemy_2 bench_freeciv_classic; do
    x run --config "$CFG/$b.yaml" --allow-measurement --resume --workers 8
    x analyze --config "$CFG/$b.yaml"
done

say "done"
