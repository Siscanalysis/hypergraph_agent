"""Action-model learners of the macq library (MIT; ``pip install macq==0.3.11``).

Both learners take STRIPS traces: sequences of fully observed states and the
actions executed between them. A RecipeQuest log converts into one when items
are observable (``traces.successful_steps``): every step that changed the
state is an executed action (``gather_t<type>``, ``activate_t<type>`` or
``craft_<k>``, k the recipe's index among the world's sorted signatures), and
every fact type of the world is a fluent ``t<type>``. Failed attempts, repeats,
waits and submissions are dropped: STRIPS traces have no representation of an
action whose preconditions did not hold, so neither learner sees failures.

``observer_preconditions``  macq's Observer, in this process: the precondition of an
                            action is the intersection of its pre-states (the SAM rule)
``arms_preconditions``      macq's ARMS (weighted MAX-SAT over action, information and
                            plan constraints), in a separate worker process
                            (``arms_worker``, which documents its run-time changes): ARMS
                            iterates over Python sets of fluents, so its result depends on
                            the string hash seed, which a worker fixes
                            (``PYTHONHASHSEED=0``); a job stops after the worker's
                            deterministic step limit with the model learned so far, and a
                            job that still exceeds ``ARMS_TIMEOUT`` seconds of wall clock
                            (a safety net, dependent on machine load) has no model

Fixed settings: ARMS ``upper_bound=3`` (macq: an action counts as fully learned
once its precondition, add or delete list has that many relations), the
library defaults otherwise (``min_support=2``, weights 110 / 100, threshold
0.6, defaults 30); full observation of the states (``percent_missing=0``).
"""

from __future__ import annotations

import json
import os
import queue
import re
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from ..envs.public_schema import ACTIVATE, CRAFT, GATHER
from .traces import successful_steps

ARMS_SETTINGS = {"upper_bound": 3, "min_support": 2}
ARMS_TIMEOUT = 120.0  # seconds per job
HASH_SEED = "0"
_FLUENT = re.compile(r"\bt(\d+)\b")


def action_names(infos: dict) -> dict:
    """Craft action name of every recipe signature (index in sorted order)."""
    return {s: f"craft_{k}" for k, s in enumerate(sorted(infos))}


def step_name(step, names: dict) -> str:
    if step.kind == CRAFT:
        return names[step.target]
    if step.kind in (GATHER, ACTIVATE):
        return f"{'gather' if step.kind == GATHER else 'activate'}_t{step.target}"
    raise ValueError(f"step kind {step.kind} changes no state")


def fact_universe(infos: dict, logs) -> list[int]:
    facts = {f for i in infos.values() for f in (i.effect, *i.items, *i.pool, *(i.known or ()))}
    facts |= {f for lg in logs for f in lg.initial}
    return sorted(int(f) for f in facts)


def trace_records(logs, infos: dict) -> list[dict]:
    """One record per log with at least one state change: the states (lists of
    present fact types) and the executed actions between them."""
    names = action_names(infos)
    out = []
    for lg in logs:
        steps = successful_steps(lg, infos)
        if not steps:
            continue
        states = [sorted(steps[0].before)] + [sorted(s.after) for s in steps]
        out.append({"states": states, "actions": [step_name(s, names) for s in steps]})
    return out


def preconditions_by_signature(precond: dict, infos: dict) -> dict:
    """{action name: fact types} -> {signature: frozenset of fact types} for crafts."""
    inv = {v: k for k, v in action_names(infos).items()}
    return {inv[a]: frozenset(int(x) for x in p) for a, p in precond.items() if a in inv}


def parse_fluents(strings) -> list[int]:
    out = []
    for s in strings:
        m = _FLUENT.search(str(s))
        if m:
            out.append(int(m.group(1)))
    return sorted(set(out))


def _quiet_macq():
    """macq's imports, with the warnings that macq's own modules issue silenced
    (nothing else is silenced)."""
    import logging
    import warnings
    warnings.filterwarnings("ignore", module=r"macq(\.|$)")
    logging.getLogger("macq").setLevel(logging.ERROR)
    from macq.extract import Extract, modes
    from macq.trace import Action, Fluent, PlanningObject, State, Step, Trace, TraceList
    return Extract, modes, Action, Fluent, PlanningObject, State, Step, Trace, TraceList


def macq_tracelist(records: list[dict], fluents: list[int], with_object: bool):
    """A macq ``TraceList`` from ``trace_records``; ARMS needs every fluent and
    action to take a parameter, so ``with_object`` gives them one object."""
    _, _, Action, Fluent, PlanningObject, State, Step, Trace, TraceList = _quiet_macq()
    params = [PlanningObject("agent", "me")] if with_object else []
    F = {f: Fluent(f"t{f}", list(params)) for f in fluents}
    traces = []
    for rec in records:
        steps = []
        for i, present in enumerate(rec["states"]):
            p = set(present)
            act = Action(rec["actions"][i], list(params)) if i < len(rec["actions"]) else None
            steps.append(Step(State({F[f]: f in p for f in fluents}), act, i + 1))
        traces.append(Trace(steps))
    return TraceList(traces)


def observer_preconditions(logs, infos: dict) -> dict:
    """macq's Observer on the successful steps of ``logs``: {signature: facts}."""
    records = trace_records(logs, infos)
    if not records:
        return {}
    Extract, modes, *_ = _quiet_macq()
    from macq.observation import IdentityObservation
    tl = macq_tracelist(records, fact_universe(infos, logs), with_object=False)
    model = Extract(tl.tokenize(IdentityObservation), modes.OBSERVER)
    return preconditions_by_signature({a.name: parse_fluents(a.precond) for a in model.actions}, infos)


# ----------------------------------------------------------------------- ARMS

def arms_job(job_id: str, logs, infos: dict) -> dict:
    return {"id": job_id, "fluents": fact_universe(infos, logs), "traces": trace_records(logs, infos),
            "settings": dict(ARMS_SETTINGS)}


def _reader(stream, q):
    for line in stream:
        q.put(line)
    q.put(None)


def _src_root() -> str:
    return str(Path(__file__).resolve().parents[2])


def run_arms(jobs: list[dict], *, timeout: float = ARMS_TIMEOUT, workers: int = 1,
             python: str | None = None, on_result=None, hash_seed: str = HASH_SEED) -> dict:
    """{job id: result} for ARMS jobs. Jobs are split into ``workers`` chunks;
    each chunk runs in order in one worker process at a time with
    ``PYTHONHASHSEED`` fixed (``hash_seed``; a job's result does not depend on
    its chunk, but it depends on the hash seed, which decides ties); a job that
    times out, or whose worker dies, is recorded with its status and the
    chunk's worker restarts after it. ``on_result(result)`` is called as each
    result arrives (from the chunks' threads)."""
    results: dict = {}
    lock = threading.Lock()

    def keep(res: dict) -> None:
        with lock:
            results[res["id"]] = res
            if on_result is not None:
                on_result(res)

    pending = [j for j in jobs if j["traces"]]
    for j in jobs:
        if not j["traces"]:
            keep({"id": j["id"], "status": "no_traces", "precond": {}, "seconds": 0.0})
    env = {**os.environ, "PYTHONHASHSEED": str(hash_seed),
           "PYTHONPATH": os.pathsep.join([_src_root()] + [p for p in [os.environ.get("PYTHONPATH")] if p])}
    chunks = [pending[i::max(1, workers)] for i in range(max(1, workers))]
    errors: list = []

    def run(chunk):
        try:
            _run_chunk(chunk, timeout, env, python, keep)
        except Exception as exc:  # noqa: BLE001 (re-raised in the caller's thread)
            errors.append(exc)

    threads = [threading.Thread(target=run, args=(c,)) for c in chunks if c]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errors:
        raise errors[0]
    return results


def _run_chunk(pending: list[dict], timeout: float, env: dict, python: str | None, keep) -> None:
    worker = [python or sys.executable, "-m", "hypergraph_agent.inference_baselines.arms_worker"]
    while pending:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "jobs.json"
            path.write_text(json.dumps(pending), encoding="utf-8")
            err = open(Path(tmp) / "stderr.txt", "w+", encoding="utf-8")
            proc = subprocess.Popen(worker + [str(path)], stdout=subprocess.PIPE, stderr=err, env=env,
                                    text=True, encoding="utf-8")
            q: queue.Queue = queue.Queue()
            threading.Thread(target=_reader, args=(proc.stdout, q), daemon=True).start()
            done = 0
            for job in pending:
                t0 = time.time()
                try:
                    line = q.get(timeout=timeout)
                except queue.Empty:
                    proc.kill()
                    keep({"id": job["id"], "status": "timeout", "precond": {},
                          "seconds": round(time.time() - t0, 1)})
                    done += 1
                    break
                if line is None:
                    proc.wait()
                    err.seek(0)
                    tail = err.read()[-400:]
                    keep({"id": job["id"], "status": "worker_exit", "precond": {},
                          "error": f"exit {proc.returncode}: {tail}", "seconds": 0.0})
                    done += 1
                    break
                keep(json.loads(line))
                done += 1
            proc.wait()
            if proc.stdout:
                proc.stdout.close()
            err.close()
        pending = pending[done:]


ARMS_STATUSES = ("ok", "iteration_cap", "error", "timeout", "worker_exit", "no_traces")


def arms_preconditions(result: dict | None, infos: dict) -> dict:
    """{signature: facts} from one ARMS job result: its model when it finished
    or stopped at the step limit, nothing when it failed."""
    if not result or result.get("status") not in ("ok", "iteration_cap"):
        return {}
    return preconditions_by_signature(result["precond"], infos)
