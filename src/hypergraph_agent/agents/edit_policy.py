"""Learned edit proposals for the hypergraph walker.

The policy scores every single-incidence edit of the current node (add,
remove or swap one candidate base fact of one attempted recipe) from public
features of the logged evidence and samples one; the walk then applies the
same Metropolis acceptance as the uniform and focused walks.

Training (``train_edit_policy``) is REINFORCE on internal search problems
built from training-world evidence: start at a random node, walk with the
policy until a node consistent with every logged observation is reached or
the evaluation cap is spent; the return is minus the normalized number of
hypothesis evaluations, with a penalty when no consistent node is found.
Consistency with public evidence is the only training signal; hidden rules are
never used. Internal search costs no environment interactions and is reported
as search work.
"""

from __future__ import annotations

import copy

import numpy as np
import torch
from torch import nn

from ..envs.public_schema import ACTIVATE, CRAFT, GATHER
from .walker import Edit, Evidence, WorldState, local_walk, neighbours, random_node

EDIT_FEATS = 12


def support_stats(logs, infos) -> dict:
    """For each (recipe, candidate base): the fraction of its craft attempts, in
    episodes where the goal was eventually observed, made while that base was
    present. Uses observed base facts only."""
    counts: dict = {}
    totals: dict = {}
    for log in logs:
        if not log.goal_seen():
            continue
        bases = {b for b in log.initial}
        for kind, x in log.steps:
            if kind in (GATHER, ACTIVATE):
                bases.add(x)
            elif kind == CRAFT and infos[x].known is None:
                totals[x] = totals.get(x, 0) + 1
                for b in infos[x].pool:
                    counts[(x, b)] = counts.get((x, b), 0) + (b in bases)
    return {k: v / totals[k[0]] for k, v in counts.items()}


def edit_features(cur: dict, edits: list[Edit], w: WorldState, logs, focus: dict, prev: Edit | None,
                  support: dict) -> np.ndarray:
    n_logs = max(len(logs), 1)
    attempts = {}
    for lg in logs:
        for s in lg.attempted():
            attempts[s] = attempts.get(s, 0) + 1
    focus_total = max(sum(focus.values()), 1)
    focus_by_recipe: dict = {}
    for e, c in focus.items():
        focus_by_recipe[e.signature] = focus_by_recipe.get(e.signature, 0) + c
    X = np.zeros((len(edits), EDIT_FEATS), dtype=np.float32)
    for i, e in enumerate(edits):
        X[i, {"add": 0, "remove": 1, "swap": 2}[e.op]] = 1.0
        c = focus.get(e, 0)
        X[i, 3] = np.log1p(c)
        X[i, 4] = float(c > 0)
        X[i, 5] = len(cur[e.signature]) / w.max_size
        X[i, 6] = attempts.get(e.signature, 0) / n_logs
        X[i, 7] = support.get((e.signature, e.add), 0.0) if e.add is not None else 0.0
        X[i, 8] = support.get((e.signature, e.remove), 0.0) if e.remove is not None else 0.0
        X[i, 9] = float(prev is not None and e == prev)
        X[i, 10] = focus_by_recipe.get(e.signature, 0) / focus_total
        X[i, 11] = 1.0
    return X


class EditPolicy(nn.Module):
    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(EDIT_FEATS, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.label = "learned_edit_policy"

    def logits(self, X) -> torch.Tensor:
        return self.net(torch.as_tensor(X)).squeeze(-1)

    def propose(self, cur, w, logs, focus, attempted, prev, rng, trace=None):
        edits = neighbours(cur, w.infos, attempted, w.max_size)
        if not edits:
            return None
        X = edit_features(cur, edits, w, logs, focus, prev, support_stats(logs, w.infos))
        with torch.no_grad():
            p = torch.softmax(self.logits(X), -1).double().numpy()
        idx = int(rng.choice(len(edits), p=p / p.sum()))
        if trace is not None:
            trace.append((X, idx))
        return edits[idx]


def snapshot(w: WorldState, k: int) -> WorldState:
    """The world's state of knowledge after its first ``k`` logged episodes."""
    s = copy.copy(w)
    s.evidence = Evidence(w.infos)
    s.evidence.logs = list(w.evidence.logs[:k])
    s.node = dict(w.node)
    return s


def search_problems(worlds, min_logs: int = 2) -> list[WorldState]:
    out = []
    for w in worlds:
        for k in range(min_logs, len(w.evidence.logs) + 1):
            out.append(snapshot(w, k))
    return out


def train_edit_policy(policy: EditPolicy, problems: list[WorldState], *, iters: int, max_evals: int,
                      temperature: float, restart_after: int, lr: float, rng: np.random.Generator,
                      max_size: int = 3, ent_coef: float = 0.01, log=None) -> dict:
    if not problems:
        raise ValueError("no search problems to train on")
    opt = torch.optim.Adam(policy.parameters(), lr=lr)
    baseline, evals_hist, solved_hist = None, [], []
    total_evals = 0
    for it in range(iters):
        w = problems[int(rng.integers(len(problems)))]
        attempted = sorted({s for lg in w.evidence.logs for s in lg.attempted() if w.infos[s].known is None})
        start = random_node(w, attempted, rng, w.node)
        trace: list = []
        _, st = local_walk(start, w, None, "learned", rng, max_size, max_evals, temperature,
                           restart_after, policy, trace)
        total_evals += st["evals"]
        evals_hist.append(st["evals"])
        solved_hist.append(st["violations"] == 0)
        G = -st["evals"] / max_evals - (1.0 if st["violations"] > 0 else 0.0)
        baseline = G if baseline is None else 0.95 * baseline + 0.05 * G
        if trace:
            logp, ent = 0.0, 0.0
            for X, idx in trace:
                lp = torch.log_softmax(policy.logits(X), -1)
                logp = logp + lp[idx]
                ent = ent - (lp.exp() * lp).sum()
            loss = -(G - baseline) * logp / len(trace) - ent_coef * ent / len(trace)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
        if log is not None and (it + 1) % 100 == 0:
            log({"type": "edit_policy_training", "iter": it + 1,
                 "mean_evals_last100": float(np.mean(evals_hist[-100:])),
                 "solved_last100": float(np.mean(solved_hist[-100:]))})
    return {"iters": iters, "search_evaluations": total_evals,
            "mean_evals_first100": float(np.mean(evals_hist[:100])),
            "mean_evals_last100": float(np.mean(evals_hist[-100:])),
            "solved_last100": float(np.mean(solved_hist[-100:]))}


def benchmark_search(problems: list[WorldState], modes, *, policy=None, starts: int = 5, seed: int = 0,
                     max_evals: int, temperature: float, restart_after: int, max_size: int = 3) -> dict:
    """Paired offline comparison of walks: identical problems and start nodes
    for every mode; reports evaluations to a consistent node and success."""
    out = {m: {"evals": [], "solved": []} for m in modes}
    for pi, w in enumerate(problems):
        attempted = sorted({s for lg in w.evidence.logs for s in lg.attempted() if w.infos[s].known is None})
        for k in range(starts):
            start = random_node(w, attempted, np.random.default_rng((seed, pi, k)), w.node)
            for m in modes:
                wp = snapshot(w, len(w.evidence.logs))
                _, st = local_walk(start, wp, None, m, np.random.default_rng((seed, pi, k, 1)), max_size,
                                   max_evals, temperature, restart_after, policy)
                out[m]["evals"].append(st["evals"])
                out[m]["solved"].append(st["violations"] == 0)
    return {m: {"mean_evals": float(np.mean(v["evals"])), "median_evals": float(np.median(v["evals"])),
                "solved": float(np.mean(v["solved"])), "n": len(v["evals"])} for m, v in out.items()}
