"""Study PG arms: a policy graph for TechTree (public information only).

Agents hold only the public facade (``env.agent_interface``); this module is under the same source
check as ``explorers``, ``layered`` and ``baselines`` (no reflective builtins, dunder, frame or
traceback attributes, no private attributes of objects other than ``self``; imports restricted).

The graph
---------
A node is the public state of an episode plus a bounded memory of the agent's own presses:
(goal, held concepts, the last m presses), with m = ``order``, by default the longest public
sequence length minus 1, so that a node and its next press can spell any concept of the world. The
context is shorter at the start of an episode (fewer presses so far). Nodes, and the transitions
observed between them (edges, with counts), are kept per world and dropped with it: a fresh world
starts with an empty graph. Edges are bookkeeping for the reports (graph size); no arm reads them.

Each node holds a Dirichlet distribution over the actions available there (the available
primitives: base symbols and announced active slots; ``wait`` is never chosen and ``submit`` is
pressed as soon as the goal is held, as by the random agent), with prior alpha0 (default 1, the
uniform prior) plus success counts.

Learning (credit)
-----------------
After every successful episode, and only then, the episode's decisions are loop-erased in node
space (chronologically: a node visited again cuts the path back to its first visit) and every
(node, action) left on the loop-erased path gains one count. Each node of that path therefore
credits the action taken at its last visit, the one after which the episode reached the goal
without returning; actions inside loops are not credited. Every successful episode counts, not
only the shortest so far. Without loop erasure the counts would add nothing in a combination lock:
in a uniformly random successful episode a node is left by the wrong presses about as often as the
uniform rule predicts, so success-weighted action frequencies stay uniform there.

Acting
------
``pg``: Thompson sampling with a fresh draw at every decision: a categorical drawn from the node's
Dirichlet, then an action drawn from it. A single action drawn that way has the marginal
distribution (alpha0 + n_a) / sum_b (alpha0 + n_b), so the agent samples that predictive directly
with one uniform variate per decision. ``pg_uniform``: the same graph and bookkeeping (counts are
kept but never read) with uniform choices: the control that isolates the learned distributions; it
is distributed exactly as the random agent. ``pg_greedy``: an action of maximal count, ties
uniform (with no counts, uniform). All three map the decision's uniform variate to an action the
same way, and the variates of an episode come from a stream seeded by (the agent seed, the world,
the episode index within the world): every arm of a seed reads the same numbers, so the three arms
act identically in a world until its first success (common random numbers).

The factory also builds the comparison arms of studies U and L unchanged (``pooled``, ``remember``,
``promote``, ...) and the anchors (``random``, ``oracle``, ``oracle_library``), so one study config
can run them side by side. ``ROW_SUMMARIES`` declares the analysis summary of the graph counters
that the policy-graph rows carry.
"""

from __future__ import annotations

from collections import Counter

import numpy as np

from .explorers import Episode, make_explorer
from .layered import make_layered

DEFAULT_PARAMS = {"order": "max_length_minus_1", "alpha0": 1.0}
MODES = {"pg": "thompson", "pg_uniform": "uniform", "pg_greedy": "greedy"}
REPORTING_ARMS = ("pg_uniform",)  # its actions never depend on evidence


def check_params(params: dict | None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    extra = set(p) - set(DEFAULT_PARAMS)
    if extra:
        raise KeyError(f"unknown study PG agent parameters {sorted(extra)}")
    order = p["order"]
    if order != "max_length_minus_1" and not (isinstance(order, int) and not isinstance(order, bool)
                                              and order >= 0):
        raise ValueError("agent parameter order must be 'max_length_minus_1' or an integer >= 0")
    if isinstance(p["alpha0"], bool) or not isinstance(p["alpha0"], (int, float)) or not p["alpha0"] > 0:
        raise ValueError("agent parameter alpha0 must be a positive number")
    return p


# --------------------------------------------------------- environment-independent core
def pick(weights, u: float) -> int:
    """Index drawn with probability proportional to ``weights``, from one uniform variate ``u``."""
    cum = np.cumsum(np.asarray(weights, dtype=float))
    return int(min(np.searchsorted(cum, u * cum[-1], side="right"), len(cum) - 1))


class LoopErasedPath:
    """The decisions of one episode, loop-erased in node space as they are made."""

    def __init__(self):
        self.steps: list[tuple] = []  # (node, action)
        self.pos: dict = {}

    def visit(self, node, action):
        i = self.pos.get(node)
        if i is not None:  # a loop closed at this node: cut the path back to its first visit
            for n, _ in self.steps[i:]:
                del self.pos[n]
            del self.steps[i:]
        self.pos[node] = len(self.steps)
        self.steps.append((node, action))


class WorldGraph:
    """What one agent has learned about one world: nodes, observed transitions and the success
    counts of the node distributions."""

    def __init__(self, alpha0: float = 1.0):
        self.alpha0 = alpha0
        self.counts: dict = {}  # node -> Counter(action -> credited count)
        self.edges: Counter = Counter()  # (node, action, next node) -> times observed
        self.nodes: set = set()
        self.episodes = 0
        self.successes = 0

    def weights(self, node, actions: tuple, mode: str) -> list[float]:
        n = self.counts.get(node)
        if mode == "uniform" or not n:
            return [1.0] * len(actions)
        if mode == "greedy":
            top = max(n[a] for a in actions)
            return [1.0 if n[a] == top else 0.0 for a in actions]
        return [self.alpha0 + n[a] for a in actions]

    def choose(self, node, actions: tuple, mode: str, u: float):
        """(action, whether the node had counts)."""
        self.nodes.add(node)
        return actions[pick(self.weights(node, actions, mode), u)], bool(self.counts.get(node))

    def observe(self, node, action, nxt):
        self.nodes.add(nxt)
        self.edges[(node, action, nxt)] += 1

    def credit(self, path: LoopErasedPath) -> int:
        self.successes += 1
        for node, action in path.steps:
            self.counts.setdefault(node, Counter())[action] += 1
        return len(path.steps)

    def counters(self) -> dict:
        return {"nodes": len(self.nodes), "edges": len(self.edges), "count_nodes": len(self.counts),
                "successes": self.successes}


# ------------------------------------------------------------------------- agent
class PolicyGraph:
    """Policy-graph arm (see the module docstring); ``mode`` is thompson, uniform or greedy."""

    def __init__(self, rng: np.random.Generator, mode: str, params: dict | None = None):
        # drawn first, so every arm built from one seed shares it (common random numbers)
        self.stream_seed = int(rng.integers(2 ** 62))
        if mode not in MODES.values():
            raise ValueError(f"unknown policy-graph mode {mode!r}")
        self.mode = mode
        p = check_params(params)
        self.order_param, self.alpha0 = p["order"], float(p["alpha0"])
        self.worlds: dict[str, WorldGraph] = {}

    def reveal(self, world_key: str, links: dict):
        pass  # no arm of this module is given links

    def order(self, spec) -> int:
        return max(spec.level_lengths) - 1 if self.order_param == "max_length_minus_1" else self.order_param

    def node(self, ep: Episode, m: int) -> tuple:
        return (ep.spec.goal, ep.held, tuple(ep.trace[-m:]) if m else ())

    def run_episode(self, env, token, meter) -> dict:
        ep = Episode(env, token, meter)
        spec = ep.spec
        g = self.worlds.setdefault(spec.world_key, WorldGraph(self.alpha0))
        rng = np.random.default_rng([self.stream_seed, g.episodes, *spec.world_key.encode()])
        g.episodes += 1
        m = self.order(spec)
        path = LoopErasedPath()
        decisions = informed = 0
        node = self.node(ep, m)
        while not ep.done:
            if spec.goal in ep.held:
                ep.step(spec.submit_action)
                continue
            actions = tuple(s for s in range(spec.n_symbols) if ep.obs.available[s])
            a, had = g.choose(node, actions, self.mode, float(rng.random()))
            decisions += 1
            informed += had
            path.visit(node, a)
            ep.step(a)
            if ep.status == "budget_exhausted":
                break
            nxt = self.node(ep, m)
            g.observe(node, a, nxt)
            node = nxt
        credited = g.credit(path) if ep.success else 0
        row = ep.row()
        row.update({"pg_decisions": decisions, "pg_informed": informed, "pg_path": len(path.steps),
                    "pg_credited": credited, **{f"pg_{k}": v for k, v in g.counters().items()}})
        return row


# ------------------------------------------------------------------------- row summaries
def graph_summary(rows: list[dict]) -> dict:
    """{variant: {arm: ...}}: per episode means of the decisions, of the decisions taken at a node
    with counts, of the loop-erased path and of the credited pairs; the graph size (nodes, edges,
    nodes with counts) at the end of each world, averaged over worlds. Policy-graph rows only."""
    groups: dict = {}
    for r in rows:
        if "pg_decisions" in r:
            groups.setdefault((r.get("variant"), r.get("arm")), []).append(r)
    out: dict = {}
    for (variant, arm), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        last: dict = {}
        for r in sorted(rs, key=lambda r: (r.get("seed", 0), r["world_key"], r["world_episode"])):
            last[(r.get("seed", 0), r["world_key"])] = r
        ends = list(last.values())
        out.setdefault(variant, {})[arm] = {
            "episodes": len(rs),
            **{f"{k}_per_episode": float(np.mean([r[f"pg_{k}"] for r in rs]))
               for k in ("decisions", "informed", "path", "credited")},
            "informed_share": sum(r["pg_informed"] for r in rs) / max(1, sum(r["pg_decisions"] for r in rs)),
            **{f"{k}_per_world_end": float(np.mean([r[f"pg_{k}"] for r in ends]))
               for k in ("nodes", "edges", "count_nodes", "successes")},
        }
    return out


ROW_SUMMARIES = {"policy_graph": graph_summary}


# ------------------------------------------------------------------------------ factory
U_ARMS = ("pooled", "isolated", "nomem", "blind", "oracle", "random")
L_ARMS = ("none", "remember", "promote", "sham_promote", "adaptive", "oracle_library")


def make_policygraph(name: str, rng: np.random.Generator, params: dict | None = None):
    """Study PG arms; the comparison arms and anchors of studies U and L are built unchanged."""
    p = check_params(params)
    if name in MODES:
        return PolicyGraph(rng, MODES[name], p)
    if name in U_ARMS:
        return make_explorer(name, rng)
    if name in L_ARMS:
        return make_layered(name, rng)
    raise KeyError(f"unknown study PG arm {name!r}")
