"""Primitive-step executor with a bounded call stack.

One ``Executor`` owns one environment episode. Every choice, whether made by
the root agent, a skill controller or a nested child, ends in actual primitive
transitions executed here one at a time; each is charged to the budget meter
exactly once (nested durations are sums of these, never re-counted).

Termination of a skill call: public target predicate, timeout (primitive
steps, capped by the caller's remaining time), task terminal (checked after
every primitive, including inside nested calls), dispatch limit, or budget
exhaustion. Calls that cannot do anything useful (target already present,
invalid binding, depth limit) execute one ``wait`` primitive, so no choice is
free and zero-step loops cannot occur. Skills cannot submit and cannot edit
state except through primitive actions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from ..envs.public_schema import CRAFT, SKILL, SUBMIT, WAIT, PublicTaskSpec
from ..envs.vocabulary import TYPE_NAMES
from ..representations.features import SkillCandidate, build_structure, memory_features
from ..training.budget import BudgetExhausted
from ..training.returns import option_return
from .binding import binding_valid, bindings
from .spec import SkillSpec


@dataclass
class OptionOutcome:
    status: str
    tau: int
    rewards: list
    task_terminal: bool


@dataclass
class DecisionTrace:
    present: list = field(default_factory=list)
    mem: list = field(default_factory=list)
    actions: list = field(default_factory=list)
    logp: list = field(default_factory=list)
    values: list = field(default_factory=list)
    R: list = field(default_factory=list)
    tau: list = field(default_factory=list)
    valid: list = field(default_factory=list)
    choice_keys: list = field(default_factory=list)
    events: list = field(default_factory=list)  # public per-decision trace (for mining)
    alpha_means: list = field(default_factory=list)
    status: str = ""
    terminal: bool = False
    final_value: float = 0.0
    tau_total: int = 0
    n_skill_calls: int = 0


def contract_support(spec: SkillSpec, task: PublicTaskSpec, topo_snapshot, live: bool):
    """Predicted prerequisite links of a bound skill, as (fact index, weight).

    ``live``: derive them from the current dependency snapshot (dependency
    revision reaches the skill graph); otherwise use the contract frozen at
    admission (construction links only)."""
    by_type = task.fact_by_type()
    if live and topo_snapshot is not None:
        support: dict[int, float] = {}
        for r in task.rules:
            if task.facts[r.effect].type_id != spec.target_type:
                continue
            for i in r.item_inputs:
                support[task.facts[i].type_id] = 1.0
            if r.known_base is not None:
                for i in r.known_base:
                    support[task.facts[i].type_id] = 1.0
                continue
            view = topo_snapshot.rule(r.signature)
            if view is None:
                continue
            for t in view.active:
                support[t] = max(support.get(t, 0.0), view.marginal(t))
    else:
        support = dict(spec.contract_support)
    return tuple((by_type[t], float(w)) for t, w in sorted(support.items()) if t in by_type)


class Executor:
    def __init__(self, env, meter, purpose: str, action_gen: torch.Generator, *,
                 topo=None, topo_snapshot=None, graph_mode: str = "known",
                 use_posterior: bool = True, library=None, lib_snapshot=None,
                 live_contracts: bool = False, max_depth: int = 3, dispatch_factor: int = 4,
                 greedy_skills: bool = False):
        self.env, self.meter, self.purpose, self.gen = env, meter, purpose, action_gen
        self.topo, self.topo_snapshot = topo, topo_snapshot
        self.graph_mode, self.use_posterior = graph_mode, use_posterior
        self.library, self.lib_snapshot = library, lib_snapshot
        self.live_contracts, self.max_depth = live_contracts, max_depth
        self.dispatch_factor, self.greedy_skills = dispatch_factor, greedy_skills

    # ------------------------------------------------------------ episode
    def reset(self, task, dyn_seed: int):
        obs, _ = self.env.reset(seed=dyn_seed, options={"task": task})
        self.spec: PublicTaskSpec = self.env.public_spec
        self.obs = obs
        self.terminated, self.success, self.termination = False, False, None
        self.n_primitive, self.n_noop = 0, 0
        self.rewards: list[float] = []
        self.call_log: list[dict] = []
        self.wait_index = self.spec.action_index()["wait"]
        return obs

    def skill_candidates(self, refs=None) -> tuple[SkillCandidate, ...]:
        if self.lib_snapshot is None:
            return ()
        specs = self.lib_snapshot.skills if refs is None else [
            s for s in self.lib_snapshot.skills if s.ref in {tuple(r) for r in refs}]
        out = []
        for s in specs:
            for fact in bindings(s, self.spec):
                out.append(SkillCandidate(
                    s.skill_key, s.version, s.level, fact,
                    contract_support(s, self.spec, self.topo_snapshot, self.live_contracts),
                    s.success_est, s.duration_est))
        return tuple(out)

    def structure(self, skills=(), target_fact=None, include_submit=True):
        return build_structure(self.spec, self.graph_mode, self.topo_snapshot, self.use_posterior,
                               skills=skills, target_fact=target_fact, include_submit=include_submit)

    # ---------------------------------------------------------- primitive
    def primitive(self, a: int) -> float:
        self.meter.charge(1, self.purpose)
        desc = self.spec.actions[a]
        before = self.obs.present
        obs, r, term, _, info = self.env.step(a)
        if self.topo is not None and desc.kind == CRAFT:
            self.topo.record(self.spec, desc.rule, before, obs.present)
        if desc.kind != WAIT and not info["changed"] and not (desc.kind == SUBMIT and r > 0):
            self.n_noop += 1
        self.obs = obs
        self.n_primitive += 1
        self.rewards.append(r)
        if term:
            self.terminated = True
            self.termination = info["termination"]
            self.success = info["termination"] == "success"
        return r

    # -------------------------------------------------------------- skill
    def run_skill(self, cand: SkillCandidate, depth: int, limit: int | None = None) -> OptionOutcome:
        spec = self.lib_snapshot.get((cand.skill_key, cand.version))
        start = self.n_primitive
        status = None
        if depth > self.max_depth:
            status = "depth_exceeded"
        elif not binding_valid(spec, self.spec, cand.target_fact):
            status = "invalid_binding"
        elif self.obs.present[cand.target_fact]:
            status = "already_satisfied"
        if status is not None:
            self.primitive(self.wait_index)
        elif spec.macro:
            status = self._run_macro(spec, cand.target_fact, limit)
        else:
            controller = self.library.controller(spec.ref)
            children = self.skill_candidates(refs=spec.children)
            struct = self.structure(skills=children, target_fact=cand.target_fact, include_submit=False)
            lim = spec.timeout if limit is None else min(spec.timeout, limit)
            tr = self.run_decisions(controller, struct, target=cand.target_fact, limit=lim,
                                    depth=depth, greedy=self.greedy_skills)
            status = tr.status
        tau = self.n_primitive - start
        self.call_log.append({"depth": depth, "ref": [spec.skill_key, spec.version],
                              "target": TYPE_NAMES[spec.target_type], "status": status, "tau": tau,
                              "t_start": start})
        return OptionOutcome(status, tau, self.rewards[start:], self.terminated)

    def _run_macro(self, spec: SkillSpec, target: int, limit: int | None) -> str:
        """Open-loop recorded sequence of (choice kind, target type) steps.

        A pure macro ignores observations. ``spec.macro_retries > 0`` is the
        explicitly named retry wrapper: it replays the sequence while the
        public target is absent, up to that many extra times."""
        by_key = {}
        for i, a in enumerate(self.spec.actions):
            t = (self.spec.facts[a.fact].type_id if a.fact is not None else
                 self.spec.facts[self.spec.rules[a.rule].effect].type_id if a.rule is not None else None)
            by_key.setdefault((a.kind, t), []).append(i)
        lim = spec.timeout if limit is None else min(spec.timeout, limit)
        used = 0
        for attempt in range(1 + spec.macro_retries):
            if attempt and self.obs.present[target]:
                break
            for kind, t in spec.macro:
                if used >= lim:
                    return "timeout"
                for a in by_key.get((kind, t), [self.wait_index]):
                    if used >= lim or self.terminated:
                        break
                    self.primitive(a)
                    used += 1
                if self.terminated:
                    return self.termination
        return "target_success" if self.obs.present[target] else "macro_end"

    # --------------------------------------------------------- decisions
    def run_decisions(self, policy, struct, *, target: int | None = None, limit: int | None = None,
                      depth: int = 0, record: bool = False, greedy: bool = False,
                      gamma: float = 1.0) -> DecisionTrace:
        """Decision loop shared by the root agent, practice episodes and skills.

        ``target`` set: subgoal mode (stop and reward 1 when the target fact
        holds). ``limit``: primitive-step timeout. Budget exhaustion is caught
        at depth 0 only and ends the trace as an administrative cutoff.
        """
        tr = DecisionTrace()
        h = policy.initial_state()
        prev = (None, None, None, 0.0, 0)
        cap = (limit if limit is not None else self.spec.budget) * self.dispatch_factor + 1
        dispatches = 0
        while True:
            if self.terminated:
                tr.status, tr.terminal = self.termination, True
                break
            if target is not None and self.obs.present[target]:
                tr.status, tr.terminal = "target_success", True
                break
            if limit is not None and tr.tau_total >= limit:
                tr.status, tr.terminal = "timeout", True
                break
            dispatches += 1
            if dispatches > cap:
                tr.status, tr.terminal = "dispatch_limit", True
                break
            present = torch.tensor(self.obs.present, dtype=torch.float32)
            mem = torch.tensor(memory_features(self.spec, self.obs, struct.goal_fact, *prev),
                               dtype=torch.float32)
            with torch.no_grad():
                logits, value, h_new, alpha = policy.step(struct, present, mem, h)
                logp_all = torch.log_softmax(logits, -1)
                if greedy:
                    a = int(torch.argmax(logits))
                else:
                    a = int(torch.multinomial(logp_all.exp(), 1, generator=self.gen))
            choice = struct.cand_choices[a]
            if record:
                tr.present.append(present)
                tr.mem.append(mem)
                tr.actions.append(a)
                tr.logp.append(float(logp_all[a]))
                tr.values.append(float(value))
                tr.choice_keys.append(struct.cand_keys[a])
            if alpha is not None:
                tr.alpha_means.append(float(alpha.mean()))
            start = self.n_primitive
            before = self.obs.present
            try:
                if choice[0] == "prim":
                    if not self.meter.can_charge(1):
                        self.meter.stop_reason = "interaction_cap"
                        raise BudgetExhausted("no budget for the next primitive")
                    self.primitive(choice[1])
                    kind = self.spec.actions[choice[1]].kind
                    d = self.spec.actions[choice[1]]
                    tgt_fact = d.fact if d.fact is not None else (
                        self.spec.rules[d.rule].effect if d.rule is not None else None)
                else:
                    tr.n_skill_calls += 1
                    remaining = None if limit is None else limit - tr.tau_total
                    self.run_skill(choice[1], depth + 1, remaining)
                    kind, tgt_fact = SKILL, choice[1].target_fact
            except BudgetExhausted:
                if depth > 0:
                    raise
                tau = self.n_primitive - start
                if record:
                    if tau == 0:  # never executed: drop it
                        for lst in (tr.present, tr.mem, tr.actions, tr.logp, tr.values, tr.choice_keys):
                            lst.pop()
                    else:  # partial option: keep its cost, exclude it from the update
                        tr.R.append(option_return(self.rewards[start:], gamma))
                        tr.tau.append(tau)
                        tr.valid.append(False)
                tr.tau_total += tau
                tr.final_value = float(value)
                tr.status, tr.terminal = "budget_exhausted", False
                break
            tau = self.n_primitive - start
            if target is not None:
                R = 1.0 if self.obs.present[target] else 0.0
            else:
                R = option_return(self.rewards[start:], gamma)
            tr.tau_total += tau
            if record:
                tr.R.append(R)
                tr.tau.append(tau)
                tr.valid.append(True)
            changed = self.obs.present != before
            ttype = self.spec.facts[tgt_fact].type_id if tgt_fact is not None else None
            if record:
                new = [self.spec.facts[i].type_id for i, (b, a_) in
                       enumerate(zip(before, self.obs.present)) if a_ and not b]
                tr.events.append({
                    "kind": kind, "ttype": ttype, "new": new, "tau": tau,
                    "skill": ([choice[1].skill_key, choice[1].version] if choice[0] == "skill" else None),
                    "skill_level": choice[1].level if choice[0] == "skill" else 0,
                    "t_start": start})
            prev = (kind, ttype, changed, R, tau)
            h = h_new
        return tr
