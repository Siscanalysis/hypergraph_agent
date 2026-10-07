"""Interaction-derived skill discovery (algorithm-designed, data-driven).

Pipeline for one round, all on training/development tasks and public data:

1. ``mine_fragments``: windows of the agent's own decisions (2-6 primitive
   steps, or up to three skill calls) ending where an item fact newly
   appeared; failed craft attempts are mined as attempt evidence.
2. ``propose``: canonicalize by (target item type, level, child skills),
   deduplicate deterministically, rank by observed achievements (a proposal
   heuristic, not a competence measure), cap per round.
3. ``behaviour_clone`` (optional, logged): supervised initialization on the
   agent's own successful fragments; counted as optimizer updates, no
   interactions.
4. ``practice``: PPO on a public target predicate (reward 1 when the target
   fact holds) in bounded practice episodes; every step is charged as
   ``skill_practice``.
5. ``validate``: separate ``skillval`` tasks with other inventories and
   bindings; charged as ``candidate_validation``.
6. ``admit`` under fixed criteria or ``reject`` with a reason.

The proposer is heuristic. The learned parts are the controllers (and the
manager that later selects them). Initiation and termination are specified
public predicates, not learned.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import torch

from ..agents.policy import ActorCritic
from ..envs.generator import TaskConfig, derive_seed, make_task
from ..envs.public_schema import CRAFT
from ..envs.recipequest import RecipeQuestEnv
from ..envs.vocabulary import TYPE_NAMES, is_base
from ..representations.features import build_structure
from ..training.ppo import PPOConfig, ppo_update
from ..training.rollout import run_practice_episode
from .executor import Executor, contract_support
from .library import SkillLibrary
from .spec import SkillSpec, controller_hash, skill_key_for


@dataclass(frozen=True)
class Fragment:
    source: str
    target_type: int
    level: int
    children: tuple
    steps: tuple  # ((choice kind, target type), ...)
    success: bool
    record: int  # index into the mined record list
    decisions: tuple  # decision indices of the window


def mine_fragments(records, min_primitives: int = 2, max_len: int = 6, max_children: int = 3,
                   max_level: int = 3) -> list[Fragment]:
    out = []
    for ri, rec in enumerate(records):
        boundary = -1
        for k, e in enumerate(rec.trace):
            new_items = [t for t in e["new"] if not is_base(t)]
            failed = e["kind"] == CRAFT and not e["new"]
            if not new_items and not failed:
                continue
            lo = max(boundary + 1, k - max_len + 1)
            window = rec.trace[lo:k + 1]
            children = tuple(sorted({tuple(w["skill"]) for w in window if w["skill"]}))[:max_children]
            level = min(max_level, 1 + max((w["skill_level"] for w in window), default=0))
            prims = sum(w["tau"] for w in window)
            steps = tuple((w["kind"], w["ttype"]) for w in window)
            if prims >= min_primitives:
                for t in (new_items or [e["ttype"]]):
                    out.append(Fragment(f"{rec.task_key}:{k}", t, level, children, steps,
                                        bool(new_items), ri, tuple(range(lo, k + 1))))
            if new_items:
                boundary = k
    return out


def propose(fragments: list[Fragment], library: SkillLibrary, round_idx: int,
            max_proposals: int = 8) -> list:
    groups: dict[tuple, dict] = {}
    for f in fragments:
        g = groups.setdefault((f.target_type, f.level, f.children),
                              {"successes": 0, "failures": 0, "sources": []})
        g["successes" if f.success else "failures"] += 1
        if f.success and len(g["sources"]) < 16:
            g["sources"].append(f.source)
    deferred = {c.skill_key: c for c in library.candidates.values() if c.status == "deferred"}
    seen = {c.skill_key for c in library.candidates.values() if c.status != "deferred"}
    ranked = sorted(
        ((k, g) for k, g in groups.items() if g["successes"] > 0),
        key=lambda kg: (-kg[1]["successes"], -kg[1]["failures"], kg[0][1], TYPE_NAMES[kg[0][0]],
                        kg[0][2]))
    out = []
    for (target, level, children), g in ranked:
        key = skill_key_for(target, level)
        if key in seen:
            continue  # one live line of development per canonical skill in this version
        seen.add(key)
        evidence = {"successful_fragments": g["successes"], "failed_attempts": g["failures"],
                    "sources": g["sources"]}
        if key in deferred:  # reuse the record (and its id) of a deferred candidate
            rec = deferred[key]
            rec.status, rec.evidence, rec.children = "proposed", evidence, children
            library.events.append({"type": "skill_reproposed", "cand_id": rec.cand_id,
                                   "skill_key": key, "round": round_idx, "evidence": evidence})
            out.append(rec)
        else:
            out.append(library.propose(key, level, target, children, round_idx, evidence))
        if len(out) >= max_proposals:
            break
    return out


def fragments_for(cand, fragments):
    return [f for f in fragments if f.success and f.target_type == cand.target_type
            and f.level == cand.level and f.children == cand.children]


def _controller_struct(spec, cand, ex_factory, graph_mode, topo_snapshot, use_posterior, lib_snapshot):
    by_type = spec.fact_by_type()
    if cand.target_type not in by_type:
        return None
    ex = ex_factory()
    ex.spec = spec
    ex.topo_snapshot = topo_snapshot
    children = ex.skill_candidates(refs=cand.children) if lib_snapshot is not None else ()
    return build_structure(spec, graph_mode, topo_snapshot, use_posterior, skills=children,
                           target_fact=by_type[cand.target_type], include_submit=False)


def behaviour_clone(controller, cand, fragments, records, ex_factory, graph_mode, topo,
                    use_posterior, lib_snapshot, epochs: int = 20, lr: float = 1e-3) -> int:
    """Supervised initialization on the agent's own successful fragments.
    Returns the number of optimizer updates (logged as additional optimization)."""
    data = []
    for f in fragments_for(cand, fragments):
        rec = records[f.record]
        snap = topo.snapshot(rec.world_key) if topo is not None else None
        struct = _controller_struct(rec.spec, cand, ex_factory, graph_mode, snap, use_posterior,
                                    lib_snapshot)
        if struct is None:
            continue
        idx = list(f.decisions)
        keys = [rec.choice_keys[d] for d in idx]
        if not all(k in struct.cand_keys for k in keys):
            continue
        target = rec.spec.fact_by_type()[cand.target_type]
        mem = rec.mem[idx].clone()
        mem[:, 3] = rec.present[idx, target]  # goal-present feature refers to the skill target
        acts = torch.tensor([struct.cand_keys.index(k) for k in keys])
        data.append((struct, rec.present[idx], mem, acts))
    if not data:
        return 0
    opt = torch.optim.Adam(controller.parameters(), lr=lr)
    updates = 0
    for _ in range(epochs):
        loss = 0.0
        for struct, present, mem, acts in data:
            logits, _, _, _ = controller.forward_sequence(struct, present, mem)
            loss = loss + torch.nn.functional.cross_entropy(logits, acts)
        opt.zero_grad()
        (loss / len(data)).backward()
        torch.nn.utils.clip_grad_norm_(controller.parameters(), 0.5)
        opt.step()
        updates += 1
    return updates


class TargetTasks:
    """Tasks whose goal is the candidate's target type, in a declared namespace
    (``practice`` for training, ``skillval`` for validation; never ``test``)."""

    def __init__(self, world, task_cfg: TaskConfig, namespace: str, base_seed: int):
        if namespace == "test":
            raise ValueError("skill admission must not use test tasks")
        self.world, self.cfg, self.namespace, self.base_seed = world, task_cfg, namespace, base_seed

    def task(self, target_type: int, i: int):
        seed = derive_seed("target_task", self.base_seed, self.namespace, target_type, i)
        return make_task(self.world, seed, self.cfg, self.namespace, goal=target_type)


def practice(controller, cand, tasks: TargetTasks, meter, streams, topo, graph_mode, use_posterior,
             library, lib_snapshot, live_contracts, timeout: int, budget: int, batch_steps: int,
             ppo: PPOConfig, log) -> dict:
    env = RecipeQuestEnv()
    opt = torch.optim.Adam(controller.parameters(), lr=ppo.lr)
    start, i, batches, history = meter.used, 0, 0, []
    while meter.used - start < budget and meter.remaining > 0:
        records, steps = [], 0
        while steps < batch_steps and meter.used - start < budget and meter.remaining > 0:
            task = tasks.task(cand.target_type, i)
            i += 1
            snap = topo.snapshot(task.world.world_key) if topo is not None else None
            ex = Executor(env, meter, "skill_practice", streams.actions, topo=topo, topo_snapshot=snap,
                          graph_mode=graph_mode, use_posterior=use_posterior, library=library,
                          lib_snapshot=lib_snapshot, live_contracts=live_contracts)
            lim = min(timeout, budget - (meter.used - start))
            rec = run_practice_episode(controller, ex, task, streams.next_seed("dynamics"),
                                       cand.target_type, cand.children, lim)
            if rec is None:
                continue
            records.append(rec)
            steps += rec.n_primitive
            if rec.status == "budget_exhausted":
                break
        if not records:
            break
        expected = lambda r: ((topo.snapshot(r.world_key).snapshot_id if topo else None),
                              lib_snapshot.library_id if lib_snapshot is not None else None)
        stats = ppo_update(controller, opt, records, ppo, streams.np["minibatch"], expected)
        rate = float(np.mean([r.success for r in records]))
        history.append(rate)
        log({"type": "skill_practice_batch", "cand_id": cand.cand_id, "batch": batches,
             "steps": steps, "success_rate": rate, "episodes": len(records),
             "ppo_updates": stats.get("n_updates", 0)})
        batches += 1
    return {"interactions": meter.used - start, "batches": batches, "success_history": history}


def validate(controller, cand, tasks: TargetTasks, meter, streams, topo, graph_mode, use_posterior,
             library, lib_snapshot, live_contracts, timeout: int, n_episodes: int) -> dict:
    env = RecipeQuestEnv()
    start = meter.used
    eps, i = [], 0
    while len(eps) < n_episodes and meter.remaining >= timeout and i < 10 * n_episodes:
        task = tasks.task(cand.target_type, i)
        i += 1
        snap = topo.snapshot(task.world.world_key) if topo is not None else None
        ex = Executor(env, meter, "candidate_validation", streams.actions, topo=None,
                      topo_snapshot=snap, graph_mode=graph_mode, use_posterior=use_posterior,
                      library=library, lib_snapshot=lib_snapshot, live_contracts=live_contracts)
        rec = run_practice_episode(controller, ex, task, streams.next_seed("validation"),
                                   cand.target_type, cand.children, timeout, record=False)
        if rec is None:
            continue
        eps.append({"task_key": rec.task_key, "success": rec.success, "tau": rec.n_primitive,
                    "status": rec.status, "noop": rec.n_noop, "skill_calls": rec.n_skill_calls})
    n = len(eps)
    succ = [e for e in eps if e["success"]]
    return {
        "n": n, "successes": len(succ), "success_rate": len(succ) / n if n else 0.0,
        "mean_tau": float(np.mean([e["tau"] for e in eps])) if eps else None,
        "mean_tau_success": float(np.mean([e["tau"] for e in succ])) if succ else None,
        "interactions": meter.used - start, "episodes": eps,
    }


def macro_steps(cand, fragments) -> tuple | None:
    """Most frequent successful primitive-only fragment (matched-macro control)."""
    from collections import Counter
    from ..envs.public_schema import SKILL
    seqs = Counter(f.steps for f in fragments_for(cand, fragments)
                   if all(k != SKILL for k, _ in f.steps))
    if not seqs:
        return None
    return sorted(seqs.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def validate_macro(spec: SkillSpec, tasks: TargetTasks, meter, streams, topo, graph_mode: str,
                   n_episodes: int) -> dict:
    from ..representations.features import SkillCandidate
    from .library import LibrarySnapshot
    env = RecipeQuestEnv()
    snap_lib = LibrarySnapshot(0, (spec,), "macro-validation")
    start, eps, i = meter.used, [], 0
    while len(eps) < n_episodes and meter.remaining >= spec.timeout and i < 10 * n_episodes:
        task = tasks.task(spec.target_type, i)
        i += 1
        snap = topo.snapshot(task.world.world_key) if topo is not None else None
        ex = Executor(env, meter, "candidate_validation", streams.actions, topo_snapshot=snap,
                      graph_mode=graph_mode, library=SkillLibrary(), lib_snapshot=snap_lib)
        ex.reset(task, streams.next_seed("validation"))
        target = ex.spec.fact_by_type().get(spec.target_type)
        if target is None or ex.obs.present[target]:
            continue
        cand = SkillCandidate(spec.skill_key, spec.version, spec.level, target, (), 0.0, 0.0)
        out = ex.run_skill(cand, depth=1)
        ok = bool(ex.obs.present[target])
        eps.append({"task_key": task.task_key, "success": ok, "tau": out.tau, "status": out.status})
    n, succ = len(eps), [e for e in eps if e["success"]]
    return {"n": n, "successes": len(succ), "success_rate": len(succ) / n if n else 0.0,
            "mean_tau": float(np.mean([e["tau"] for e in eps])) if eps else None,
            "mean_tau_success": float(np.mean([e["tau"] for e in succ])) if succ else None,
            "interactions": meter.used - start, "episodes": eps}


def admission_decision(val: dict, criteria: dict) -> tuple[bool, str]:
    if val["n"] < criteria["min_episodes"]:
        return False, f"insufficient validation coverage ({val['n']} < {criteria['min_episodes']})"
    if val["success_rate"] < criteria["min_success"]:
        return False, f"success {val['success_rate']:.2f} below {criteria['min_success']}"
    if (val["mean_tau_success"] or 0) < criteria["min_primitive_steps"]:
        return False, "trivial competence (fewer primitive steps than required)"
    return True, "meets declared criteria"


def build_spec(cand, controller, library: SkillLibrary, timeout: int, val: dict, practice_stats: dict,
               topo_snapshot, spec_for_contract, profile: str) -> SkillSpec:
    state = controller.state_dict()
    contract = ()
    if spec_for_contract is not None:
        probe = SkillSpec(cand.skill_key, 0, cand.level, cand.target_type, timeout, cand.children,
                          "", (), 0.0, 0.0, "{}", 0, "{}")
        support = contract_support(probe, spec_for_contract, topo_snapshot, live=True)
        contract = tuple((spec_for_contract.facts[i].type_id, w) for i, w in support)
    return SkillSpec(
        skill_key=cand.skill_key, version=library.next_version(cand.skill_key), level=cand.level,
        target_type=cand.target_type, timeout=timeout,
        children=tuple(tuple(c) for c in cand.children), controller_ref=controller_hash(state),
        contract_support=contract, success_est=val["success_rate"],
        duration_est=val["mean_tau_success"] or float(timeout),
        provenance=json.dumps({"cand_id": cand.cand_id, "round": cand.created_round,
                               "evidence": cand.evidence, "bc_updates": cand.bc_updates,
                               "proposer": "heuristic_fragment_mining"}),
        training_cost=practice_stats["interactions"] + val["interactions"],
        validation=json.dumps({k: v for k, v in val.items() if k != "episodes"}),
        admission_profile=profile,
    )
