"""Episode records for on-policy updates.

A record stores, for each decision, exactly what is needed to recompute the
behaviour likelihood: the immutable graph structure (candidate order included),
present flags and history features at the decision, the chosen candidate, its
behaviour log probability and value, the option return R_k and duration tau_k.
Topology and library snapshot ids are stored so an update can refuse data
collected under a different structure.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from ..envs.vocabulary import TYPE_NAMES
from ..representations.features import GraphStructure
from ..skills.executor import DecisionTrace, Executor
from .returns import segment_gae


@dataclass
class EpisodeRecord:
    task_key: str
    world_key: str
    namespace: str
    depth: int
    goal: str
    struct: GraphStructure
    present: torch.Tensor
    mem: torch.Tensor
    actions: torch.Tensor
    logp: torch.Tensor
    values: torch.Tensor
    R: list
    tau: list
    valid: list
    terminal: bool
    final_value: float
    success: bool
    status: str
    n_primitive: int
    n_noop: int
    n_skill_calls: int
    call_log: list
    alpha_means: list
    choice_keys: list
    snapshot_id: str | None = None
    library_id: str | None = None
    target: str | None = None
    spec: object = None  # PublicTaskSpec (public; used by discovery and behaviour cloning)
    trace: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    @property
    def n_decisions(self) -> int:
        return len(self.actions)

    def advantages(self, gamma: float, lam: float):
        k = len(self.valid)
        adv, tgt = np.zeros(k), np.zeros(k)
        mask = np.array(self.valid, dtype=bool)
        if k == 0 or not mask.any():
            return adv, tgt, mask
        if mask[-1]:
            n, final, terminal = k, self.final_value, self.terminal
        else:  # trailing incomplete option: bootstrap at its true start boundary
            n, final, terminal = k - 1, float(self.values[-1]), False
        a, t = segment_gae(self.R[:n], self.tau[:n], self.values[:n].tolist(), final, terminal, gamma, lam)
        adv[:n], tgt[:n] = a, t
        return adv, tgt, mask

    def summary(self) -> dict:
        return {
            "task_key": self.task_key, "world_key": self.world_key, "namespace": self.namespace,
            "depth": self.depth, "goal": self.goal, "target": self.target,
            "success": self.success, "status": self.status,
            "primitive_length": self.n_primitive, "manager_decisions": self.n_decisions,
            "noop_or_invalid": self.n_noop, "skill_calls": self.n_skill_calls,
            "incomplete_items": int(sum(1 for v in self.valid if not v)),
            "snapshot_id": self.snapshot_id, "library_id": self.library_id,
            **self.extra,
        }


def _stack(xs, dim_hint: int) -> torch.Tensor:
    return torch.stack(xs) if xs else torch.zeros(0, dim_hint)


def make_record(task, ex: Executor, struct: GraphStructure, tr: DecisionTrace, *, success: bool,
                target: str | None = None) -> EpisodeRecord:
    return EpisodeRecord(
        task_key=task.task_key, world_key=task.world.world_key, namespace=task.namespace,
        depth=task.depth, goal=TYPE_NAMES[task.goal], struct=struct,
        present=_stack(tr.present, struct.n_facts), mem=_stack(tr.mem, 1),
        actions=torch.tensor(tr.actions, dtype=torch.long),
        logp=torch.tensor(tr.logp, dtype=torch.float32),
        values=torch.tensor(tr.values, dtype=torch.float32),
        R=list(tr.R), tau=list(tr.tau), valid=list(tr.valid),
        terminal=tr.terminal, final_value=tr.final_value, success=success, status=tr.status,
        n_primitive=ex.n_primitive, n_noop=ex.n_noop, n_skill_calls=tr.n_skill_calls,
        call_log=list(ex.call_log), alpha_means=list(tr.alpha_means),
        choice_keys=list(tr.choice_keys),
        snapshot_id=struct.meta.get("snapshot_id"),
        library_id=ex.lib_snapshot.library_id if ex.lib_snapshot is not None else None,
        target=target, spec=ex.spec, trace=list(tr.events),
    )


def run_episode(policy, ex: Executor, task, dyn_seed: int, *, record: bool = True,
                greedy: bool = False, gamma: float = 1.0) -> EpisodeRecord:
    """One root-level episode on the task objective (P1/P2 agents, P3 manager)."""
    ex.reset(task, dyn_seed)
    struct = ex.structure(skills=ex.skill_candidates())
    tr = ex.run_decisions(policy, struct, record=record, greedy=greedy, gamma=gamma)
    return make_record(task, ex, struct, tr, success=ex.success)


def run_practice_episode(controller, ex: Executor, task, dyn_seed: int, target_type: int,
                         children: tuple, timeout: int, *, record: bool = True,
                         greedy: bool = False) -> EpisodeRecord | None:
    """One practice episode on a public target predicate (skill controller).

    Returns None, without consuming budget, if the target is absent from the
    task or already holds at reset.
    """
    ex.reset(task, dyn_seed)
    by_type = ex.spec.fact_by_type()
    if target_type not in by_type or ex.obs.present[by_type[target_type]]:
        return None
    target = by_type[target_type]
    struct = ex.structure(skills=ex.skill_candidates(refs=children), target_fact=target,
                          include_submit=False)
    tr = ex.run_decisions(controller, struct, target=target, limit=timeout, record=record, greedy=greedy)
    return make_record(task, ex, struct, tr, success=(tr.status == "target_success"),
                       target=TYPE_NAMES[target_type])
