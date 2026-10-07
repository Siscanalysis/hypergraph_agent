"""Skill specifications.

A skill is an executable closed-loop controller with initiation semantics, an
internal policy and termination semantics. Predicates and control flow are
stored as structured data, never as code evaluated at run time.

Declared semantics of the first implementation (not learned):
* argument role: ``target``, the unique fact of item type ``target_type`` in
  the current task (bound by the public binder, never by task-local ids);
* initiation ``public_target_absent``: a call whose target already holds
  executes one ``wait`` primitive and returns ``already_satisfied``;
* termination ``target_present_or_timeout``: the public target predicate, the
  primitive-step timeout, or a task terminal.
The internal policy (controller) is learned; initiation and termination are
specified from public predicates.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, replace

from ..envs.vocabulary import TYPE_NAMES

STATUSES = ("proposed", "training", "admitted", "rejected", "retired")
PROFILES = ("diagnostic", "study")


def skill_key_for(target_type: int, level: int) -> str:
    return f"achieve[{TYPE_NAMES[target_type]}]/L{level}"


@dataclass(frozen=True)
class SkillSpec:
    skill_key: str
    version: int
    level: int
    target_type: int
    timeout: int
    children: tuple  # pinned (skill_key, version) of lower-level skills
    controller_ref: str  # hash of the frozen controller parameters
    contract_support: tuple  # ((type id, weight), ...): predicted prerequisites
    success_est: float
    duration_est: float
    provenance: str  # JSON: candidate id, evidence, data sources
    training_cost: int  # primitive interactions spent on practice
    validation: str  # JSON: validation record summary
    status: str = "admitted"
    admission_profile: str = "diagnostic"
    arg_roles: tuple = (("target", "item"),)
    target_predicate: tuple = ("present", "target")
    initiation: str = "public_target_absent"
    termination: str = "target_present_or_timeout"
    # matched-macro control: open-loop (choice kind, target type) steps; empty
    # for learned closed-loop controllers
    macro: tuple = ()
    macro_retries: int = 0

    @property
    def ref(self) -> tuple[str, int]:
        return (self.skill_key, self.version)

    @property
    def provisional(self) -> bool:
        return self.admission_profile == "diagnostic"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["children"] = [list(c) for c in self.children]
        d["contract_support"] = [list(c) for c in self.contract_support]
        d["macro"] = [list(m) for m in self.macro]
        return d

    @staticmethod
    def from_dict(d: dict) -> "SkillSpec":
        d = dict(d)
        d["children"] = tuple(tuple(c) for c in d["children"])
        d["contract_support"] = tuple((int(t), float(w)) for t, w in d["contract_support"])
        d["arg_roles"] = tuple(tuple(r) for r in d["arg_roles"])
        d["target_predicate"] = tuple(d["target_predicate"])
        d["macro"] = tuple((int(k), int(t)) for k, t in d.get("macro", ()))
        return SkillSpec(**d)

    def with_status(self, status: str) -> "SkillSpec":
        if status not in STATUSES:
            raise ValueError(status)
        return replace(self, status=status)


def controller_hash(state_dict) -> str:
    h = hashlib.sha256()
    for k, v in sorted(state_dict.items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return "C" + h.hexdigest()[:16]


def dumps(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=str)
