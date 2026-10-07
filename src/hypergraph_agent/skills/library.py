"""Versioned skill library with an acyclic call hierarchy.

* Admission creates an immutable ``(skill_key, version)`` entry with a frozen
  copy of its controller parameters. Nothing is overwritten in place.
* Candidate records (proposed, training, admitted, rejected) keep ids and
  reasons forever; a revision is a new candidate pointing to its predecessor.
* A level-l skill may call primitives and strictly lower-level skills, pinned
  by version. Missing, cyclic or level-violating references are rejected on
  admission and on load.
* Retiring a skill marks every available parent that pins it as retired too
  (no silent dangling calls); old versions stay readable.
* ``snapshot()`` returns the immutable view used for one rollout/update batch.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import torch

from ..agents.policy import ActorCritic
from ..envs.generator import stable_hash
from .spec import SkillSpec, controller_hash


class LibraryError(ValueError):
    pass


@dataclass(frozen=True)
class LibrarySnapshot:
    version: int
    skills: tuple  # available SkillSpec entries
    library_id: str

    def get(self, ref) -> SkillSpec:
        for s in self.skills:
            if s.ref == tuple(ref):
                return s
        raise KeyError(ref)


@dataclass
class CandidateRecord:
    cand_id: str
    skill_key: str
    level: int
    target_type: int
    children: tuple
    created_round: int
    evidence: dict
    status: str = "proposed"
    reason: str = ""
    practice_cost: int = 0
    validation_cost: int = 0
    bc_updates: int = 0
    validation: dict = field(default_factory=dict)
    revision_of: str | None = None
    admitted_ref: tuple | None = None


class SkillLibrary:
    def __init__(self, max_admitted: int = 24, max_depth: int = 3):
        self.max_admitted = max_admitted
        self.max_depth = max_depth
        self.specs: dict[tuple, SkillSpec] = {}
        self.controllers: dict[tuple, dict] = {}
        self.controller_configs: dict[tuple, dict] = {}
        self.candidates: dict[str, CandidateRecord] = {}
        self.version = 0
        self.events: list[dict] = []
        self._cache: dict[tuple, ActorCritic] = {}

    # ---------------------------------------------------------- candidates
    def propose(self, skill_key: str, level: int, target_type: int, children: tuple,
                created_round: int, evidence: dict, revision_of: str | None = None) -> CandidateRecord:
        cand_id = f"cand{len(self.candidates):04d}"
        rec = CandidateRecord(cand_id, skill_key, level, target_type, tuple(children),
                              created_round, evidence, revision_of=revision_of)
        self.candidates[cand_id] = rec
        self.events.append({"type": "skill_proposed", "cand_id": cand_id, "skill_key": skill_key,
                            "level": level, "children": [list(c) for c in children],
                            "evidence": evidence, "round": created_round})
        return rec

    def defer(self, cand_id: str, reason: str):
        """Not trained in this round (cap); may be proposed again later."""
        rec = self.candidates[cand_id]
        rec.status = "deferred"
        rec.reason = reason
        self.events.append({"type": "skill_deferred", "cand_id": cand_id, "skill_key": rec.skill_key,
                            "reason": reason})

    def reject(self, cand_id: str, reason: str, validation: dict | None = None):
        rec = self.candidates[cand_id]
        rec.status = "rejected"
        rec.reason = reason
        if validation is not None:
            rec.validation = validation
        self.events.append({"type": "skill_rejected", "cand_id": cand_id, "skill_key": rec.skill_key,
                            "reason": reason, "validation": rec.validation})

    # ----------------------------------------------------------- admission
    def available(self) -> list[SkillSpec]:
        return [s for s in self.specs.values() if s.status == "admitted"]

    def next_version(self, skill_key: str) -> int:
        return 1 + max((v for k, v in self.specs if k == skill_key), default=0)

    def _check(self, spec: SkillSpec, pool: dict):
        if spec.level < 1 or spec.level > self.max_depth:
            raise LibraryError(f"level {spec.level} outside 1..{self.max_depth}")
        for child in spec.children:
            child = tuple(child)
            if child not in pool:
                raise LibraryError(f"{spec.ref} pins missing child {child}")
            if pool[child].level >= spec.level:
                raise LibraryError(f"{spec.ref} calls {child} at a level that is not lower")
        # levels strictly decrease along every call, so the call graph is acyclic

    def admit(self, cand_id: str, spec: SkillSpec, controller_state: dict,
              controller_config: dict) -> LibrarySnapshot:
        if spec.ref in self.specs:
            raise LibraryError(f"{spec.ref} already exists; create a new version")
        if len(self.available()) >= self.max_admitted:
            raise LibraryError("admission cap reached")
        avail = {s.ref: s for s in self.available()}
        self._check(spec, avail)
        frozen = {k: v.detach().clone() for k, v in controller_state.items()}
        if controller_hash(frozen) != spec.controller_ref:
            raise LibraryError("controller_ref does not match the stored parameters")
        self.specs[spec.ref] = spec
        self.controllers[spec.ref] = frozen
        self.controller_configs[spec.ref] = dict(controller_config)
        rec = self.candidates[cand_id]
        rec.status = "admitted"
        rec.admitted_ref = spec.ref
        self.version += 1
        snap = self.snapshot()
        self.events.append({"type": "skill_admitted", "cand_id": cand_id, "ref": list(spec.ref),
                            "level": spec.level, "children": [list(c) for c in spec.children],
                            "profile": spec.admission_profile, "library_version": self.version,
                            "library_id": snap.library_id, "controller_ref": spec.controller_ref})
        return snap

    def retire(self, ref: tuple, reason: str):
        ref = tuple(ref)
        if ref not in self.specs:
            raise KeyError(ref)
        todo = [ref]
        while todo:
            r = todo.pop()
            if self.specs[r].status != "admitted":
                continue
            self.specs[r] = self.specs[r].with_status("retired")
            self.events.append({"type": "skill_retired", "ref": list(r),
                                "reason": reason if r == ref else f"pins retired {list(ref)}"})
            todo += [p.ref for p in self.available() if r in [tuple(c) for c in p.children]]
        self.version += 1

    # ------------------------------------------------------------ queries
    def snapshot(self) -> LibrarySnapshot:
        skills = tuple(sorted(self.available(), key=lambda s: s.ref))
        lid = "L" + stable_hash([self.version, [(s.ref, s.controller_ref) for s in skills]])
        return LibrarySnapshot(self.version, skills, lid)

    def controller(self, ref: tuple) -> ActorCritic:
        ref = tuple(ref)
        if ref not in self._cache:
            net = ActorCritic(**self.controller_configs[ref])
            net.load_state_dict(self.controllers[ref])
            for p in net.parameters():
                p.requires_grad_(False)
            self._cache[ref] = net
        return self._cache[ref]

    def validate(self):
        pool = dict(self.specs)
        for spec in self.specs.values():
            self._check(spec, pool)

    # -------------------------------------------------------- persistence
    def save(self, directory: str | Path):
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        meta = {
            "version": self.version, "max_admitted": self.max_admitted, "max_depth": self.max_depth,
            "specs": [s.to_dict() for s in self.specs.values()],
            "configs": [[list(r), c] for r, c in self.controller_configs.items()],
            "candidates": [asdict(c) for c in self.candidates.values()],
        }
        (d / "library.json").write_text(json.dumps(meta, indent=1, default=list))
        torch.save({"|".join(map(str, r)): sd for r, sd in self.controllers.items()}, d / "controllers.pt")

    @staticmethod
    def load(directory: str | Path) -> "SkillLibrary":
        d = Path(directory)
        meta = json.loads((d / "library.json").read_text())
        lib = SkillLibrary(meta["max_admitted"], meta["max_depth"])
        lib.version = meta["version"]
        for sd in meta["specs"]:
            s = SkillSpec.from_dict(sd)
            lib.specs[s.ref] = s
        for r, c in meta["configs"]:
            lib.controller_configs[(r[0], int(r[1]))] = c
        raw = torch.load(d / "controllers.pt", weights_only=True)
        for k, sd in raw.items():
            key, ver = k.rsplit("|", 1)
            lib.controllers[(key, int(ver))] = sd
        for cd in meta["candidates"]:
            cd["children"] = tuple(tuple(c) for c in cd["children"])
            if cd.get("admitted_ref") is not None:
                cd["admitted_ref"] = tuple(cd["admitted_ref"])
            lib.candidates[cd["cand_id"]] = CandidateRecord(**cd)
        for ref, spec in lib.specs.items():
            if controller_hash(lib.controllers[ref]) != spec.controller_ref:
                raise LibraryError(f"stored controller for {ref} does not match its spec")
        lib.validate()
        return lib
