"""Public parameter binder.

Enumerates type-compatible role bindings from public fact records only. It
never consults solutions, reference plans or hidden prerequisites, and it
handles unseen task-local ids through the public type schema.
"""

from __future__ import annotations

from ..envs.public_schema import PublicTaskSpec
from ..envs.vocabulary import KIND_ITEM
from .spec import SkillSpec


def bindings(spec: SkillSpec, task: PublicTaskSpec, max_bindings: int = 4) -> list[int]:
    out = [i for i, f in enumerate(task.facts)
           if f.kind == KIND_ITEM and f.type_id == spec.target_type]
    return out[:max_bindings]


def binding_valid(spec: SkillSpec, task: PublicTaskSpec, fact: int) -> bool:
    if not 0 <= fact < len(task.facts):
        return False
    f = task.facts[fact]
    return f.kind == KIND_ITEM and f.type_id == spec.target_type
