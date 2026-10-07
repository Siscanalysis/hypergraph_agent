import json

import pytest
import torch

from hypergraph_agent.skills import discovery as disc
from hypergraph_agent.skills.library import LibraryError, SkillLibrary
from hypergraph_agent.skills.spec import SkillSpec, controller_hash, skill_key_for
from hypergraph_agent.training.rollout import run_episode

from helpers import CHAIN_FACTS, T, ScriptedPolicy, chain_world, task
from skill_fixtures import admit_scripted, executor, ingot_key_library


def test_admission_creates_immutable_versions():
    lib, ingot, key = ingot_key_library()
    assert lib.version == 2 and lib.next_version(ingot.skill_key) == 2
    with pytest.raises(LibraryError):
        cand = lib.propose(ingot.skill_key, 1, T["ingot"], (), 0, {})
        lib.admit(cand.cand_id, ingot, lib.controllers[ingot.ref], lib.controller_configs[ingot.ref])
    v2 = admit_scripted(lib, "ingot", 1, prefs=["wait"])
    assert v2.version == 2 and lib.specs[ingot.ref] is ingot  # the old version is untouched


def test_call_hierarchy_is_checked():
    lib = SkillLibrary()
    with pytest.raises(LibraryError):
        admit_scripted(lib, "key", 2, children=[("achieve[ingot]/L1", 1)])  # dangling child
    ingot = admit_scripted(lib, "ingot", 1)
    with pytest.raises(LibraryError):
        admit_scripted(lib, "key", 1, children=[ingot.ref])  # child not at a lower level
    with pytest.raises(LibraryError):
        admit_scripted(SkillLibrary(max_depth=1), "key", 2)


def test_retirement_cascades_to_dependent_parents():
    lib, ingot, key = ingot_key_library()
    lib.retire(ingot.ref, "superseded")
    assert lib.specs[ingot.ref].status == "retired" and lib.specs[key.ref].status == "retired"
    assert lib.snapshot().skills == ()
    assert lib.specs[key.ref].children == (ingot.ref,)  # still readable for old rollouts


def test_rejections_keep_ids_and_reasons(tmp_path):
    lib = SkillLibrary()
    c = lib.propose("achieve[lens]/L1", 1, T["lens"], (), 0, {})
    lib.reject(c.cand_id, "success 0.10 below 0.5")
    assert lib.candidates[c.cand_id].status == "rejected"
    assert any(e["type"] == "skill_rejected" and e["reason"].startswith("success") for e in lib.events)


def test_save_load_round_trip_and_tamper_detection(tmp_path):
    lib, ingot, key = ingot_key_library()
    lib.save(tmp_path / "lib")
    again = SkillLibrary.load(tmp_path / "lib")
    assert again.snapshot().library_id == lib.snapshot().library_id
    raw = torch.load(tmp_path / "lib" / "controllers.pt", weights_only=True)
    k = next(iter(raw))
    raw[k] = {n: t + 1 for n, t in raw[k].items()}
    torch.save(raw, tmp_path / "lib" / "controllers.pt")
    with pytest.raises(LibraryError):
        SkillLibrary.load(tmp_path / "lib")


def _scripted_records(n=3):
    recs = []
    for i in range(n):
        ex, _ = executor(SkillLibrary())
        p = f"p{i}"  # different opaque ids in every task
        pol = ScriptedPolicy([f"gather:{p}fore", f"gather:{p}ffuel", f"activate:{p}ffurnace_ready",
                              f"craft:{p}r0_", f"activate:{p}fmould_ready", f"craft:{p}r1_", "submit"])
        rec = run_episode(pol, ex, task(chain_world(), "key", CHAIN_FACTS, key=f"T{i}", prefix=p), i)
        assert rec.success
        recs.append(rec)
    return recs


def test_mining_is_deterministic_and_role_canonical():
    recs = _scripted_records()
    a, b = disc.mine_fragments(recs), disc.mine_fragments(recs)
    assert a == b and a
    assert {f.target_type for f in a if f.success} == {T["ingot"], T["key"]}
    lib = SkillLibrary()
    props = disc.propose(a, lib, 0)
    keys = [p.skill_key for p in props]
    assert keys == sorted(set(keys), key=keys.index)  # deduplicated
    assert set(keys) == {skill_key_for(T["ingot"], 1), skill_key_for(T["key"], 1)}
    assert all("p0" not in k and "f" + "ore" not in k for k in keys)  # no task-local ids
    assert disc.propose(a, lib, 1) == []  # nothing is proposed twice


def test_deferred_candidates_are_reproposed_with_the_same_id():
    recs = _scripted_records()
    frags = disc.mine_fragments(recs)
    lib = SkillLibrary()
    first = disc.propose(frags, lib, 0)
    lib.defer(first[0].cand_id, "cap")
    again = disc.propose(frags, lib, 1)
    assert [c.cand_id for c in again] == [first[0].cand_id]


def test_admission_criteria_and_test_namespace_guard():
    crit = {"min_episodes": 8, "min_success": 0.5, "min_primitive_steps": 2}
    assert disc.admission_decision({"n": 4, "success_rate": 1.0, "mean_tau_success": 4}, crit)[0] is False
    assert disc.admission_decision({"n": 10, "success_rate": 0.4, "mean_tau_success": 4}, crit)[0] is False
    assert disc.admission_decision({"n": 10, "success_rate": 0.9, "mean_tau_success": 1}, crit)[0] is False
    assert disc.admission_decision({"n": 10, "success_rate": 0.9, "mean_tau_success": 4}, crit)[0] is True
    with pytest.raises(ValueError):
        disc.TargetTasks(chain_world(), None, "test", 0)


def test_profile_marks_provisional_admissions():
    lib, ingot, _ = ingot_key_library()
    assert ingot.provisional and ingot.admission_profile == "diagnostic"
    assert json.loads(ingot.provenance)["label"] == "scripted_fixture"
