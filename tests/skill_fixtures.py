"""Scripted skill fixtures (``scripted_fixture``): they verify executor
mechanics and are never reported as learned skills."""

import torch

from hypergraph_agent.agents.policy import ActorCritic
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.skills.library import SkillLibrary
from hypergraph_agent.skills.spec import SkillSpec, controller_hash, skill_key_for

from helpers import T, ScriptedPolicy, meter


def admit_scripted(lib, target, level, children=(), timeout=8, prefs=(), macro=(), retries=0):
    key = skill_key_for(T[target], level)
    cand = lib.propose(key, level, T[target], tuple(children), 0, {"fixture": "scripted"})
    torch.manual_seed(0)
    net = ActorCritic("incidence", d=16, hidden=8)
    state = net.state_dict()
    spec = SkillSpec(key, lib.next_version(key), level, T[target], timeout, tuple(children),
                     controller_hash(state), (), 0.5, float(timeout), '{"label": "scripted_fixture"}',
                     0, "{}", macro=tuple(macro), macro_retries=retries)
    lib.admit(cand.cand_id, spec, state, net.config)
    lib._cache[spec.ref] = ScriptedPolicy(prefs)  # replace the controller by the scripted fixture
    return spec


def ingot_key_library(timeout_ingot=8, timeout_key=12):
    lib = SkillLibrary()
    ingot = admit_scripted(lib, "ingot", 1, timeout=timeout_ingot,
                           prefs=["gather:fore", "gather:ffuel", "activate:ffurnace_ready", "craft:r0_"])
    key = admit_scripted(lib, "key", 2, children=[ingot.ref], timeout=timeout_key,
                         prefs=["skill:achieve[ingot]", "activate:fmould_ready", "craft:r1_"])
    return lib, ingot, key


def executor(lib, cap=1000, max_depth=3, graph_mode="known", seed=0):
    m = meter(cap)
    ex = Executor(RecipeQuestEnv(), m, "exploration", torch.Generator().manual_seed(seed),
                  graph_mode=graph_mode, library=lib, lib_snapshot=lib.snapshot(), max_depth=max_depth)
    return ex, m


def cand_for(ex, target):
    return next(c for c in ex.skill_candidates() if ex.spec.facts[c.target_fact].type_id == T[target])
