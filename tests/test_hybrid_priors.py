"""Studies HP1 and HP2 (docs/studies/HP1_size_prior.md, docs/studies/HP2_schema_priors.md).

Frozen behaviour first: with the new options off, every existing config loads to the same
dictionary, every frozen stream generates the same worlds and tasks, and every existing agent acts
exactly as before (digests computed with the code of commit 8b06e2a, before these studies; no file
they depend on changed up to 7dacf57). Then the schema option of the generator, the opt-in prior
hooks, the HP1 posterior and size priors, the HP2 rule learners and agents, the configs, the
analyses and the hygiene of the new files.
"""

import hashlib
import json
import math
import re
from dataclasses import replace
from itertools import product
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from helpers import T, meter, recipe, task, world

ROOT = Path(__file__).resolve().parents[1]
HP2_DIR = ROOT / "configs" / "hybrid_priors"
HP1_DIR = ROOT / "configs" / "hybrid_priors_offline"

# ============================================================== frozen behaviour (digests of 8b06e2a)

FROZEN_STREAM_CONFIGS = (
    "configs/replication/f0.yaml", "configs/replication/f1.yaml", "configs/replication/f2.yaml",
    "configs/markov/m1_collect.yaml", "configs/markov/m2.yaml",
    "configs/boundary/a.yaml", "configs/boundary/d.yaml", "configs/boundary/dh.yaml",
    "configs/boundary/i.yaml", "configs/boundary/p.yaml", "configs/boundary/f2_minus_p.yaml",
    "configs/policygraph_rq/pg_rq_main.yaml", "configs/walker_stage_d.yaml",
)
AGENT_CASES = (
    ("goal_only", 0.0, ("maximal", "local_uniform", "local_focused", "exact", "focused_sample",
                        "random_omit", "learned", "learned_sample", "hyper_rank", "pair_sample",
                        "pair_rank", "hyper_sample_ppr", "pg", "pg_uniform", "pg_greedy")),
    ("all", 0.0, ("maximal", "sample", "optimistic", "focused_sample")),
    ("goal_only", 0.1, ("focused_sample", "learned_sample")),
)
LOADED_CONFIGS = {
    "configs/ablations/p2_noisy.yaml": "8aad2f28be222aa9a75f35e2",
    "configs/ablations/p3_macro.yaml": "80d432fd8aa1b5bcee79f859",
    "configs/ablations/p3_unstructured.yaml": "69ae890d41865c52de2e6c94",
    "configs/adaptation_stream.yaml": "95a65958ec51e4b4d449895f",
    "configs/boundary/a.yaml": "cede31415101e50412f0a72d",
    "configs/boundary/d.yaml": "216692c7c44b65fdc7edb9f9",
    "configs/boundary/d_s12.yaml": "c3c941aa1731cdc4883f9482",
    "configs/boundary/d_s6.yaml": "b6157f86537823832a945772",
    "configs/boundary/dev_a.yaml": "9d52dc80d21795d3eaefcea3",
    "configs/boundary/dev_d.yaml": "b89c34837d4c3be49be6a96d",
    "configs/boundary/dev_d_s12.yaml": "ebfa89a7804b04237887af52",
    "configs/boundary/dev_d_s6.yaml": "50e8bd0663097335a232c3a0",
    "configs/boundary/dev_dh.yaml": "41354f6e040c4d35e3e0dd99",
    "configs/boundary/dev_f2.yaml": "443fedf4bd831656cc9a8c1a",
    "configs/boundary/dev_f2_minus_a.yaml": "06ba970a5c821a6e23d06c43",
    "configs/boundary/dev_f2_minus_d.yaml": "9c53d5db6e880245fe8965af",
    "configs/boundary/dev_f2_minus_p.yaml": "7ab473e570e8a74cb00d0240",
    "configs/boundary/dev_f2_s12.yaml": "166a790998ae2d4d71146e8a",
    "configs/boundary/dev_f2_s6.yaml": "a115d0b3f482c994209bee29",
    "configs/boundary/dev_i.yaml": "319456d5bfdd9025d00fcf18",
    "configs/boundary/dev_p.yaml": "da31068453c5e5dddb11c67a",
    "configs/boundary/dh.yaml": "5cf26d1d1fad4336296fa458",
    "configs/boundary/f2.yaml": "81b8c1713e7eef4b8cc8cb48",
    "configs/boundary/f2_minus_a.yaml": "1f99821a256e4c942cc47f1a",
    "configs/boundary/f2_minus_d.yaml": "f11943bc9f36a76efd64989b",
    "configs/boundary/f2_minus_p.yaml": "56063e0846c03dc315ad039a",
    "configs/boundary/f2_s12.yaml": "b6e6ffc96a3b11de121046d0",
    "configs/boundary/f2_s6.yaml": "f322e8c806dd97e5d9f0c004",
    "configs/boundary/i.yaml": "47c20e4d42182864e7657d5e",
    "configs/boundary/p.yaml": "ff115373f7afa3460085fd8f",
    "configs/diagnostic_shallow.yaml": "dbcad7d6974956fc639b004a",
    "configs/diagnostic_shallow_gated.yaml": "515aa65286176cf193b5cce2",
    "configs/markov/dev.yaml": "511216d393ecf103c5cd7e70",
    "configs/markov/dev_collect.yaml": "907c892476598fed99ad2dd2",
    "configs/markov/m1_collect.yaml": "2159f6097f20ca1ddb87a6c6",
    "configs/markov/m2.yaml": "e0634cbe2ae91f2d5ff63a44",
    "configs/markov/m2_bounds.yaml": "266d331779813bbb16e229ff",
    "configs/phase1_pilot.yaml": "4b570ef1dbb337d32fa4d1c0",
    "configs/phase2_pilot.yaml": "bb86e370aef9baa0e4d5c876",
    "configs/phase3_pilot.yaml": "ab6fef691e45a7d85cf201cc",
    "configs/policygraph_rq/pg_rq_dev.yaml": "410b154e6802af95187a1806",
    "configs/policygraph_rq/pg_rq_main.yaml": "526f49bc1cb382db5e2ec7fa",
    "configs/replication/dev_f0.yaml": "ae8b9c7245a48680dea23e6e",
    "configs/replication/dev_f1.yaml": "da6b196b08d7e3d6db3a8832",
    "configs/replication/dev_f2.yaml": "2d8fae75b497feba13942b9c",
    "configs/replication/dev_stage_d.yaml": "9a0ab428b5800e2e1bd083bc",
    "configs/replication/f0.yaml": "566217258e801593fc213289",
    "configs/replication/f0_bounds.yaml": "43305dc621508642a0ff35a0",
    "configs/replication/f1.yaml": "8c15cd556af418bafbb85a26",
    "configs/replication/f1_bounds.yaml": "34431aaa9361a151999b29f6",
    "configs/replication/f2.yaml": "e813a03bc4e3ea55dbfcfac2",
    "configs/replication/f2_bounds.yaml": "2e8df9e1b1b94795c54f674b",
    "configs/smoke.yaml": "93cf9e155740f0977a799398",
    "configs/smoke_p2.yaml": "2a95556e6cc71180884c4e6e",
    "configs/smoke_p3.yaml": "2cad13fb6cbaef6b087dd595",
    "configs/study.yaml": "f84771ff4120b0ea704e8c3c",
    "configs/transfer_frozen.yaml": "e453d3d64029957140838c4c",
    "configs/transfer_inference.yaml": "ddd755fe5ca349f954da1ed9",
    "configs/walker_stage_a.yaml": "dcdb282a31222fe6dcee1fc6",
    "configs/walker_stage_a_bounds.yaml": "e495d39ca01461b78bcfaa39",
    "configs/walker_stage_a_p1.yaml": "d2c5a29ca8adfc9952154217",
    "configs/walker_stage_a_p3.yaml": "db3206227f45ab32f084b0c4",
    "configs/walker_stage_b.yaml": "decd3eb3aea5ec64645e1cc4",
    "configs/walker_stage_b_bounds.yaml": "c591a3a25b5873e3b280b882",
    "configs/walker_stage_c.yaml": "df939c2eadea42c4c4ec05f2",
    "configs/walker_stage_d.yaml": "e4a1f74c0edeca8aa93ee51f",
    "configs/walker_stage_d_bounds.yaml": "fec3c69979fc9a1a410410d1",
}
FROZEN_STREAMS = {
    "configs/replication/f0.yaml": "3ffcf482ae7dc6a1a79fdb57",
    "configs/replication/f1.yaml": "6f9a5c131467f5055eb69527",
    "configs/replication/f2.yaml": "a4b1caeb4e2a139cddd7ac9c",
    "configs/markov/m1_collect.yaml": "50237f34b8fb7858ae8a74d1",
    "configs/markov/m2.yaml": "cab0e8bfdd22a1d2477da401",
    "configs/boundary/a.yaml": "4314099f601a67bbd8580645",
    "configs/boundary/d.yaml": "68fa8b03d7be7b640b83c160",
    "configs/boundary/dh.yaml": "ff6d2dac6b37d8b2d6338e94",
    "configs/boundary/i.yaml": "bdd331266a245d93715bd058",
    "configs/boundary/p.yaml": "25c1f7ada12b88b80835300e",
    "configs/boundary/f2_minus_p.yaml": "f0f38a3c1b4c2c72c12266f4",
    "configs/policygraph_rq/pg_rq_main.yaml": "4c07063362d8c154e0ef9692",
    "configs/walker_stage_d.yaml": "71e2f4312d6336d19f163cab",
}
X2_GENERATED = "36a9cd95dedc683745535f37"
FROZEN_AGENTS = {
    "goal_only:0.0:maximal": "df9a3a04848730e07f824c69",
    "goal_only:0.0:local_uniform": "1dff3eb2c9c0a628a3b97c69",
    "goal_only:0.0:local_focused": "749f03bd0b70c7c6903d9175",
    "goal_only:0.0:exact": "9682ed9a408052f81dabbdef",
    "goal_only:0.0:focused_sample": "8187a09350fe857c09415588",
    "goal_only:0.0:random_omit": "48a0da7c1c82d605c933d450",
    "goal_only:0.0:learned": "7a015e12e3be9ebc8e472e11",
    "goal_only:0.0:learned_sample": "fb632e75b6ee36a37ea96759",
    "goal_only:0.0:hyper_rank": "18e0146f04fc16e53ee5afba",
    "goal_only:0.0:pair_sample": "d1fc2ae968eb8bff962164cd",
    "goal_only:0.0:pair_rank": "7889da6963f694d7e66c51d5",
    "goal_only:0.0:hyper_sample_ppr": "f579835e173880d7b943227e",
    "goal_only:0.0:pg": "25013f2c3f87b74772850c5a",
    "goal_only:0.0:pg_uniform": "25013f2c3f87b74772850c5a",
    "goal_only:0.0:pg_greedy": "25013f2c3f87b74772850c5a",
    "all:0.0:maximal": "4811e05325d502115ff7ed1a",
    "all:0.0:sample": "a46d0905a9597326a5aaa967",
    "all:0.0:optimistic": "bd125a4c9da56bd77293ab45",
    "all:0.0:focused_sample": "a031146a07d734faf7f53fc6",
    "goal_only:0.1:focused_sample": "5f7fee070ffc397dc7319f99",
    "goal_only:0.1:learned_sample": "be554a08c2ad74ac2f3ad4e3",
}


def sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:24]


def recipe_rec(r):
    return [r.effect, list(r.item_inputs), list(r.pool), list(r.true_base), r.variant]


def task_rec(t):
    w = t.world
    return [t.task_key, t.namespace, t.seed, w.world_key, w.seed, [list(x) for x in w.levels],
            [recipe_rec(r) for r in w.recipes], t.profile, t.goal, t.depth, list(t.fact_types),
            list(t.fact_keys), [recipe_rec(r) for r in t.rules], list(t.rule_keys), sorted(t.initial_true),
            t.budget, t.failure_prob, t.observe_items]


def stream_digest(rel: str) -> str:
    from hypergraph_agent.config import eval_stream_config, load_config
    from hypergraph_agent.envs.generator import TaskStream
    from hypergraph_agent.evaluation.walkers import eval_order, variant_config
    cfg = load_config(ROOT / rel)
    parts = {}
    for seed in cfg["run"]["seeds"]:
        for v in cfg["walker"]["eval_variants"] or [{"name": "default", "tasks": {}}]:
            scfg = eval_stream_config(variant_config(cfg, seed, v))
            order = eval_order(scfg, cfg["eval"]["n_tasks"], cfg["eval"]["episodes_per_world"])
            s = TaskStream(scfg)
            parts[f"s{seed}:{v['name']}"] = sha({"tasks": [task_rec(s.task(i)) for i in order],
                                                 "config_hash": scfg.config_hash(),
                                                 "manifest": s.manifest(16)["manifest_hash"]})
    return sha(parts)


def test_frozen_configs_load_to_the_same_dictionaries():
    from hypergraph_agent.config import load_config
    for rel, digest in LOADED_CONFIGS.items():
        cfg = load_config(ROOT / rel)
        assert sha({k: v for k, v in cfg.items() if k != "_source"}) == digest, rel
        assert "schema" not in cfg["world"] and "hybrid" not in cfg["walker"]


@pytest.mark.parametrize("rel", FROZEN_STREAM_CONFIGS)
def test_frozen_streams_generate_the_same_worlds_and_tasks(rel):
    assert stream_digest(rel) == FROZEN_STREAMS[rel]


def test_x2_generated_streams_are_unchanged():
    from hypergraph_agent.inference_baselines.collection import load, stream_for
    cfg = load(ROOT / "configs" / "inference_baselines" / "x2_main.yaml")
    parts = {}
    for s in cfg["sets"]:
        if s["kind"] == "generated":
            for seed in s["seeds"]:
                stream, order = stream_for(s, seed)
                parts[f"{s['name']}:s{seed}"] = sha([task_rec(stream.task(i)) for i in order])
    assert sha(parts) == X2_GENERATED


def test_every_existing_agent_acts_as_before():
    from hypergraph_agent.agents.edit_policy import EditPolicy
    from hypergraph_agent.envs.generator import (
        PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig,
    )
    from hypergraph_agent.evaluation.walkers import eval_order, make_walker, play
    from hypergraph_agent.training.budget import BudgetMeter, SessionLedger
    wc = {"max_evals": 400, "temperature": 0.5, "restart_after": 80, "exact_cap": 20_000, "plan_cap": 4096,
          "consistent_moves": 50, "omit_prob": 0.663,
          "markov": {"restart": 1.0, "ppr_seed": "recipe", "prior_strength": 2.0, "rank_neighbours": True,
                     "proposal_floor": 0.1}}
    for observe, eps, strategies in AGENT_CASES:
        tasks = TaskConfig(profile=PROFILE_UNKNOWN, depth_min=1, depth_max=3, n_distractor_base=0,
                           observe_items=observe, failure_prob=eps)
        namespace = f"golden_{observe}_{int(eps * 100)}"
        scfg = TaskStreamConfig(namespace, 4242, "pool", 1, 0, WorldConfig(), tasks)
        order = eval_order(scfg, 5, 5)
        for strategy in strategies:
            policy = None
            if strategy in ("learned", "learned_sample"):
                torch.manual_seed(0)
                policy = EditPolicy(32)
            agent = make_walker(strategy, wc, (7, strategy, "golden"), eps, policy)
            m = BudgetMeter(SessionLedger(None), "golden", "diagnostics", 10_000, kind="adaptive")
            rows = play(agent, strategy, TaskStream(scfg), order, m, 7, None, "exploration", {})
            got = sha([{k: v for k, v in r.items() if k != "choose_seconds"} for r in rows])
            assert got == FROZEN_AGENTS[f"{observe}:{eps}:{strategy}"], (observe, eps, strategy)


def test_default_configs_build_the_plain_world_config():
    from dataclasses import asdict
    from hypergraph_agent.config import load_config, world_config
    from hypergraph_agent.envs.generator import WorldConfig
    cfg = load_config(ROOT / "configs" / "replication" / "f1.yaml")
    wc = world_config(cfg)
    assert type(wc) is WorldConfig and "schema" not in asdict(wc)
    cfg["world"]["schema"] = None  # an explicit null is the default too
    assert type(world_config(cfg)) is WorldConfig


# ======================================================================= the schema option

def schema_cfg(scope="world", hidden=False, **world):
    from hypergraph_agent.envs.generator import SchemaWorldConfig
    return SchemaWorldConfig(**world, schema={"n_classes": 3, "n_families": 4, "scope": scope,
                                              "hidden_classes": hidden})


def schema_stream(scope="world", hidden=False, namespace="hp_test_dev", seed=5, n_worlds=3, **world):
    from hypergraph_agent.envs.generator import PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig
    tasks = TaskConfig(profile=PROFILE_UNKNOWN, depth_min=1, depth_max=3, n_distractor_base=0,
                       observe_items="goal_only")
    wcfg = schema_cfg(scope, hidden, **world)
    return TaskStream(TaskStreamConfig(namespace, seed, "pool", n_worlds, 0, wcfg, tasks))


@pytest.mark.parametrize("hidden", [False, True])
def test_schema_requirements_take_one_fact_per_class_of_the_family(hidden):
    from hypergraph_agent.envs.generator import family_of, public_class
    s = schema_stream(hidden=hidden)
    for widx in range(3):
        w = s.world(widx)
        cls = w.schema.classes
        assert sorted(np.bincount(cls)) == [6, 6, 6]  # balanced partition
        if not hidden:
            assert all(cls[b] == public_class(b, 3) for b in range(18))
        for r in w.recipes:
            assert {cls[b] for b in r.pool} == {0, 1, 2}  # every pool holds every class
            want = w.schema.families[family_of(r.effect, 4)]
            assert sorted(cls[b] for b in r.true_base) == list(want)
            assert set(r.true_base) <= set(r.pool)


def test_schema_scopes_world_stream_and_none():
    world_scope = schema_stream("world", seed=11)
    stream_scope = schema_stream("stream", seed=11)
    fams = [world_scope.world(i).schema.families for i in range(3)]
    assert len(set(fams)) > 1  # drawn per world (seed chosen so the three differ)
    shared = [stream_scope.world(i).schema for i in range(3)]
    assert shared[0] == shared[1] == shared[2] == stream_scope.stream_schema()
    hidden = schema_stream("stream", hidden=True, seed=11)
    assert hidden.world(0).schema.classes == hidden.world(2).schema.classes
    none = schema_stream("none", seed=11)
    sizes = []
    for i in range(3):
        w = none.world(i)
        assert w.schema.families == ()
        sizes += [len(r.true_base) for r in w.recipes]
        assert all({w.schema.classes[b] for b in r.pool} == {0, 1, 2} for r in w.recipes)
    assert set(sizes) <= {1, 2, 3} and len(set(sizes)) > 1


def test_schema_worlds_are_deterministic_and_differ_from_default_worlds():
    a, b = schema_stream(seed=3), schema_stream(seed=3)
    assert [task_rec(a.task(i)) for i in range(8)] == [task_rec(b.task(i)) for i in range(8)]
    assert a.world(0).schema == b.world(0).schema
    assert a.cfg.config_hash() != schema_stream(scope="stream", seed=3).cfg.config_hash()


def test_schema_config_validation():
    from hypergraph_agent.envs.generator import SchemaConfig, SchemaWorldConfig
    with pytest.raises(ValueError):
        SchemaConfig(scope="galaxy")
    with pytest.raises(ValueError):
        SchemaConfig(n_classes=4)  # does not divide the 18 base types
    with pytest.raises(ValueError):
        SchemaWorldConfig(schema={"n_classes": 2})  # max_true 3 > 2 classes
    assert isinstance(SchemaWorldConfig(schema={"n_classes": 3}).schema, SchemaConfig)


def test_the_public_view_never_carries_the_schema():
    from hypergraph_agent.envs.generator import WorldSchema
    from hypergraph_agent.envs.public_schema import public_view
    s = schema_stream()
    t = s.task(0)
    other = WorldSchema(tuple(reversed(t.world.schema.classes)), ((0,),) * 4, True, "world")
    t2 = replace(t, world=replace(t.world, schema=other))
    assert public_view(t) == public_view(t2)


def test_schema_configs_reach_the_generator_through_the_config_loader(tmp_path):
    from hypergraph_agent.config import eval_stream_config, load_config, world_config
    from hypergraph_agent.envs.generator import SchemaWorldConfig, TaskStream
    cfg = load_config(HP2_DIR / "hp2_v1_world.yaml")
    wc = world_config(cfg)
    assert isinstance(wc, SchemaWorldConfig) and wc.schema.scope == "world"
    stream = TaskStream(eval_stream_config(cfg))
    assert stream.task(0).world.schema is not None
    bad = tmp_path / "bad.yaml"
    bad.write_text("world: {schema: {n_classes: 3, scope: nowhere}}\n")
    with pytest.raises(ValueError):
        world_config(load_config(bad))


# ======================================================================== opt-in prior hooks

def test_prior_marginal_and_size_log_prior():
    from hypergraph_agent.topology.inference import enumerate_hypotheses, prior_marginal, size_log_prior
    assert prior_marginal(6, 3) == pytest.approx(16 / 41)
    assert prior_marginal(6, 3, (0.2, 0.5, 0.3)) == pytest.approx(2.1 / 6)
    hyps = enumerate_hypotheses(6, 3)
    lp = size_log_prior((6 / 41, 15 / 41, 20 / 41), hyps)
    assert np.allclose(lp, -math.log(41))
    lp = size_log_prior((0.2, 0.5, 0.3), hyps)
    assert np.exp(lp).sum() == pytest.approx(1.0)
    assert np.exp(lp[0]) == pytest.approx(0.2 / 6) and np.exp(lp[-1]) == pytest.approx(0.3 / 20)
    assert size_log_prior((0.0, 1.0, 0.0), hyps)[0] == -math.inf


def test_dependency_model_prior_hook():
    from hypergraph_agent.priors.size import SizePrior
    from hypergraph_agent.topology.inference import DependencyModel, Evidence
    pool = (0, 1, 2, 3, 4, 5)
    a = DependencyModel("s", 0.0)
    b = DependencyModel("s", 0.0, log_prior=SizePrior((0.2, 0.5, 0.3)))
    assert a.config()["prior"] == "uniform" and b.config()["prior"] == "size"
    for dm in (a, b):
        dm.register_rule("r", (), pool)
        dm.apply(Evidence("e", "s", "r", frozenset({0, 1, 2}), True))  # 3, 4 and 5 not required
    pa, pb = a.rules["r"].posterior(), b.rules["r"].posterior()
    hyps = a.rules["r"].hypotheses
    ok = np.array([all(p in (0, 1, 2) for p in h) for h in hyps])
    assert np.allclose(pa[ok], 1 / 7) and np.allclose(pa[~ok], 0)
    w = np.array([{1: 0.2 / 6, 2: 0.5 / 15, 3: 0.3 / 20}[len(h)] for h in hyps]) * ok
    assert np.allclose(pb, w / w.sum())


class FixedPrior:
    """A test prior: every recipe's prior puts 0.9 on one fixed hypothesis."""

    def __init__(self, pick):
        self.pick = pick

    def log_prior(self, w, sig, h):
        n = len(w.hypotheses[sig])
        return math.log(0.9 + 0.1 / n) if h == self.pick(w, sig) else math.log(0.1 / n)

    def draw(self, w, sig, rng):
        return self.pick(w, sig)


def test_mix_consistent_targets_the_prior_on_consistent_nodes():
    from hypergraph_agent.agents.walker import EpisodeLog, RecipeInfo, WorldState, mix_consistent
    from hypergraph_agent.topology.inference import enumerate_hypotheses
    w = WorldState("Wt", 0.0, 3)
    info = RecipeInfo("r", T["key"], (), (0, 1, 2, 3), None)
    w.infos["r"] = info
    w.hypotheses["r"] = [frozenset(info.pool[p] for p in h) for h in enumerate_hypotheses(4, 3)]
    log = EpisodeLog(1, T["key"], frozenset({0, 1}))  # crafted with 0 and 1 present: success
    log.steps, log.goal_obs, log.effect_obs, log.base_obs = [(2, "r")], [True], [None], [None]
    w.evidence.logs = [log]
    consistent = [h for h in w.hypotheses["r"] if h <= {0, 1}]
    target = {frozenset({0}): 4.0, frozenset({1}): 1.0, frozenset({0, 1}): 1.0}
    rng = np.random.default_rng(0)
    counts = {h: 0 for h in consistent}
    node = {"r": frozenset({0, 1})}
    for _ in range(3000):
        node, _ = mix_consistent(node, w, None, rng, 3, 3,
                                 log_prior=lambda s, h: math.log(target.get(h, 1e-9)))
        counts[node["r"]] += 1
    p = np.array([counts[h] for h in consistent]) / 3000
    q = np.array([target[h] for h in consistent]) / 6.0
    assert 0.5 * np.abs(p - q).sum() < 0.05


def test_walker_draws_recipes_without_evidence_from_the_prior():
    from hypergraph_agent.agents.walker import Walker
    from hypergraph_agent.envs.recipequest import RecipeQuestEnv
    from hypergraph_agent.skills.executor import Executor
    s = schema_stream()
    t = s.task(0)
    pick = lambda w, sig: w.hypotheses[sig][-1]  # noqa: E731
    wk = Walker("focused_sample", np.random.default_rng(0), prior=FixedPrior(pick))
    ex = Executor(RecipeQuestEnv(), meter(), "exploration", torch.Generator().manual_seed(0))
    ex.reset(t, 0)
    w = wk.world(ex.spec.world_key)
    w.register(ex.spec, wk.rng)
    node, _ = wk.choose_node(w, None)
    assert all(node[sig] == w.hypotheses[sig][-1] for sig in w.hidden())


# ===================================================================== HP1: posterior and sizes

def small_world_logs(seed=0, episodes=5):
    """Logs of ``focused_sample`` on a hand-built world with pools of four and
    goal-only observation (for brute-force checks)."""
    from hypergraph_agent.agents.walker import Walker
    from hypergraph_agent.envs.generator import PROFILE_UNKNOWN
    from hypergraph_agent.envs.recipequest import RecipeQuestEnv
    from hypergraph_agent.skills.executor import Executor
    w = world([recipe("ingot", (), ("ore", "fuel", "sand", "wood"), ("ore", "fuel")),
               recipe("key", ("ingot",), ("wax", "salt", "clay", "fiber"), ("wax",)),
               recipe("key", ("ingot",), ("flint", "oil", "stone", "resin"), ("flint", "oil", "stone"), 1)],
              {"ingot": 1, "key": 2})
    facts = ("ore", "fuel", "sand", "wood", "wax", "salt", "clay", "fiber", "flint", "oil", "stone", "resin",
             "ingot", "key")
    t = replace(task(w, "key", facts, profile=PROFILE_UNKNOWN, budget=30), observe_items="goal_only")
    wk = Walker("focused_sample", np.random.default_rng(seed))
    env, m = RecipeQuestEnv(), meter()
    for e in range(episodes):
        wk.run_episode(Executor(env, m, "exploration", torch.Generator().manual_seed(e)), t, e)
    ws = next(iter(wk.worlds.values()))
    truth = {r.signature(t.profile): frozenset(r.true_base) for r in w.recipes}
    return ws.infos, ws.evidence.logs, truth


def brute_force(infos, logs, weights):
    """Candidate marginals and size-profile counts over every complete node."""
    from hypergraph_agent.evaluation.markov_study import consistent
    from hypergraph_agent.priors.posterior import class_hypotheses
    sigs = sorted({s for lg in logs for s in lg.attempted() if infos[s].known is None})
    classes = [class_hypotheses(infos[s]) for s in sigs]
    marg, total, counts = {}, 0.0, {}
    for combo in product(*classes):
        node = dict(zip(sigs, combo))
        if all(consistent(node, lg, infos) for lg in logs):
            wt = math.prod(weights(infos[s].pool)[classes[i].index(h)]
                           for i, (s, h) in enumerate(node.items()))
            total += wt
            for s, h in node.items():
                for b in h:
                    marg[(s, b)] = marg.get((s, b), 0.0) + wt
            prof = (sum(len(h) == 1 for h in combo), sum(len(h) == 2 for h in combo))
            counts[prof] = counts.get(prof, 0) + 1
    return {(s, b): marg.get((s, b), 0.0) / total for s in sigs for b in infos[s].pool}, counts


def test_weighted_posterior_equals_brute_force_and_study_m():
    from hypergraph_agent.evaluation.markov_study import posterior_marginals
    from hypergraph_agent.priors.posterior import UniformPrior, joint_marginals
    from hypergraph_agent.priors.size import SizePrior
    infos, logs, _ = small_world_logs()
    sp = SizePrior((0.2, 0.5, 0.3))
    (mu, ms), diag = joint_marginals(infos, logs, [UniformPrior(), sp], 0)
    assert diag["exact_recipes"] >= 2 and diag["sampled_recipes"] == 0
    bu, _ = brute_force(infos, logs, UniformPrior().weights)
    bs, _ = brute_force(infos, logs, sp.weights)
    assert all(abs(mu[e] - bu[e]) < 1e-9 for e in bu)
    assert all(abs(ms[e] - bs[e]) < 1e-9 for e in bs)
    m1, _ = posterior_marginals(infos, logs, np.random.default_rng(0))
    assert all(abs(m1[e] - mu[e]) < 1e-12 for e in m1)


def test_size_polynomials_count_consistent_nodes_by_size():
    from hypergraph_agent.priors.posterior import UniformPrior, joint_marginals
    infos, logs, _ = small_world_logs(seed=2)
    polys = []
    joint_marginals(infos, logs, [UniformPrior()], 0, polys=polys)
    _, counts = brute_force(infos, logs, UniformPrior().weights)
    total = polys[0]
    for p in polys[1:]:  # independent groups multiply
        out = np.zeros((total.shape[0] + p.shape[0] - 1,) * 2)
        for i, j in zip(*np.nonzero(p)):
            out[i:i + total.shape[0], j:j + total.shape[1]] += p[i, j] * total
        total = out
    got = {(int(i), int(j)): int(round(total[i, j])) for i, j in zip(*np.nonzero(total))}
    assert got == counts


def test_gibbs_fallback_approaches_the_exact_posterior():
    from hypergraph_agent.priors.posterior import UniformPrior, joint_marginals
    from hypergraph_agent.priors.size import SizePrior
    infos, logs, _ = small_world_logs(seed=1)
    priors = [UniformPrior(), SizePrior((0.2, 0.5, 0.3))]
    exact, _ = joint_marginals(infos, logs, priors, 0)
    sampled, diag = joint_marginals(infos, logs, priors, 0, exact_work=1, sweeps=200)
    assert diag["sampled_recipes"] > 0 and diag["exact_recipes"] == 0
    for e, s in zip(exact, sampled):
        assert max(abs(e[k] - s[k]) for k in e) < 0.08


def test_factorized_posterior_matches_study_x2_and_gives_polynomials():
    from hypergraph_agent.agents.walker import Walker
    from hypergraph_agent.envs.generator import (
        PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig,
    )
    from hypergraph_agent.envs.recipequest import RecipeQuestEnv
    from hypergraph_agent.inference_baselines.learners import factorized_marginals as x2_factorized
    from hypergraph_agent.inference_baselines.traces import craft_examples
    from hypergraph_agent.priors.posterior import UniformPrior, factorized_marginals
    from hypergraph_agent.priors.size import SizePrior
    from hypergraph_agent.skills.executor import Executor
    st = TaskStream(TaskStreamConfig("hp_obs_dev", 9, "pool", 1, 0, WorldConfig(),
                                     TaskConfig(profile=PROFILE_UNKNOWN, depth_min=1, depth_max=2,
                                                n_distractor_base=0, observe_items="all")))
    wk = Walker("focused_sample", np.random.default_rng(0))
    env, m = RecipeQuestEnv(), meter()
    for i in range(6):
        wk.run_episode(Executor(env, m, "exploration", torch.Generator().manual_seed(i)), st.task(i), i)
    ws = next(iter(wk.worlds.values()))
    sigs = sorted({x for lg in ws.evidence.logs for x in lg.attempted()})
    polys = []
    priors = [UniformPrior(), SizePrior((0.2, 0.5, 0.3))]
    (mu, ms), _ = factorized_marginals(ws.infos, ws.evidence.logs, sigs, priors, polys=polys)
    ref, _ = x2_factorized(ws.infos, craft_examples(ws.evidence.logs, ws.infos), sigs)
    assert all(abs(ref[e] - mu[e]) < 1e-12 for e in ref)
    assert len(polys) == len(sigs) and all(p.shape == (2, 2) for p in polys)
    assert any(abs(mu[e] - ms[e]) > 1e-6 for e in mu)


def test_size_priors_and_empirical_bayes():
    from hypergraph_agent.priors.size import learned_sizes, marginal_sizes, uniform_sizes
    u = uniform_sizes()
    assert u == pytest.approx((6 / 41, 15 / 41, 20 / 41))
    assert learned_sizes([], 2.0) == pytest.approx(u)
    want = ((2 * u[0] + 1) / 5, 2 * u[1] / 5, (2 * u[2] + 2) / 5)  # one pin of size 1, two of size 3
    assert learned_sizes([3, 3, 1], 2.0) == pytest.approx(want)
    assert marginal_sizes([], 2.0) == pytest.approx(u)
    unconstrained = np.array([[20.0, 15.0], [6.0, 0.0]])
    assert marginal_sizes([unconstrained] * 3, 2.0) == pytest.approx(u)
    pins = [np.array([[1.0, 0.0], [0.0, 0.0]]), np.array([[0.0, 0.0], [1.0, 0.0]])]  # sizes 3 and 1
    assert marginal_sizes(pins, 2.0) == pytest.approx(learned_sizes([3, 1], 2.0))
    # an evidence polynomial of two recipes against numerical integration over the simplex (a
    # hyperprior strength of 10 keeps the density smooth for the quadrature)
    poly = np.array([[0.0, 3.0, 1.0], [2.0, 4.0, 0.0], [1.0, 0.0, 0.0]])
    grid = 800
    i, j = np.meshgrid(np.arange(grid), np.arange(grid), indexing="ij")
    p1, p2 = (i.ravel() + 0.5) / grid, (j.ravel() + 0.5) / grid
    p3 = 1.0 - p1 - p2
    keep = p3 > 0
    p = np.stack([p1[keep], p2[keep], p3[keep]], axis=1)
    q = p / np.array([6.0, 15.0, 20.0])
    lik = sum(poly[c1, c2] * q[:, 0] ** c1 * q[:, 1] ** c2 * q[:, 2] ** (2 - c1 - c2)
              for c1 in range(3) for c2 in range(3 - c1))
    dens = np.prod(p ** (10.0 * np.array(u) - 1.0), axis=1) * lik
    assert marginal_sizes([poly], 10.0) == pytest.approx(tuple((dens[:, None] * p).sum(0) / dens.sum()),
                                                         abs=0.002)


def test_pinned_recipes_and_size_prior_rankings():
    from hypergraph_agent.priors.posterior import UniformPrior, joint_marginals, pinned
    from hypergraph_agent.priors.size import SizePrior
    infos, logs, truth = small_world_logs(seed=3, episodes=8)
    (mu, ms), _ = joint_marginals(infos, logs, [UniformPrior(), SizePrior((0.6, 0.3, 0.1))], 0)
    sigs = sorted({s for lg in logs for s in lg.attempted()})
    pins = pinned(mu, infos, sigs)
    assert all(pins[s] == truth[s] for s in pins)
    assert pinned(ms, infos, sigs) == pins  # being pinned does not depend on the prior


# ===================================================================== HP1: evaluation pipeline

def hp1_records(tmp_path):
    """Evidence records of a tiny M1-style collection (in memory, scratch namespace). Shallow worlds
    (two levels, goals of depth 1-2) and four episodes keep the exact enumeration small."""
    from hypergraph_agent.config import load_config
    from hypergraph_agent.evaluation.markov_study import collect, load_evidence
    from hypergraph_agent.training.budget import SessionLedger
    cfg = load_config(ROOT / "configs" / "markov" / "dev_collect.yaml")
    cfg["eval"].update(namespace="hp_unit_dev", n_tasks=8, n_worlds=2, episodes_per_world=4)
    cfg["run"].update(name="hpunit-m1", seeds=[0])
    cfg["world"]["n_levels"] = 2
    cfg["train"]["tasks"]["depth_max"] = 2
    collect(cfg, SessionLedger(None, allocations={"dev": 50_000}), str(tmp_path))
    return load_evidence(tmp_path, "hpunit-m1-")


def test_hp1_pipeline_on_a_tiny_collection(tmp_path):
    from hypergraph_agent.evaluation.hybrid_priors import PRIORS, stream_rows, summarize_hp1, validate_hp1
    recs, _ = hp1_records(tmp_path)
    cfg = validate_hp1({
        "name": "hp1-unit-dev",
        "sources": [{"name": "focused", "study": "m1", "prefix": "hpunit-m1-", "strategy": "focused_sample",
                     "namespace": "hp_unit_dev", "expected_worlds": 2},
                    {"name": "maximal", "study": "m1", "prefix": "hpunit-m1-", "strategy": "maximal",
                     "namespace": "hp_unit_dev", "expected_worlds": 2}],
        "checkpoints": [2, 4], "priors": list(PRIORS),
        "posterior": {"exact_work": 100_000},  # a bound on any one group (beyond it: Gibbs)
        "primary": {"source": "focused", "k": 2, "a": "learned_stream", "b": "uniform", "metric": "brier",
                    "margin": 0.005},
        "secondary": [{"name": "S", "source": "focused", "k": 4, "a": "oracle", "b": "uniform",
                       "metric": "brier", "holds": "below"}]})
    rows = []
    for src in ("focused", "maximal"):
        strategy = "maximal" if src == "maximal" else "focused_sample"
        ws = sorted([r for r in recs if r["strategy"] == strategy],
                    key=lambda r: min(x["order"] for x in r["rows"]))
        rows += stream_rows((src, ws, cfg))
    assert len(rows) == 8 and all(set(r["metrics"]) == set(PRIORS) for r in rows)
    focused = [r for r in rows if r["source"] == "focused"]
    first = [r for r in focused if r["world_rank"] == min(x["world_rank"] for x in focused)]
    for r in first:  # no earlier world: the stream priors equal the world priors
        assert r["learned"]["learned_stream"] == r["learned"]["learned_world"]
        assert r["learned"]["marginal_stream"] == r["learned"]["marginal_world"]
    for r in rows:
        if r["source"] == "maximal":  # brute-force evidence: nothing pinned, every prior at chance
            assert r["pinned"] == 0 and all(m["auroc"] == 0.5 for m in r["metrics"].values())
    exact = [r for r in rows if not r["posterior"].get("sampled_recipes")]
    assert exact and all(r["pinned_wrong"] == 0 for r in exact)  # pins of exact posteriors are the truth
    status = {s: {"expected_worlds": 2, "problems": []} for s in ("focused", "maximal")}
    s = summarize_hp1(rows, status, cfg)
    assert s["validity"]["V1"]["holds"] is True
    assert s["primary"]["complete"] and s["primary"]["decision"] in (
        "learned_stream better", "uniform better", "equivalent", "inconclusive")
    assert s["secondary"][0]["n_worlds"] == 2 and s["secondary"][0]["holds"] in (True, False)
    assert s["primary_stream_bootstrap"]["clusters"] == 1  # one seed: one stream


def test_hp1_decision_rule():
    from hypergraph_agent.evaluation.hybrid_priors import decision
    assert decision([0.02, 0.03, 0.025, 0.021] * 5, 0.01)["decision"] == "learned better"
    assert decision([-0.02, -0.03, -0.025, -0.021] * 5, 0.01)["decision"] == "uniform better"
    assert decision([0.0005, -0.0004, 0.0002, -0.0001] * 5, 0.01)["decision"] == "equivalent"
    assert decision([0.05, -0.05, 0.04, -0.06], 0.01)["decision"] == "inconclusive"
    # lower is better (Brier score): a negative difference favours a
    assert decision([-0.02, -0.03, -0.025] * 5, 0.005, lower_is_better=True)["decision"] == "learned better"


def hp1_fake_rows(diffs, wrong_pin=False):
    """Per-world rows of one source at 4 episodes: oracle - uniform Brier differences ``diffs``, two
    streams, one exact pin per world (wrong in the first world with ``wrong_pin``)."""
    rows, learned_arms = [], ("learned_world", "learned_stream", "marginal_world", "marginal_stream")
    for i, d in enumerate(diffs):
        m = {p: {x: 0.2 for x in ("auroc", "f1", "precision", "recall", "gather", "brier")}
             for p in ("uniform", "oracle")}
        m["oracle"]["brier"] = 0.2 + d
        rows.append({"source": "src", "k": 4, "world": f"s{i % 2}:W{i}", "run_id": f"run{i % 2}",
                     "metrics": m, "recipes": 3, "pinned": 1, "pinned_wrong": int(wrong_pin and i == 0),
                     "pinned_sampled": 0,
                     "pinned_sampled_wrong": 0, "posterior": {}, "seconds": 0.0,
                     "learned": {p: [0.2, 0.4, 0.4] for p in learned_arms},
                     "pinned_sizes": {"3": 1}, "true_sizes_scored": {"2": 3}})
    return rows


def test_hp1_verdict_vocabulary_follows_studies_x2_and_rl():
    from hypergraph_agent.evaluation.hybrid_priors import result_of, summarize_hp1, validate_hp1
    cfg = validate_hp1({
        "name": "hp1-vocab-dev", "checkpoints": [4], "priors": ["uniform", "oracle"],
        "sources": [{"name": "src", "study": "m1", "prefix": "x-", "strategy": "focused_sample",
                     "namespace": "v_dev", "expected_worlds": 6}],
        "primary": {"source": "src", "k": 4, "a": "oracle", "b": "uniform", "metric": "brier",
                    "margin": 0.005},
        "secondary": [{"name": "S", "source": "src", "k": 4, "a": "oracle", "b": "uniform", "metric": "brier",
                       "holds": "below"}]})
    diffs = [-0.02, -0.021, -0.019, -0.022, -0.018, -0.02]
    ok = {"src": {"expected_worlds": 6, "problems": []}}
    s = summarize_hp1(hp1_fake_rows(diffs), ok, cfg)
    assert s["primary"]["complete"] and s["primary_verdict"] == s["primary"]["decision"] == "oracle better"
    assert s["secondary"][0]["holds"] is True and result_of(s["secondary"][0]) is True
    # a test that does not apply is None and invalidates nothing
    assert s["validity"]["V1"]["holds"] is None and s["validity"]["V3"]["holds"] is None
    assert s["validity"]["V2"]["holds"] is True and s["invalid"] == []
    assert s["primary_stream_bootstrap"]["clusters"] == 2
    # a missing world: the primary and every test of the source are incomplete (None)
    s = summarize_hp1(hp1_fake_rows(diffs), {"src": {"expected_worlds": 7, "problems": []}}, cfg)
    assert s["primary_verdict"] is None and s["primary"]["decision"] is None and not s["primary"]["complete"]
    assert s["secondary"][0]["holds"] is None and s["secondary"][0]["problems"] == ["6 of 7 worlds"]
    # a failed validity test makes a complete result invalid
    s = summarize_hp1(hp1_fake_rows(diffs, wrong_pin=True), ok, cfg)
    assert s["validity"]["V2"]["holds"] is False and s["primary_verdict"] == "invalid"


def test_hp1_pins_come_only_from_exactly_computed_groups(tmp_path, monkeypatch):
    import hypergraph_agent.priors.posterior as post
    from hypergraph_agent.evaluation.hybrid_priors import validate_hp1, world_checkpoint
    from hypergraph_agent.priors.size import learned_sizes
    infos, logs, _ = small_world_logs(seed=3, episodes=8)
    attempted = {s for lg in logs for s in lg.attempted() if infos[s].known is None}
    assert set(post.joint_marginals(infos, logs, [post.UniformPrior()], 0)[1]["exact_sigs"]) == attempted
    assert post.joint_marginals(infos, logs, [post.UniformPrior()], 0, exact_work=1)[1]["exact_sigs"] == []
    recs, _ = hp1_records(tmp_path)
    rec = next(r for r in recs if r["strategy"] == "focused_sample")
    truth = {s: frozenset(v) for s, v in rec["truth"]["requirements"].items()}
    real, box = post.joint_marginals, {}

    def one_exact(*a, **kw):  # only one recipe counts as exactly computed
        out, diag = real(*a, **kw)
        box["exact"] = sorted(diag["exact_sigs"])[:1]
        return out, {**diag, "exact_sigs": box["exact"]}

    def all_pinned(marginals, infos_, sigs, tol=1e-9):  # every scored recipe pinned, sampled ones wrongly
        return {s: truth[s] if s in box["exact"] else frozenset(infos_[s].pool) for s in sigs if s in truth}

    monkeypatch.setattr(post, "joint_marginals", one_exact)
    monkeypatch.setattr(post, "pinned", all_pinned)
    cfg = validate_hp1({"name": "hp1-unit-dev", "checkpoints": [4], "posterior": {"exact_work": 100_000},
                        "sources": [{"name": "f", "study": "m1", "prefix": "hpunit-m1-",
                                     "strategy": "focused_sample", "namespace": "hp_unit_dev"}],
                        "primary": {"source": "f", "k": 4, "a": "learned_world", "b": "uniform",
                                    "metric": "brier", "margin": 0.005}})
    r = world_checkpoint(rec, 4, {"sizes": [], "polys": []}, cfg)
    (sig,) = box["exact"]
    assert r["recipes"] >= 2 and r["pinned"] == 1 and r["pinned_wrong"] == 0
    assert r["pinned_sampled"] == r["recipes"] - 1 == r["pinned_sampled_wrong"]  # reported, never used
    assert r["pinned_sizes"] == {len(truth[sig]): 1}
    assert r["learned"]["learned_world"] == pytest.approx(learned_sizes([len(truth[sig])], 2.0), abs=1e-6)


def test_run_hp1_keeps_finished_streams_when_one_fails(tmp_path, monkeypatch):
    import hypergraph_agent.evaluation.hybrid_priors as hp
    hp1_records(tmp_path)  # the tiny collection's runs, in tmp_path
    cfg = hp.validate_hp1({
        "name": "hp1-unit-dev", "checkpoints": [2], "posterior": {"exact_work": 100_000},
        "sources": [{"name": n, "study": "m1", "prefix": "hpunit-m1-", "strategy": st,
                     "namespace": "hp_unit_dev", "expected_worlds": 2}
                    for n, st in (("focused", "focused_sample"), ("maximal", "maximal"))],
        "primary": {"source": "maximal", "k": 2, "a": "oracle", "b": "uniform", "metric": "brier",
                    "margin": 0.005}})
    real = hp.stream_rows

    def flaky(job):
        if job[0] == "maximal":
            raise RuntimeError("boom")
        return real(job)

    monkeypatch.setattr(hp, "stream_rows", flaky)
    cache, quiet = tmp_path / "cache", (lambda *_: None)
    rows, status = hp.run_hp1(cfg, tmp_path, workers=1, cache_dir=cache, progress=quiet)
    assert rows and {r["source"] for r in rows} == {"focused"}  # the finished stream is kept
    assert status["maximal"]["failed_streams"] and "boom" in status["maximal"]["problems"][-1]
    s = hp.summarize_hp1(rows, status, cfg)
    assert not s["primary"]["complete"] and s["primary_verdict"] is None
    assert len(list(cache.rglob("*.json"))) == 1  # and cached
    monkeypatch.setattr(hp, "stream_rows", real)
    rows, status = hp.run_hp1(cfg, tmp_path, workers=1, cache_dir=cache, resume=True, progress=quiet)
    assert {r["source"] for r in rows} == {"focused", "maximal"} and not status["maximal"]["problems"]


def test_hp1_cache_files_are_written_whole_or_not_at_all(tmp_path, monkeypatch):
    import hypergraph_agent.evaluation.hybrid_priors as hp
    hp1_records(tmp_path)
    cfg = hp.validate_hp1({
        "name": "hp1-unit-dev", "checkpoints": [2],
        "sources": [{"name": "focused", "study": "m1", "prefix": "hpunit-m1-", "strategy": "focused_sample",
                     "namespace": "hp_unit_dev", "expected_worlds": 2}],
        "primary": {"source": "focused", "k": 2, "a": "oracle", "b": "uniform", "metric": "brier",
                    "margin": 0.005}})
    monkeypatch.setattr(hp, "stream_rows", lambda job: [{"source": job[0], "run_id": job[1][0]["run_id"],
                                                         "world_rank": 0, "k": 2}])
    real_write, cache, quiet = Path.write_text, tmp_path / "cache", (lambda *_: None)

    def killed(self, data, *args, **kwargs):  # half the bytes reach the disk, then the process dies
        real_write(self, data[: len(data) // 2], *args, **kwargs)
        raise RuntimeError("killed mid-write")

    monkeypatch.setattr(Path, "write_text", killed)
    with pytest.raises(RuntimeError):
        hp.run_hp1(cfg, tmp_path, workers=1, cache_dir=cache, progress=quiet)
    monkeypatch.setattr(Path, "write_text", real_write)
    assert not list(cache.rglob("*.json"))  # the torn bytes sit in a temporary file only
    rows, _ = hp.run_hp1(cfg, tmp_path, workers=1, cache_dir=cache, resume=True, progress=quiet)
    assert rows and len(list(cache.rglob("*.json"))) == 1 and not list(cache.rglob("*.tmp"))


def test_hp1_configs_are_valid_and_dev_reads_only_dev_namespaces():
    from hypergraph_agent.evaluation.hybrid_priors import load_hp1, validate_hp1
    dev, main = load_hp1(HP1_DIR / "hp1_dev.yaml"), load_hp1(HP1_DIR / "hp1_main.yaml")
    assert all(s["namespace"].endswith("_dev") for s in dev["sources"])
    assert not any(s["namespace"].endswith("_dev") for s in main["sources"])
    assert {s["namespace"] for s in main["sources"]} == {"markov_sel", "x2_gen"}
    for cfg in (main, dev):  # switched before the freeze; the design note's estimator is S1
        assert cfg["primary"] == {"source": "m1_focused", "k": 4, "a": "marginal_stream", "b": "uniform",
                                  "metric": "brier", "margin": 0.005}
        s1 = cfg["secondary"][0]
        assert s1["name"].startswith("S1 ")
        assert (s1["a"], s1["b"], s1["holds"]) == ("learned_stream", "uniform", "decision")
    assert not any(t["a"] == "marginal_stream" and t["b"] == "uniform" and t["k"] == 4
                   and t["source"] == "m1_focused" and t["metric"] == "brier" for t in main["secondary"])
    raw = yaml.safe_load((HP1_DIR / "hp1_dev.yaml").read_text())
    raw["sources"][0]["namespace"] = "markov_sel"  # a development config reading a measurement namespace
    with pytest.raises(ValueError):
        validate_hp1(raw)
    raw = yaml.safe_load((HP1_DIR / "hp1_dev.yaml").read_text())
    raw["priors"] = ["uniform", True]  # a YAML boolean is not a prior
    with pytest.raises(ValueError):
        validate_hp1(raw)
    with pytest.raises(KeyError):
        validate_hp1({**yaml.safe_load((HP1_DIR / "hp1_dev.yaml").read_text()), "extra": 1})


# ======================================================================= HP2: the rule learners

def test_rainbow_and_allowed_sets():
    from hypergraph_agent.priors.schema import Observation, class_size, rainbow
    pool, cls = (0, 1, 2, 3, 4, 5), (0, 1, 2, 0, 1, 2)
    r = rainbow(pool, cls, (0, 2))
    assert sum(r.values()) == pytest.approx(1.0) and len(r) == 4
    assert all(sorted(cls[pool.index(b)] for b in h) == [0, 2] for h in r)
    assert rainbow(pool, (0, 0, 0, 0, 0, 0), (1,)) == {}
    o = Observation(("W", "s"), 0, pool, frozenset({0}), frozenset({5}))
    allowed = o.allowed()
    assert all(0 in h and 5 not in h and len(h) <= 3 for h in allowed)
    assert len(allowed) == 1 + 4 + 6 and class_size(pool) == 41
    assert Observation(("W", "s"), 0, pool, frozenset({0, 1}), frozenset({2, 3, 4, 5})).pinned


def test_schema_v1_learns_the_class_set_exactly():
    from hypergraph_agent.envs.generator import public_class
    from hypergraph_agent.priors.schema import ModelAverage, Observation, SchemaV1, options, rainbow
    pool = (0, 1, 2, 3, 4, 5)  # classes 0 1 2 0 1 2
    pin = Observation(("W", "a"), 1, pool, frozenset({0, 4}), frozenset({1, 2, 3, 5}))  # classes 0 and 1
    partial = Observation(("W", "b"), 1, (6, 7, 8, 9, 10, 11), frozenset({7}), frozenset())  # class 1
    learner = SchemaV1(3, 4)
    learner.fit([pin, partial])
    post = np.exp(learner.log_post[1] - learner.log_post[1].max())
    post /= post.sum()
    assert post[options(3).index((0, 1))] == pytest.approx(1.0)
    pred = learner.predictive(1, pool)
    assert set(pred) == set(rainbow(pool, tuple(public_class(b, 3) for b in pool), (0, 1)))
    # brute force of the marginal likelihood of the family's data
    lik = []
    for c in options(3):
        p = 1.0
        for o in (pin, partial):
            r = rainbow(o.pool, tuple(public_class(b, 3) for b in o.pool), c)
            p *= sum(r.get(h, 0.0) for h in o.allowed())
        lik.append(p / 7)
    assert learner.log_marginal == pytest.approx(math.log(sum(lik)))
    avg = ModelAverage(SchemaV1(3, 4))
    avg.fit([pin, partial])
    assert 0.5 < avg.weight() < 1.0
    lp = avg.log_prior(1, pool, [frozenset({0, 4}), frozenset({0, 3})])
    assert lp[0] > lp[1]


def test_model_average_drops_a_contradicted_schema():
    from hypergraph_agent.priors.schema import ModelAverage, Observation, SchemaV1
    pool = (0, 1, 2, 3, 4, 5)
    same_class = Observation(("W", "a"), 0, pool, frozenset({0, 3}), frozenset({1, 2, 4, 5}))  # classes 0, 0
    avg = ModelAverage(SchemaV1(3, 4))
    avg.fit([same_class])
    assert avg.weight() == 0.0
    hyps = [frozenset({0}), frozenset({0, 3}), frozenset({1, 2, 4})]
    assert np.allclose(avg.log_prior(0, pool, hyps), -math.log(len(hyps)))


def test_schema_v2_enumeration_matches_brute_force():
    pytest.importorskip("clingo")
    from hypergraph_agent.priors.schema import Observation, SchemaV2, options
    pins = [Observation(("W", "a"), 0, (0, 1, 2, 3, 4, 5), frozenset({0, 1}), frozenset({2, 3, 4, 5})),
            Observation(("W", "b"), 0, (1, 6, 7, 8, 9, 10), frozenset({1, 6}), frozenset({7, 8, 9, 10})),
            Observation(("W", "c"), 2, (0, 6, 11, 12, 13, 14), frozenset({11}), frozenset())]
    learner = SchemaV2(3, 4)
    learner.fit(pins)
    got = {(tuple(sorted(c.items())), tuple(sorted(s.items()))) for c, s in learner.models}
    facts = [0, 1, 6, 11]
    want = set()
    for z in product(range(3), repeat=len(facts)):
        cls = dict(zip(facts, z))
        for c0 in options(3):
            for c2 in options(3):
                ok = len(c0) == 2 and {cls[0], cls[1]} == set(c0) and {cls[1], cls[6]} == set(c0) \
                    and cls[0] != cls[1] and cls[1] != cls[6] and cls[11] in c2
                if ok:
                    want.add((tuple(sorted(cls.items())), tuple(sorted({0: c0, 2: c2}.items()))))
    assert got == want
    pred = learner.predictive(0, (0, 1, 2, 3, 4, 5))
    assert sum(pred.values()) == pytest.approx(1.0)
    assert max(pred, key=pred.get) == frozenset({0, 1})


def test_model_average_keeps_its_previous_fit_on_an_overflow():
    pytest.importorskip("clingo")
    from hypergraph_agent.priors.schema import ModelAverage, Observation, SchemaV2
    first = [Observation(("W", "a"), 0, (0, 1, 2, 3, 4, 5), frozenset({0, 1}), frozenset({2, 3, 4, 5}))]
    more = first + [Observation(("W", "b"), 1, (6, 7, 8, 9, 10, 11), frozenset({6, 7, 8}), frozenset())]
    avg = ModelAverage(SchemaV2(3, 4, max_models=10))
    avg.fit(first)  # 6 models: two facts of distinct classes fix family 0's class set
    assert avg.learner.overflow == 0 and len(avg.learner.models) == 6
    before = (avg.version, avg.log_odds(), len(avg.observations))
    avg.fit(more)  # three facts of distinct classes in family 1 as well: 36 models, over the cap
    assert avg.learner.overflow == 1
    assert (avg.version, avg.log_odds(), len(avg.observations)) == before


def test_oracle_schema_puts_its_mass_on_the_true_rainbow_sets():
    from hypergraph_agent.envs.generator import WorldSchema
    from hypergraph_agent.evaluation.hybrid_priors import OracleSchema  # privileged: evaluator side
    schema = WorldSchema(tuple(b % 3 for b in range(18)), ((0, 2), (1,), (0, 1, 2), (2,)), False, "world")
    o = OracleSchema(schema, 4, (0.2, 0.5, 0.3), floor=0.01)
    from hypergraph_agent.priors.posterior import class_hypotheses
    from hypergraph_agent.agents.walker import RecipeInfo
    info = RecipeInfo("s", 18, (), (0, 1, 2, 3, 4, 5), None)
    hyps = class_hypotheses(info)
    p = np.exp(o.log_prior(0, info.pool, hyps))
    assert p.sum() == pytest.approx(1.0)
    top = {h for h, x in zip(hyps, p) if x > 0.1}
    assert top == {frozenset(x) for x in ((0, 2), (0, 5), (3, 2), (3, 5))}
    none = OracleSchema(replace(schema, families=()), 4, (0.2, 0.5, 0.3))
    q = np.exp(none.log_prior(0, info.pool, hyps))
    assert q[0] == pytest.approx(0.99 * 0.2 / 6 + 0.01 / 41)


# ============================================================================== HP2: agents

def hp2_wc(scope="world", classes="public"):
    return {"max_evals": 400, "temperature": 0.5, "restart_after": 80, "exact_cap": 20_000, "plan_cap": 4096,
            "consistent_moves": 50, "omit_prob": None,
            "hybrid": {"scope": scope, "classes": classes, "n_classes": 3, "n_families": 4}}


def test_hybrid_params_are_validated():
    from hypergraph_agent.agents.hybrid import hybrid_params
    with pytest.raises(ValueError):
        hybrid_params({})
    with pytest.raises(KeyError):
        hybrid_params({**hp2_wc()["hybrid"], "temperature": 2})
    with pytest.raises(ValueError):
        hybrid_params({**hp2_wc()["hybrid"], "scope": "galaxy"})
    assert hybrid_params(hp2_wc()["hybrid"])["model_prior"] == 0.5


def test_hp2_strategy_names_never_shadow_other_strategies():
    from hypergraph_agent.agents.hybrid import HP_STRATEGIES
    from hypergraph_agent.agents.markov import MARKOV_STRATEGIES
    from hypergraph_agent.agents.policygraph import PG_STRATEGIES
    from hypergraph_agent.agents.walker import STRATEGIES
    assert not set(HP_STRATEGIES) & (set(STRATEGIES) | set(PG_STRATEGIES) | set(MARKOV_STRATEGIES))


@pytest.mark.parametrize("scope,classes,hidden", [("world", "public", False), ("stream", "hidden", True)])
def test_every_hp2_arm_plays_schema_worlds(scope, classes, hidden):
    from hypergraph_agent.envs.generator import TaskStream
    from hypergraph_agent.evaluation.walkers import eval_order, make_walker, play
    stream = schema_stream(scope, hidden, n_worlds=2, n_levels=4)
    order = eval_order(stream.cfg, 8, 4)
    for strategy in ("hybrid", "rules_only", "oracle_schema"):
        agent = make_walker(strategy, hp2_wc(scope, classes), (0, strategy, "unit"), 0.0)
        assert agent.prior is not None and agent.strategy == "focused_sample"
        rows = play(agent, strategy, TaskStream(stream.cfg), order, meter(), 0, None, "exploration", {})
        assert len(rows) == 8
        assert all({"choose_seconds", "hp_observed", "hp_pinned", "hp_schema_weight"} <= set(r) for r in rows)
        if classes == "hidden" and strategy != "oracle_schema":
            assert all("hp_models" in r for r in rows)


def test_oracle_schema_refuses_worlds_without_a_schema():
    from hypergraph_agent.envs.generator import (
        PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig,
    )
    from hypergraph_agent.evaluation.walkers import make_walker, play
    tasks = TaskConfig(profile=PROFILE_UNKNOWN, depth_min=1, depth_max=2, n_distractor_base=0,
                       observe_items="goal_only")
    s = TaskStream(TaskStreamConfig("hp_plain_dev", 1, "pool", 1, 0, WorldConfig(n_levels=3), tasks))
    agent = make_walker("oracle_schema", hp2_wc(), (0, "oracle", "unit"), 0.0)
    with pytest.raises(ValueError):
        play(agent, "oracle_schema", s, [0], meter(), 0, None, "exploration", {})


def test_settled_candidates_agree_with_the_truth():
    from hypergraph_agent.agents.hybrid import observations
    from hypergraph_agent.agents.walker import Walker
    from hypergraph_agent.envs.recipequest import RecipeQuestEnv
    from hypergraph_agent.skills.executor import Executor
    s = schema_stream(n_worlds=1)
    wk = Walker("focused_sample", np.random.default_rng(4))
    env, m = RecipeQuestEnv(), meter()
    tasks = [s.task(i) for i in range(10)]
    for i, t in enumerate(tasks):
        wk.run_episode(Executor(env, m, "exploration", torch.Generator().manual_seed(i)), t, i)
    ws = next(iter(wk.worlds.values()))
    truth = {r.signature(tasks[0].profile): set(r.true_base) for r in tasks[0].world.recipes}
    obs = observations(ws, ws.evidence.logs, 4)
    assert obs
    for o in obs:
        assert o.required <= truth[o.key[1]] and not o.eliminated & truth[o.key[1]]


def test_hp2_agents_refuse_noise():
    from hypergraph_agent.evaluation.walkers import make_walker
    with pytest.raises(ValueError):
        make_walker("hybrid", hp2_wc(), (0, "h", "unit"), 0.1)


# ============================================================================== HP2: configs

def hp2_configs():
    from hypergraph_agent.config import load_config
    return {p.name: load_config(p) for p in sorted(HP2_DIR.glob("*.yaml"))}


def test_hp2_configs_share_one_ledger_and_cover_their_streams():
    from hypergraph_agent.walk import plan
    cfgs = hp2_configs()
    assert len(cfgs) == 10
    decl = {json.dumps([c["run"]["ledger"], c["budget"]["ledger_allocations"],
                        c["budget"]["ledger_adaptive_cap"], c["budget"]["ledger_reporting_cap"]],
                       sort_keys=True) for c in cfgs.values()}
    assert len(decl) == 1
    led = next(iter(cfgs.values()))["budget"]
    assert led["ledger_adaptive_cap"] == sum(led["ledger_allocations"].values())
    adaptive = {"dev": 0, "hp2": 0}
    reporting = 0
    for name, c in cfgs.items():
        p = plan(c)
        assert p["caps"]["per_run_cap_covers_every_stream"], name
        assert c["walker"]["per_run_cap"] == max(p["caps"]["stream_budget_sums"].values()), name
        adaptive[c["run"]["allocation"]] += p["caps"]["worst_case_adaptive"]
        reporting += p["caps"]["worst_case_reporting"]
    assert adaptive["dev"] <= led["ledger_allocations"]["dev"]
    assert adaptive["hp2"] <= led["ledger_allocations"]["hp2"]
    assert reporting <= led["ledger_reporting_cap"]


def test_hp2_configs_agree_with_their_worlds_and_namespaces():
    from hypergraph_agent.agents.hybrid import hybrid_params
    from hypergraph_agent.config import world_config
    for name, c in hp2_configs().items():
        dev = name.startswith("hp2_dev_")
        assert (c["run"]["allocation"] == "dev") == dev and c["eval"]["namespace"].endswith("_dev") == dev
        w, h = world_config(c).schema, hybrid_params(c["walker"]["hybrid"])
        assert (w.n_classes, w.n_families) == (h["n_classes"], h["n_families"])
        assert (h["classes"] == "hidden") == w.hidden_classes
        assert h["scope"] == ("stream" if w.scope == "stream" else "world")
        assert c["walker"]["eval_world_offset_by_seed"] and c["eval"]["tasks"] == {}
        arms = ["focused_sample", "hybrid", "rules_only", "oracle_schema", "maximal", "reference"]
        assert [a["strategy"] for a in c["arms"]] == arms
    names = [c["eval"]["namespace"] for c in hp2_configs().values()]
    assert len(names) == len(set(names))


def test_hp2_configs_load_with_the_walker_schema_and_plan():
    from hypergraph_agent.train import plan
    for c in hp2_configs().values():
        out = plan(c)
        assert out["arms"] and out["interactions"]["per_run_adaptive_cap"] > 0


# ============================================================================== HP2: analysis

def test_hp2_analysis_end_to_end(tmp_path):
    import shutil
    from hypergraph_agent.config import load_config
    from hypergraph_agent.evaluation.hybrid_priors import TABLE_KEYS, hp2_analysis, hp2_contrast
    from hypergraph_agent.evaluation.walkers import run_walker_study
    from hypergraph_agent.training.budget import SessionLedger
    raw = yaml.safe_load((HP2_DIR / "hp2_dev_v1_world.yaml").read_text())
    raw["run"].update(name="hpunit-v1w", seeds=[0], ledger=str(tmp_path / "ledger.json"))
    raw["eval"].update(n_tasks=10, n_worlds=1, episodes_per_world=10)  # episodes 9 and 10 are late
    raw["world"]["n_levels"] = 3
    cfg_path = tmp_path / "configs" / "hybrid_priors" / "unit.yaml"
    cfg_path.parent.mkdir(parents=True)
    cfg_path.write_text(yaml.safe_dump(raw))
    cfg = load_config(cfg_path)
    run_walker_study(cfg, SessionLedger(None, allocations={"dev": 20_000}), str(tmp_path / "runs"))
    res = hp2_analysis(tmp_path / "runs", tmp_path,
                       settings={"v1_world": ("configs/hybrid_priors/unit.yaml", "world")},
                       expected={"v1_world": 1})
    assert res["primary"]["n_units"] == 1 and res["primary"]["verdict"] in ("holds", "does not hold")
    assert res["validity_reference_solves_all"]
    assert res["verdict"] in ("pass", "negligible", "falsified", "inconclusive")
    assert "clingo" in res and not any(res["settings"]["v1_world"]["problems"].values())
    table = res["settings"]["v1_world"]["table"]
    assert set(table["hybrid"]) == set(TABLE_KEYS) and table["focused_sample"]["schema_weight_end"] is None
    assert res["settings"]["v1_world"]["validity"]["overflow_worlds"]["hybrid"] == 0
    for d in (tmp_path / "runs").glob("hpunit-v1w-reference-*"):  # no reference: V1 undetermined
        shutil.rmtree(d) if d.is_dir() else d.unlink()
    res = hp2_analysis(tmp_path / "runs", tmp_path,
                       settings={"v1_world": ("configs/hybrid_priors/unit.yaml", "world")},
                       expected={"v1_world": 1})
    assert res["validity_reference_solves_all"] is None and res["verdict"] is None
    harm = hp2_contrast("H", "m", {"u": {"m": 1.02}}, {"u": {"m": 1.0}}, 1, [], "harm")
    assert harm["verdict"] == "holds"  # 1.02 - 1.0 is below 5% of 1.0
    harm = hp2_contrast("H", "m", {"u": {"m": 1.2}}, {"u": {"m": 1.0}}, 1, [], "harm")
    assert harm["verdict"] == "does not hold"
    assert hp2_contrast("x", "m", {}, {}, 2, [], "below")["verdict"] == "incomplete"


def test_hp2_config_identity_is_one_per_arm_for_first_runs_and_reruns(tmp_path):
    import copy
    from hypergraph_agent.config import load_config
    from hypergraph_agent.evaluation.hybrid_priors import clingo_version, config_identity
    from hypergraph_agent.evaluation.walkers import variant_config
    variant = {"name": "default", "tasks": {}}
    full = load_config(HP2_DIR / "hp2_v1_world.yaml")
    rerun = copy.deepcopy(full)  # what walk does with --seeds 3 --arms hybrid (HP2's rerun rule)
    rerun["run"]["seeds"] = [3]
    rerun["arms"] = [a for a in rerun["arms"] if a["id"] == "hybrid"]
    changed = copy.deepcopy(full)
    changed["walker"]["hybrid"]["model_prior"] = 0.6
    cases = {"s2": (full, 2, "hybrid"), "s3": (full, 3, "hybrid"), "rerun_s3": (rerun, 3, "hybrid"),
             "moved_s4": (full, 4, "hybrid"), "other_arm": (full, 2, "posterior"),
             "changed": (changed, 2, "hybrid")}
    for run_id, (cfg, seed, arm_id) in cases.items():
        arm = next(a for a in cfg["arms"] if a["id"] == arm_id)
        rec = {**variant_config(cfg, seed, variant), "arm": arm, "variant": variant}  # as RunContext records
        if run_id == "moved_s4":
            rec["_source"] = "elsewhere/hp2_v1_world.yaml"
        (tmp_path / run_id).mkdir()
        (tmp_path / run_id / "config.json").write_text(json.dumps(rec, default=str), encoding="utf-8")
    ident = {run_id: config_identity(tmp_path, run_id) for run_id in cases}
    assert ident["s2"] == ident["s3"] == ident["rerun_s3"] == ident["moved_s4"]
    assert ident["other_arm"] != ident["s2"] and ident["changed"] != ident["s2"]
    v = clingo_version()
    assert v is None or isinstance(v, str)


# =============================================================================== hygiene

MY_FILES = (
    "src/hypergraph_agent/priors/__init__.py", "src/hypergraph_agent/priors/size.py",
    "src/hypergraph_agent/priors/posterior.py", "src/hypergraph_agent/priors/schema.py",
    "src/hypergraph_agent/agents/hybrid.py", "src/hypergraph_agent/evaluation/hybrid_priors.py",
    "tests/test_hybrid_priors.py", "docs/studies/HP1_size_prior.md", "docs/studies/HP2_schema_priors.md",
)


def my_files():
    out = [ROOT / f for f in MY_FILES if (ROOT / f).exists()]
    return out + sorted(HP2_DIR.glob("*.yaml")) + sorted(HP1_DIR.glob("*.yaml"))


def test_new_files_keep_lines_short_and_plain():
    dashes = (chr(0x2014), chr(0x2013))  # em and en dash, built here so that this file holds neither
    for f in my_files():
        for n, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert len(line) <= 110, f"{f.name}:{n} is {len(line)} characters"
            assert not any(d in line for d in dashes), f"{f.name}:{n} has a long dash"
            assert not re.search(r"\s--\s", line), f"{f.name}:{n} has a spaced double hyphen"


def test_agent_side_prior_code_reads_no_hidden_fields():
    """The pattern of tests/test_information_boundary.py for the learning packages,
    plus the hidden schema of a world; the privileged oracle lives in
    evaluation/hybrid_priors.py."""
    pattern = re.compile(r"reference_solver|true_base|\.required\(\)|debug_view|\._task\b"
                         r"|world\.schema|task\.world")
    for f in [ROOT / "src/hypergraph_agent/priors" / n for n in ("size.py", "posterior.py", "schema.py")] + \
            [ROOT / "src/hypergraph_agent/agents/hybrid.py"]:
        for n, line in enumerate(f.read_text().splitlines(), 1):
            assert not pattern.search(line.split("#")[0]), f"{f.name}:{n}: {line.strip()}"
