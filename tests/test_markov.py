import json
import math
import pathlib
from itertools import product

import numpy as np
import pytest
import torch

from hypergraph_agent.agents.markov import (
    MARKOV_STRATEGIES, PairwiseEvidence, PPRProposal, all_of, any_of, candidate_ppr, holds, lit, literals,
    make_markov_agent, markov_params, observations, personalized_pagerank, ppr_scores, simplify,
    single_edit_degrees,
    star_scores,
)
from hypergraph_agent.agents.walker import Edit, EpisodeLog, RecipeInfo, Walker, neighbours, simulate
from hypergraph_agent.config import DEFAULTS, load_config
from hypergraph_agent.envs.generator import (
    PROFILE_UNKNOWN, TaskConfig, TaskStream, TaskStreamConfig, WorldConfig,
)
from hypergraph_agent.envs.public_schema import CRAFT, GATHER
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.evaluation import markov_study as ms
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.training.budget import SessionLedger

from helpers import T, meter

ROOT = pathlib.Path(__file__).resolve().parents[1]
WC = {**DEFAULTS["walker"], "consistent_moves": 20, "markov": {}}
P0 = 16 / 41


def stream(seed=7, n_worlds=1):
    return TaskStream(TaskStreamConfig(
        "val", seed, "pool", n_worlds, 0, WorldConfig(),
        TaskConfig(profile=PROFILE_UNKNOWN, depth_min=1, depth_max=3, n_distractor_base=0,
                   observe_items="goal_only")))


def play(agent, tasks):
    m, env, rows = meter(), RecipeQuestEnv(), []
    for i, t in enumerate(tasks):
        ex = Executor(env, m, "exploration", torch.Generator().manual_seed(i))
        rows.append(agent.run_episode(ex, t, i))
    return rows


def twins():
    infos = ms.twin_infos()
    r1, r2 = sorted(infos, key=lambda s: infos[s].effect != T["ingot"])
    return infos, r1, r2


def chain_log(infos, truth, omit, log_id=0):
    r1, r2 = sorted(infos, key=lambda s: infos[s].effect != T["ingot"])
    every = sorted(infos[r1].pool + infos[r2].pool)
    return ms.scripted_log(log_id, infos, truth, T["key"], [b for b in every if b not in omit], [r1, r2])


# ------------------------------------------------------------------ factory

def test_factory_builds_every_strategy_and_refuses_noise_and_unknown_keys():
    for s in MARKOV_STRATEGIES:
        a = make_markov_agent(s, np.random.default_rng(0), WC, 0.0)
        assert a.arm == s and a.label == f"markov:{s}" and isinstance(a, Walker)
    with pytest.raises(ValueError):
        make_markov_agent("hyper_rank", np.random.default_rng(0), WC, 0.1)
    with pytest.raises(ValueError):
        make_markov_agent("nope", np.random.default_rng(0), WC, 0.0)
    with pytest.raises(KeyError):
        markov_params({"restrt": 0.2})
    with pytest.raises(ValueError):
        markov_params({"restart": 0.0})
    assert markov_params({"restart": 1.0})["restart"] == 1.0


def test_make_walker_dispatches_markov_strategies():
    from hypergraph_agent.evaluation.walkers import make_walker
    a = make_walker("pair_rank", WC, (0, "x"), 0.0)
    assert a.arm == "pair_rank" and a.strategy == "focused_sample"
    assert make_walker("hyper_sample_ppr", WC, (0, "x"), 0.0).strategy == "learned_sample"


def test_walk_refuses_any_arm_of_a_config_with_unset_markov_parameters(tmp_path):
    import yaml
    from hypergraph_agent.walk import main
    raw = yaml.safe_load((ROOT / "configs" / "markov" / "m2.yaml").read_text())
    raw["run"].update(ledger=str(tmp_path / "ledger.json"), runs_dir=str(tmp_path / "runs"))
    raw["eval"]["n_tasks"] = 2
    raw["walker"]["markov"].update(restart=None, ppr_seed=None)  # the state before amendment MA1
    path = tmp_path / "m2_unset.yaml"
    path.write_text(yaml.safe_dump(raw))
    with pytest.raises(SystemExit):  # the check sees the whole config, before --arms
        main(["--config", str(path), "--arms", "focused_sample"])
    assert not (tmp_path / "ledger.json").exists() and not (tmp_path / "runs").exists()
    assert main(["--config", str(path), "--dry-run"]) == 0  # a dry run interacts with nothing


def test_m2_config_refuses_to_run_until_m1_is_written_in():
    from hypergraph_agent.evaluation.walkers import make_walker
    from hypergraph_agent.walk import plan
    cfg = load_config(ROOT / "configs" / "markov" / "m2.yaml")
    markov_params(cfg["walker"]["markov"])  # amendment MA1 wrote the M1 selection in
    cfg["walker"]["markov"].update(restart=None, ppr_seed=None)  # the state before MA1
    assert cfg["arms"][0]["strategy"] in MARKOV_STRATEGIES  # the first run stops before any interaction
    assert plan(cfg)["caps"]["per_run_cap_covers_every_stream"]  # the dry run still works
    for a in cfg["arms"]:
        if a["strategy"] in MARKOV_STRATEGIES:
            with pytest.raises(ValueError, match="M1"):
                make_walker(a["strategy"], cfg["walker"], (0, a["id"]), 0.0)


# ------------------------------------------------------------ observations

def test_formulas_simplify_and_list_literals():
    f = any_of([lit("r", 1), all_of([lit("s", 2), lit("s", 3)])])
    assert literals(f) == {("r", 1), ("s", 2), ("s", 3)}
    assert simplify(f, false={("r", 1)}) == ("and", (lit("s", 2), lit("s", 3)))
    assert simplify(f, true={("s", 2), ("s", 3)}) is True
    assert simplify(f, false={("r", 1), ("s", 3)}) is False
    assert any_of([]) is False and all_of([]) is True
    assert holds(f, {"r": frozenset({1})}) and not holds(f, {"s": frozenset({2})})


def test_chain_failure_is_a_disjunction_and_success_eliminates_down_the_chain():
    infos, r1, r2 = twins()
    truth = {r1: {T["ore"]}, r2: {T["oil"]}}
    fail = observations(chain_log(infos, truth, {T["ore"], T["stone"]}), infos)
    assert [o[0] for o in fail] == ["failure"]
    assert literals(fail[0][2]) == {(r1, T["ore"]), (r2, T["stone"])}
    ok = observations(chain_log(infos, truth, {T["stone"], T["sand"]}), infos)
    assert ok[0][0] == "success" and ok[0][2] == {(r2, T["stone"]), (r1, T["sand"])}


def test_alternative_producers_give_an_and_and_no_elimination():
    i_a = RecipeInfo("ingot-a", T["ingot"], (), (T["ore"], T["fuel"]), None)
    i_b = RecipeInfo("ingot-b", T["ingot"], (), (T["sand"], T["wood"]), None)
    key = RecipeInfo("key", T["key"], (T["ingot"],), (T["stone"], T["oil"]), None)
    infos = {i.signature: i for i in (i_a, i_b, key)}
    lg = EpisodeLog(0, T["key"], frozenset())
    for kind, x in [(GATHER, T["fuel"]), (GATHER, T["wood"]), (GATHER, T["stone"]), (GATHER, T["oil"]),
                    (CRAFT, "ingot-a"), (CRAFT, "ingot-b"), (CRAFT, "key")]:
        lg.steps.append((kind, x))
        lg.goal_obs.append(False)
        lg.effect_obs.append(None)
    (kind, _, f), = observations(lg, infos)
    assert kind == "failure" and f == ("and", (lit("ingot-a", T["ore"]), lit("ingot-b", T["sand"])))
    lg.goal_obs[-1] = True
    (kind, _, elim), = observations(lg, infos)
    assert kind == "success" and elim == frozenset()  # two attempted producers: nothing certain


# ------------------------------------------------------------ pairwise rule

def test_pairwise_update_rule():
    infos, r1, r2 = twins()
    pe = PairwiseEvidence(infos, prior_strength=2.0)
    assert pe.prior(r1) == pytest.approx(P0)
    pe.observe(("failure", 0, any_of([lit(r1, T["ore"]), lit(r2, T["stone"])])))
    assert pe.weight(r1, T["ore"]) == pytest.approx((2 * P0 + 0.5) / 2.5)
    assert pe.weight(r1, T["fuel"]) == pytest.approx(P0)
    pe.observe(("success", 1, frozenset({(r2, T["stone"])})))
    assert pe.weight(r2, T["stone"]) == 0.0
    assert pe.weight(r1, T["ore"]) == pytest.approx((2 * P0 + 0.5) / 2.5)  # never re-attributed
    pe.observe(("failure", 2, any_of([lit(r1, T["ore"]), lit(r2, T["stone"])])))
    assert pe.weight(r1, T["ore"]) == 1.0 and pe.counts["forced"] == 1  # one open edge: forced
    pe.observe(("failure", 3, any_of([lit(r1, T["ore"]), lit(r1, T["fuel"])])))
    assert pe.counts["explained"] == 1 and pe.weight(r1, T["fuel"]) == pytest.approx(P0)
    # a forced edge settles, and what is left of the formula is still blamed
    pe.observe(("failure", 4, all_of([lit(r2, T["oil"]), any_of([lit(r2, T["salt"]), lit(r2, T["wax"])])])))
    assert pe.weight(r2, T["oil"]) == 1.0 and pe.weight(r2, T["salt"]) == pytest.approx((2 * P0 + 0.5) / 2.5)
    # class bounds: three required candidates eliminate the rest; one open candidate left is required
    for b in ("fuel", "sand"):
        pe._settle((r1, T[b]), True)
    assert all(pe.weight(r1, b) == 0.0 for b in infos[r1].pool if b not in (T["ore"], T["fuel"], T["sand"]))
    q = PairwiseEvidence(infos)
    for b in infos[r2].pool[:-1]:
        q._settle((r2, b), False)
    assert q.weight(r2, infos[r2].pool[-1]) == 1.0


# ----------------------------------------------------------------- PageRank

def test_ppr_is_the_normalized_successor_representation_and_restart_one_is_the_edge_weight():
    rng = np.random.default_rng(0)
    W = rng.random((6, 6))
    W = W + W.T
    W[5, :] = W[:, 5] = 0.0  # an isolated node jumps to the seed
    s = np.eye(6)[1]
    for a in (0.15, 0.5):
        pi = personalized_pagerank(W, s, a)
        P = W / np.maximum(W.sum(1, keepdims=True), 1e-300)
        P[5] = s
        sr = np.linalg.inv(np.eye(6) - (1 - a) * P)[1]
        assert pi.sum() == pytest.approx(1.0) and np.allclose(pi, a * sr)
    infos = {"r": RecipeInfo("r", 0, (), (2, 3, 4), None)}
    idx = {n: n for n in range(6)}
    one = candidate_ppr(idx, W, infos, ["r"], 1.0, lambda sig: (1,))
    assert one == {("r", b): W[1, b] for b in (2, 3, 4)}
    pe = PairwiseEvidence(twins()[0])
    pe.observe(("failure", 0, any_of([lit(twins()[1], T["ore"]), lit(twins()[1], T["fuel"])])))
    w1 = star_scores(pe.infos, pe, 1.0, "goal", [T["key"]])
    assert all(w1[e] == pe.weight(*e) for e in w1)  # no propagation: the pairwise weights


def test_single_edit_degrees():
    # single candidates are pairwise adjacent (swaps); {0, 4} only to {0} (one removal): degrees 4, 3, 3, 3, 1
    k = [(frozenset({x}),) for x in range(4)] + [(frozenset({0, 4}),)]
    assert [single_edit_degrees(k)[x] for x in k] == [4, 3, 3, 3, 1]
    f = frozenset
    two = [(f({0}), f({5})), (f({1}), f({5})), (f({1}), f({6}))]
    assert [single_edit_degrees(two)[x] for x in two] == [1, 2, 1]  # one recipe changes at a time


def test_hyper_rank_normalization_removes_the_size_preference_of_the_class():
    from hypergraph_agent.topology.inference import enumerate_hypotheses
    info = RecipeInfo("r", T["ingot"], (), tuple(range(6)), None)
    keys = [(frozenset(h),) for h in enumerate_hypotheses(6, 3)]
    deg_class = {x: len(neighbours({"r": x[0]}, {"r": info}, ["r"], 3)) for x in keys}
    assert sorted({len(x[0]): d for x, d in deg_class.items()}.items()) == [(1, 10), (2, 14), (3, 12)]
    deg = single_edit_degrees(keys)
    assert deg == deg_class  # every node consistent: raw degree (stationary PageRank) favours size two
    assert {deg[x] / deg_class[x] for x in keys} == {1.0}  # the normalized score has no size preference


# ------------------------------------------------------------ rankings, M1

def test_consistent_replay_matches_simulate():
    s = stream()
    w = Walker("focused_sample", np.random.default_rng(1), consistent_moves=10)
    play(w, [s.task(i) for i in range(6)])
    ws = next(iter(w.worlds.values()))
    rng = np.random.default_rng(2)
    for _ in range(40):
        node = {sig: ws.hypotheses[sig][int(rng.integers(len(ws.hypotheses[sig])))] for sig in ws.hidden()}
        for lg in ws.evidence.logs:
            assert ms.consistent(node, lg, ws.infos) == (simulate(node, lg, ws.infos)[0] == [])


def test_posterior_gibbs_agrees_with_enumeration():
    infos, r1, r2 = twins()
    truth = {r1: {T["ore"], T["fuel"]}, r2: {T["oil"]}}
    logs = [chain_log(infos, truth, om, i) for i, om in enumerate(
        [{T["ore"], T["stone"]}, {T["sand"], T["salt"]}, {T["fuel"]}, {T["wax"], T["clay"]}])]
    exact, d1 = ms.posterior_marginals(infos, logs, np.random.default_rng(0))
    assert d1["exact_recipes"] == 2
    gibbs, d2 = ms.posterior_marginals(infos, logs, np.random.default_rng(0), exact_work=0, starts=3,
                                       sweeps=60)
    assert d2["sampled_recipes"] == 2 and d2["chains"] == 3
    for e, p in exact.items():
        assert gibbs.get(e, 0.0) == pytest.approx(p, abs=0.06)
    assert exact[(r1, T["fuel"])] == pytest.approx(1.0) and exact[(r2, T["stone"])] < 1.0


def test_exact_posterior_equals_brute_force_on_a_three_level_chain():
    names = (("ingot", (), ("ore", "fuel", "sand", "wood")),
             ("key", ("ingot",), ("clay", "fiber", "stone", "oil")),
             ("glass", ("key",), ("salt", "wax", "resin", "flint")))
    infos = {e: RecipeInfo(e, T[e], tuple(T[i] for i in its), tuple(sorted(T[b] for b in pool)), None)
             for e, its, pool in names}
    rng = np.random.default_rng(4)
    every = sorted(b for i in infos.values() for b in i.pool)
    for trial in range(4):
        truth = {s: set(rng.choice(i.pool, int(rng.integers(1, 4)), replace=False).tolist())
                 for s, i in infos.items()}
        logs = []
        for n in range(7):
            chain = ["ingot", "key", "glass"][: 1 + n % 3]
            gathers = [b for b in every if rng.random() > 0.25]
            logs.append(ms.scripted_log(n, infos, truth, T[chain[-1]], gathers,
                                        chain))
        post, d = ms.posterior_marginals(infos, logs, np.random.default_rng(0))
        assert d["sampled_recipes"] == 0
        sigs = sorted(infos)
        counts, total = {}, 0
        for combo in product(*(ms._class(infos[s], 3) for s in sigs)):
            node = dict(zip(sigs, combo))
            if all(simulate(node, lg, infos)[0] == [] for lg in logs):
                total += 1
                for s, h in node.items():
                    for b in h:
                        counts[(s, b)] = counts.get((s, b), 0) + 1
        for s in sigs:
            for b in infos[s].pool:
                assert post.get((s, b), 0.0) == pytest.approx(counts.get((s, b), 0) / total, abs=1e-9)


def test_formula_probability_is_exact_with_repeated_literals():
    a, b, c, x = ("r", 1), ("r", 2), ("s", 3), ("t", 4)
    f = all_of([any_of([lit(*a), lit(*x)]), any_of([lit(*b), lit(*x)]), lit(*c)])  # x shared by two branches
    q = {a: 0.3, b: 0.6, c: 0.8, x: 0.4}
    brute = 0.0
    for vals in product((0, 1), repeat=4):
        assign = dict(zip((a, b, c, x), vals))
        if holds(f, {s: frozenset(e[1] for e, v in assign.items() if v and e[0] == s) for s in "rst"}):
            brute += math.prod(q[e] if v else 1 - q[e] for e, v in assign.items())
    assert ms.formula_prob(f, q) == pytest.approx(brute) and ms._tree_prob(f, q) != pytest.approx(brute)


def test_andor_prior_is_the_class_prior_and_clauses_explain_away():
    infos, r1, r2 = twins()
    truth = {r1: {T["ore"]}, r2: {T["oil"]}}
    bf = chain_log(infos, truth, set())
    q, d = ms.andor_marginals(infos, [bf])
    assert d["converged"] and all(v == pytest.approx(P0, abs=1e-6) for v in q.values())
    q, _ = ms.andor_marginals(infos, [bf, chain_log(infos, truth, {T["ore"], T["stone"]}, 1),
                                      chain_log(infos, truth, {T["stone"]}, 2)])
    assert q[(r1, T["ore"])] == pytest.approx(1.0, abs=1e-6) and q[(r2, T["stone"])] == 0.0


def test_m1_star_scores_are_what_the_agents_compute():
    infos, r1, r2 = twins()
    truth = {r1: {T["ore"]}, r2: {T["oil"]}}
    omits = [set(), {T["ore"], T["stone"]}, {T["fuel"]}]
    logs = [chain_log(infos, truth, om, i) for i, om in enumerate(omits)]
    res = ms.rank_scores(infos, logs, np.random.default_rng(0), restarts=(0.3, 1.0))
    pe = PairwiseEvidence(infos).observe_logs(logs)
    for how in ("recipe", "effect", "goal"):
        for a in (0.3, 1.0):
            agent = ppr_scores(infos, pe, markov_params({"restart": a, "ppr_seed": how}), logs)
            assert all(res["scores"][f"star_ppr:{how}@{a:g}"][e] == agent[e] for e in res["scores"]["random"])


def test_m1_decision_with_an_equivalence_margin():
    rng = np.random.default_rng(0)
    assert ms.m1_decision(list(0.05 + rng.normal(0, 0.02, 36)))["decision"] == "posterior better"
    assert ms.m1_decision(list(-0.05 + rng.normal(0, 0.02, 36)))["decision"] == "PPR better"
    small = ms.m1_decision([0.012, 0.008] * 18)  # significant but inside the margin: the difference wins
    assert small["decision"] == "posterior better" and small["equivalent"]
    eq =ms.m1_decision([0.004, -0.004] * 18)
    assert eq["decision"] == "equivalent" and eq["equivalent"] and not eq["posterior_better"]
    wide = ms.m1_decision([0.2, -0.2, 0.1, -0.1] * 9)
    assert wide["decision"] == "inconclusive" and wide["margin"] == 0.03
    assert ms.m1_decision([])["decision"] is None


def test_auroc_and_gathering_cost_handle_ties():
    sc = {1: 0.9, 2: 0.5, 3: 0.5, 4: 0.1}
    assert ms.auroc(sc, {1}) == 1.0 and ms.auroc(sc, {4}) == 0.0 and ms.auroc(sc, {2}) == pytest.approx(0.5)
    assert ms.gathering_cost(sc, {1}) == 1.0 and ms.gathering_cost(sc, {2}) == pytest.approx(1 + 1.5)
    tied = {b: 0.0 for b in range(6)}
    assert ms.auroc(tied, {0, 1}) == 0.5
    assert ms.gathering_cost(tied, {0}) == pytest.approx(3.5)
    assert ms.gathering_cost(tied, {0, 1, 2}) == pytest.approx(5.25)


def test_structural_twins():
    s = ms.structural_twins()
    assert s["clique_equal"] and s["star_differs"] and s["plans_differ"]
    assert (s["one_triple"]["cheapest_bases"], s["three_pairs"]["cheapest_bases"]) == (3, 2)
    assert (s["one_triple"]["h_add_goal"], s["three_pairs"]["h_add_goal"]) == (4, 3)


def test_behavioural_twins_separate_only_jointly():
    b = ms.behavioural_twins()
    for stage in ("after_brute_force", "after_omission"):
        st = b["stages"][stage]
        assert st["observations_identical"] and not any(st["ranking_differs"].values())
    st = b["stages"]["after_probe"]
    assert not st["observations_identical"]
    diff = st["ranking_differs"]
    assert diff["posterior"] and diff["andor"]
    assert not diff["pair_weight"] and not diff["fact_ppr:effect@0.15"]
    assert not any(diff[f"star_ppr:{how}@0.15"] for how in ("recipe", "effect", "goal"))
    a, bb = st["worlds"]["A"]["scores"], st["worlds"]["B"]["scores"]
    assert a["posterior"] == pytest.approx(1.0) and bb["posterior"] == pytest.approx(P0, abs=1e-6)
    assert a["pair_weight"] == bb["pair_weight"]


# ------------------------------------------------------------------ agents

@pytest.mark.parametrize("strategy", MARKOV_STRATEGIES)
def test_markov_agents_complete_tasks(strategy):
    s = stream(seed=7)
    a = make_markov_agent(strategy, np.random.default_rng(3), WC, 0.0)
    rows = play(a, [s.task(i) for i in range(6)])
    assert any(r["success"] for r in rows) and all(r["status"] != "budget_exhausted" for r in rows)
    assert all(r["choose_seconds"] >= 0 for r in rows)
    ws = next(iter(a.worlds.values()))
    assert len(ws.evidence.logs) == 6  # episodic: every log is kept as evidence
    node, st = a.choose_node(ws, None)
    assert set(node) == set(ws.hidden())
    if strategy.startswith("hyper"):
        assert ws.evidence.violations(node) == 0 and st["evals"] > 0
    if strategy.startswith("pair"):
        pe = a.pairwise(ws, None)
        assert all(pe.weight(sig, b) > 0 for sig, h in node.items() for b in h)  # never an eliminated edge
        assert all(1 <= len(h) <= 3 for h in node.values())


class _W:  # the parts of a WorldState the pair marks read
    def __init__(self, infos):
        self.infos = infos


def test_pair_marks_are_one_per_edge_rule_shared_by_both_pair_cells():
    from hypergraph_agent.agents.markov import PairAgent
    infos, r1, r2 = twins()
    truth = {r1: {T["fuel"]}, r2: {T["stone"]}}
    log = chain_log(infos, truth, {T["ore"], T["stone"]}, log_id=5)  # fails: (ingot, ore) or (key, stone)
    pe = PairwiseEvidence(infos).observe_logs([log])
    base = {r1: frozenset({T["fuel"]}), r2: frozenset({T["oil"]})}
    out = {}
    for arm in ("pair_sample", "pair_rank"):
        a = PairAgent(arm, np.random.default_rng(0), markov_params({}))
        node = a.marks(dict(base), _W(infos), log, pe)
        # equal weights: ties go to the failed (goal) recipe's own candidate, kept for the episode
        assert a._marks == [(r2, T["stone"])] and node[r2] == {T["oil"], T["stone"]} and node[r1] == base[r1]
        assert a.marks(dict(base), _W(infos), log, pe) == node and a._counts["pair_marks"] == 1  # once
        out[arm] = node
        # at max_size the lowest-weight member goes out
        full = {r1: base[r1], r2: frozenset({T["oil"], T["salt"], T["wax"]})}
        b = PairAgent(arm, np.random.default_rng(0), markov_params({}))
        swapped = b.marks(full, _W(infos), log, pe)[r2]
        assert len(swapped) == 3 and T["stone"] in swapped and T["oil"] in swapped
        nxt = chain_log(infos, truth, set(), log_id=6)  # a new episode clears the marks
        assert b.marks(dict(base), _W(infos), nxt, pe) == base and b._marks == []
    assert out["pair_sample"] == out["pair_rank"]


@pytest.mark.parametrize("arm", ["pair_sample", "pair_rank"])
def test_pair_cells_count_marks_and_residual_refutations(arm):
    wc = {**WC, "markov": {"restart": 0.15, "ppr_seed": "goal"}}
    a = make_markov_agent(arm, np.random.default_rng(3), wc, 0.0)
    rows = play(a, [stream(seed=s).task(i) for s in (7, 8) for i in range(8)])
    for r in rows:
        assert {"pair_marks", "refuted_replans", "repeated_nodes"} <= set(r)
        assert 0 <= r["repeated_nodes"] <= r["refuted_replans"] <= r["replans"]
    assert sum(r["pair_marks"] for r in rows) > 0


def test_pair_sample_draws_over_the_class_with_odds_weights():
    from hypergraph_agent.agents.markov import PairAgent
    from hypergraph_agent.topology.inference import enumerate_hypotheses
    a = PairAgent("pair_sample", np.random.default_rng(0), markov_params({}))
    pool = tuple(range(6))
    hyps = [frozenset(h) for h in enumerate_hypotheses(6, 3)]
    flat = {b: P0 for b in pool}
    n = 41 * 600
    counts = {}
    for _ in range(n):
        h = a.draw(hyps, flat, P0, list(pool), 2)
        counts[h] = counts.get(h, 0) + 1
    assert len(counts) == 41 and max(counts.values()) / min(counts.values()) < 1.5  # the uniform class prior
    wt = {0: 1.0, 1: 0.0, 2: 0.6, 3: P0, 4: P0, 5: P0}
    r0, r2 = P0 / (1 - P0), 0.6 / 0.4
    draws = [a.draw(hyps, wt, P0, list(pool), 2) for _ in range(6000)]
    assert all(0 in h and 1 not in h for h in draws)  # required in, eliminated out
    rho = r2 / r0  # h = {0} + S, S within {2, 3, 4, 5}, |S| <= 2: 7 sets without 2, 4 with it (weight rho)
    p2 = 4 * rho / (7 + 4 * rho)
    assert np.mean([2 in h for h in draws]) == pytest.approx(p2, abs=0.03)


def test_hyper_rank_counts_its_work():
    s = stream(seed=7)
    wc = {**WC, "markov": {"rank_neighbours": False}}
    h = make_markov_agent("hyper_rank", np.random.default_rng(3), wc, 0.0)
    play(h, [s.task(i) for i in range(4)])
    ws = next(iter(h.worlds.values()))
    _, st = h.choose_node(ws, None)
    assert st["rank_evals"] == 0 and st["rank_members"] == st["rank_visited"]


def test_ppr_proposal_follows_its_weights():
    infos, r1, r2 = twins()

    class W:  # the parts of a WorldState a proposal reads
        pass

    w = W()
    w.infos = infos
    truth = {r1: {T["ore"]}, r2: {T["oil"]}}
    logs = [chain_log(infos, truth, {T["ore"], T["stone"]})]
    prop = PPRProposal(markov_params({"proposal_floor": 0.0}))
    s = prop.scores(w, logs)
    assert max(s[(r1, b)] for b in infos[r1].pool) == pytest.approx(1.0)
    edits = {Edit(r1, None, T["ore"]): 1, Edit(r1, None, T["fuel"]): 1}
    rng = np.random.default_rng(0)
    picks = [prop.propose(None, w, logs, edits, None, None, rng).add for _ in range(2000)]
    share = picks.count(T["ore"]) / len(picks)
    expected = s[(r1, T["ore"])] / (s[(r1, T["ore"])] + s[(r1, T["fuel"])])
    assert share == pytest.approx(expected, abs=0.04) and s[(r1, T["ore"])] > s[(r1, T["fuel"])]


# -------------------------------------------------------- runners, analysis

def test_logged_maximal_acts_like_maximal_and_keeps_logs():
    s = stream(seed=11)
    tasks = [s.task(i) for i in range(4)]
    plain = play(Walker("maximal", np.random.default_rng(0)), tasks)
    logged = ms.LoggedMaximal(np.random.default_rng(0))
    rows = play(logged, tasks)
    assert [r["primitive_length"] for r in rows] == [r["primitive_length"] for r in plain]
    assert len(next(iter(logged.worlds.values())).evidence.logs) == 4


def test_collection_writes_evidence_and_the_benchmark_reads_it(tmp_path):
    cfg = load_config(ROOT / "configs" / "markov" / "dev_collect.yaml")
    cfg["eval"].update(n_tasks=4, n_worlds=1, episodes_per_world=4)
    cfg["walker"]["per_run_cap"] = 200
    ledger = SessionLedger(None, {"dev": 1000})
    res = ms.collect(cfg, ledger, str(tmp_path), stamp="t")
    assert [r["strategy"] for r in res] == ["focused_sample", "maximal"] and all(r["complete"] for r in res)
    assert ledger.summary()["by_allocation"]["dev"] == res[0]["interactions"]
    assert ledger.summary()["reporting_total"] == res[1]["interactions"]
    worlds, sel = ms.load_evidence(tmp_path, "markov-dev-m1-")
    assert len(worlds) == 2 and all(len(w["logs"]) == 4 for w in worlds) and sel["ignored"] == []
    assert all(set(w["truth"]["requirements"]) == set(w["infos"]) for w in worlds)
    assert all(len(w["registered"]) == 4 and set(ms.infos_at(w, 1)) <= set(w["infos"]) for w in worlds)
    out = ms.rank_benchmark(worlds, checkpoints=(2, 4), restarts=(0.15, 1.0), selection_seeds=(0,),
                            test_seeds=(0,), posterior_kw={"starts": 2, "sweeps": 10, "burn": 2})
    row = out["table"]["test:focused_sample@4"]
    assert row["n_worlds"] == 1 and 0 <= row["posterior"]["auroc"]["estimate"] <= 1
    # brute force: no information
    assert out["table"]["test:maximal@4"]["posterior"]["auroc"]["estimate"] == pytest.approx(0.5)
    assert out["restart_selection"]["star_ppr"]["selected"]["restart"] in (0.15, 1.0)
    assert len(out["restart_selection"]["star_ppr"]["by_setting"]) == 6
    assert out["primary_verdict"] in ("posterior better", "PPR better", "equivalent", "inconclusive")
    assert out["primary"]["n"] == 1 and out["primary_decision"]["margin"] == 0.03
    assert ms.load_runs(tmp_path, "markov-dev")[0] == []  # evidence runs are not M2 runs


def _fake_run(tmp_path, name, strategy, seed, status, start, chash="h", n_tasks=2):
    d = tmp_path / name
    d.mkdir()
    (d / "manifest.json").write_text(json.dumps({
        "run_id": name, "seed": seed, "arm": strategy, "status": status, "stop_reason": "tasks_done",
        "start_time": start, "config_hash": chash, "source": {"commit": "c", "loaded_code_sha256": "x"},
        "wallclock_s": 1.0}))
    (d / "config.json").write_text(json.dumps({"arm": {"id": strategy, "strategy": strategy},
                                               "eval": {"n_tasks": n_tasks}}))
    rows = [{"world_key": "W", "world_episode": e, "success": True, "reference_length": 10,
             "primitive_length": 12, "evals": 1, "replans": 1} for e in range(n_tasks)]
    (d / "eval.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")


def test_rerun_rule_replaces_an_incomplete_run_only_by_a_matching_completed_one(tmp_path):
    _fake_run(tmp_path, "markov-m2-a", "pair_rank", 0, "running", "1")
    _fake_run(tmp_path, "markov-m2-b", "pair_rank", 0, "completed", "2")
    _fake_run(tmp_path, "markov-m2-c", "pair_rank", 0, "completed", "3")
    _fake_run(tmp_path, "markov-m2-d", "pair_sample", 0, "running", "1")
    _fake_run(tmp_path, "markov-m2-e", "pair_sample", 0, "completed", "2", chash="other")
    runs, sel = ms.load_runs(tmp_path, "markov-m2")
    picked = {r["strategy"]: r["run_id"] for r in runs}
    assert picked == {"pair_rank": "markov-m2-b", "pair_sample": "markov-m2-d"}
    assert sel["reruns_used"] == [{"incomplete": "markov-m2-a", "rerun": "markov-m2-b"}]
    assert sorted(sel["ignored"]) == ["markov-m2-c", "markov-m2-e"]
    f = ms.factorial(runs)  # cells missing or incomplete: no verdict and no selection
    assert f["primary_verdict"] is None and f["chosen"] is None
    assert f["winning_cell"] is None and f["winning_vs_runner_up"] is None and f["component_message"] is None


def _rows(cost, rng, n_worlds=12, success=True):
    rows = []
    for w in range(n_worlds):
        noise = rng.normal(0, 0.01)
        for e in range(16):
            rows.append({"world_key": f"W{w}", "world_episode": e, "success": success,
                         "reference_length": 100,
                         "primitive_length": round(100 * (cost + noise)), "evals": 10, "replans": 2,
                         "choose_seconds": 0.01})
    return rows


def test_factorial_contrasts_and_selection_rule():
    rng = np.random.default_rng(0)
    cost = {"focused_sample": 1.5, "hyper_rank": 1.45, "pair_sample": 1.9, "pair_rank": 1.8,
            "hyper_sample_ppr": 1.5, "maximal": 1.7, "reference": 1.0}
    runs = [{"run_id": s, "strategy": s, "seed": 0, "rows": _rows(c, rng), "complete": True,
             "wallclock_s": 10.0}
            for s, c in cost.items()]
    s = ms.factorial(runs, expected_worlds=12)
    rep = s["effects"]["representation (hyper - pair)"]["late_cost"]["estimate"]
    sel = s["effects"]["selector (rank - sample)"]["late_cost"]["estimate"]
    inter = s["effects"]["interaction"]["late_cost"]["estimate"]
    assert rep == pytest.approx(0.5 * ((1.5 - 1.9) + (1.45 - 1.8)), abs=0.01)
    assert sel == pytest.approx(0.5 * ((1.45 - 1.5) + (1.8 - 1.9)), abs=0.01)
    assert inter == pytest.approx((1.45 - 1.5) - (1.8 - 1.9), abs=0.01)
    assert s["effects"]["selector (rank - sample)"]["late_cost_selection"]["level"] == 0.975
    assert s["chosen"] == {"representation": "hyper", "selector": "rank"}
    assert s["winning_cell"] == "hyper_rank"
    assert s["winning_vs_runner_up"]["runner_up"] == "focused_sample"
    assert s["winning_vs_runner_up"]["status"] == "confirmed"
    assert s["primary_verdict"] == "hyper lowers late cost" and s["complete"]
    assert s["extra_vs_focused_sample"]["evals"]["estimate"] == 0.0
    assert s["arms"]["pair_rank"]["choose_seconds_per_replan"] == pytest.approx(0.01 * 16 / 32)
    flat = [{"run_id": x, "strategy": x, "seed": 0, "rows": _rows(1.5, rng), "complete": True,
             "wallclock_s": 10.0} for x in cost]
    f = ms.factorial(flat, expected_worlds=12)
    assert f["component_message"] == "no component brings more" and f["winning_cell"] is None


def test_markov_configs_share_one_ledger_declaration():
    decl = set()
    for p in sorted((ROOT / "configs" / "markov").glob("*.yaml")):
        cfg = load_config(p)
        b = cfg["budget"]
        decl.add((cfg["run"]["ledger"], tuple(sorted(b["ledger_allocations"].items())),
                  b["ledger_adaptive_cap"],
                  b["ledger_reporting_cap"]))
        assert cfg["train"]["tasks"]["failure_prob"] == 0.0 and cfg["walker"]["eval_world_offset_by_seed"]
        assert b["ledger_adaptive_cap"] == sum(b["ledger_allocations"].values())
    assert len(decl) == 1


def test_study_m_lines_are_at_most_110_characters():
    src = ROOT / "src" / "hypergraph_agent"
    files = (src / "agents" / "markov.py", src / "evaluation" / "markov_study.py", src / "markov_study.py",
             ROOT / "tests" / "test_markov.py")
    for p in files:
        long = [i for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1) if len(line) > 110]
        assert not long, (p.name, long)
