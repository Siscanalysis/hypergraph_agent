"""Study RL: exact rule learners, classical learners, collection on X2's machinery and the analysis."""

import copy
import json
import math
import pathlib
import re

import numpy as np
import pytest
import torch
import yaml

from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.inference_baselines import collection as x2col
from hypergraph_agent.inference_baselines import learners as x2learners
from hypergraph_agent.inference_baselines import traces
from hypergraph_agent.rules import analysis as an
from hypergraph_agent.rules import classical, dnf, learners
from hypergraph_agent.rules import collection as col
from hypergraph_agent.rules.__main__ import main
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.training.budget import SessionLedger

from helpers import meter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "rule_learners").glob("*.yaml"))
POOL = (1, 3, 5, 7, 9, 11)


def planted(truth, n=25, seed=0, slip=0.0):
    """Present sets of successes and failures of a planted conjunction (failures
    with the truth present are slips)."""
    rng = np.random.default_rng(seed)
    succ, fail = [], []
    for _ in range(n):
        x = frozenset(c for c in POOL if rng.random() < 0.6)
        ok = set(truth) <= x and not rng.random() < slip
        (succ if ok else fail).append(x)
    return succ, fail


def tiny_cfg(tmp_path, policies=("focused_sample", "maximal"), episodes=4, name="rl-dev"):
    raw = yaml.safe_load((ROOT / "configs" / "rule_learners" / "rl_dev.yaml").read_text())
    raw["run"].update(runs_dir=str(tmp_path / "runs"), ledger=str(tmp_path / "ledger.json"), name=name)
    raw["policies"] = list(policies)
    for s in raw["sets"]:
        s.update(n_worlds=1, episodes_per_world=episodes)
    return col.validate(raw)


def x2_tiny(tmp_path, episodes=4, bench=False):
    """X2's own collection (its dev config: set gen_obs and, with ``bench``, its
    Crafter set, one world each) into a scratch folder."""
    raw = yaml.safe_load((ROOT / "configs" / "inference_baselines" / "dev.yaml").read_text())
    raw["run"].update(runs_dir=str(tmp_path / "runs"), ledger=str(tmp_path / "x2ledger.json"))
    raw["policies"] = list(x2col.POLICIES)
    keep = ("gen_obs", "bench_crafter") if bench else ("gen_obs",)
    raw["sets"] = [dict(s, n_worlds=1, episodes_per_world=episodes) for s in raw["sets"] if s["name"] in keep]
    return x2col.collect(x2col.validate(raw), SessionLedger(None, {"dev": 10_000}), stamp="x")


# --------------------------------------------------------------- exact DNF

def test_dnf_recovers_a_planted_conjunction_with_both_solvers():
    succ, fail = planted({3, 7})
    p = dnf.make_problem(POOL, succ, fail)
    a, b = dnf.solve_clingo(p), dnf.solve_rc2(p)
    assert a.status == "optimal" and a.rules == (((3, 7),),) and a.cost == 2 and dnf.agree(a, b)
    assert dnf.rule_scores(a.rules, POOL) == {1: 0.0, 3: 1.0, 5: 0.0, 7: 1.0, 9: 0.0, 11: 0.0}
    assert all(dnf.covers(a.rules[0], x) for x in succ) and not any(dnf.covers(a.rules[0], x) for x in fail)


def test_dnf_reports_every_optimal_rule_and_needs_a_term():
    p = dnf.make_problem(POOL, [{3, 7, 9}], [])  # any single candidate of the success is smallest
    res = dnf.solve_clingo(p)
    assert res.rules == (((3,),), ((7,),), ((9,),)) and res.cost == 1
    assert dnf.rule_scores(res.rules, POOL)[3] == pytest.approx(1 / 3)
    empty = dnf.solve_clingo(dnf.make_problem(POOL, [], []))  # requirements are publicly nonempty
    assert empty.status == "optimal" and len(empty.rules) == len(POOL) and empty.cost == 1
    full = dnf.make_problem(POOL, [{3, 7}], [set(POOL)])  # a failure with every candidate present
    assert dnf.solve_clingo(full).status == "unsat" == dnf.solve_rc2(full).status


def test_dnf_terms_have_one_answer_set_each():
    # successes {1, 9} and {3, 11}; the failure {9, 11} rules out the single terms (9) and (11)
    p = dnf.make_problem(POOL, [{1, 9}, {3, 11}], [{9, 11}])
    import clingo
    ctl = clingo.Control(["0", "--opt-mode=optN"], logger=lambda c, m: None)
    ctl.add("base", [], dnf.asp_program(p))
    ctl.ground([("base", [])])
    proven = []
    ctl.solve(on_model=lambda m: proven.append(m.optimality_proven) or True)
    res = dnf.solve_clingo(p)
    assert res.cost == 2 and len(res.rules) == sum(proven)  # no permutation of the terms is repeated
    assert ((1,), (3,)) in res.rules and dnf.agree(res, dnf.solve_rc2(p))
    assert dnf.rule_scores([((1,), (3,))], POOL)[1] == 0.5


def test_rc2_agrees_with_clingo_on_random_problems():
    rng = np.random.default_rng(5)
    for trial in range(40):
        truth = set(rng.choice(POOL, int(rng.integers(1, 6)), replace=False).tolist())
        succ, fail = planted(truth, n=int(rng.integers(0, 14)), seed=trial, slip=0.2)
        p = dnf.make_problem(POOL, succ, fail, epsilon=0.1 if trial % 2 else 0.0)
        assert dnf.agree(dnf.solve_clingo(p), dnf.solve_rc2(p)), trial


def test_soft_dnf_explains_slips_and_is_the_hard_problem_without_them():
    succ, fail = planted({3, 7}, n=40, seed=2, slip=0.1)
    assert any({3, 7} <= x for x in fail)  # at least one slip
    hard = dnf.solve_clingo(dnf.make_problem(POOL, succ, fail))
    soft = dnf.solve_clingo(dnf.make_problem(POOL, succ, fail, epsilon=0.1))
    assert hard.rules != (((3, 7),),) and soft.rules == (((3, 7),),)
    lit, slip = dnf.make_problem(POOL, succ, fail, epsilon=0.1).weights()
    assert (lit, slip) == (round(1000 * math.log(6)), round(1000 * math.log(10)))
    assert dnf.make_problem(POOL, succ, fail).weights() == (1, None)
    assert ":- neg(I,_), cov(I)." in dnf.asp_program(dnf.make_problem(POOL, succ, fail))
    with pytest.raises(ValueError):
        dnf.make_problem(POOL, succ, fail, epsilon=1.0)


# ---------------------------------------------------------- classical learners

def test_classical_learners_recover_a_clean_conjunction(monkeypatch):
    succ, fail = planted({3, 7}, n=30)
    for name in ("ripper", "cart", "figs"):
        out = classical.classical_rule(name, POOL, succ, fail, 7)
        assert out["status"] == "fitted" and out["rule"] == ((3, 7),), name
        assert classical.classical_rule(name, POOL, succ, fail, 7)["rule"] == out["rule"]  # seeded
    monkeypatch.setitem(classical.GPLEARN, "population_size", 60)
    monkeypatch.setitem(classical.GPLEARN, "generations", 3)
    out = classical.classical_rule("gplearn", POOL, succ, fail, 1)
    assert out["status"] in ("fitted", "never", "always", "error") and "seconds" in out


def test_figs_rule_does_not_depend_on_numpy_global_generator():
    # candidates 1 and 3 are present together in every attempt, so their splits tie; imodels breaks
    # the tie with numpy's global generator, which the learner seeds per job and then restores
    succ = [frozenset({1, 3, 5}), frozenset({1, 3, 7}), frozenset({1, 3, 9, 11}), frozenset({1, 3})]
    fail = [frozenset({5, 7}), frozenset({9}), frozenset({5, 11}), frozenset({7, 9, 11})]
    rules = set()
    for g in range(12):
        np.random.seed(g)
        before = np.random.get_state()[1].copy()
        rules.add(classical.classical_rule("figs", POOL, succ, fail, 4)["rule"])
        assert (np.random.get_state()[1] == before).all()  # the global state is restored
    assert len(rules) == 1 and rules.pop() in (((1,),), ((3,),))


def test_classical_decoding_and_one_class_rules():
    assert classical.classical_rule("cart", POOL, [{3}], [], 0)["rule"] == ((),)
    assert classical.classical_rule("cart", POOL, [], [{3}], 0)["rule"] is None
    assert classical.classical_rule("cart", POOL, [], [], 0)["status"] == "no_examples"
    states = classical.subsets((1, 2, 3))
    assert len(states) == 8 and states[0] == ()
    pred = [set(s) >= {1, 2} or set(s) >= {3} for s in states]
    assert classical.minimal_rule(states, pred) == ((1, 2), (3,))
    assert classical.minimal_rule(states, [False] * 8) is None
    assert dnf.rule_scores([((),)], POOL) == {b: 0.0 for b in POOL}


# ---------------------------------------------------------------- learners

def rl_records(tmp_path, policies=("focused_sample", "maximal"), episodes=4, name="rl-dev"):
    cfg = tiny_cfg(tmp_path, policies, episodes, name)
    res = col.collect(cfg, SessionLedger(None, {"dev": 10_000}), stamp="t")
    recs, sel = col.load_evidence(tmp_path / "runs", f"{name}-")
    return cfg, res, recs, sel


def test_posterior_variants_and_attempts(tmp_path):
    _, _, recs, _ = rl_records(tmp_path, policies=("focused_sample",), episodes=6)
    rec = next(r for r in recs if r["set"] == "slip")
    infos, logs = rec["infos"], rec["logs"]
    ex = traces.craft_examples(logs, infos)
    sigs = sorted({e.signature for e in ex})
    ref, _ = x2learners.factorized_marginals(infos, ex, sigs)
    mine, _ = learners.posterior_scores(infos, ex, sigs, 0.0, 3)
    assert ref.keys() == mine.keys() and all(abs(ref[e] - mine[e]) < 1e-12 for e in ref)
    full, _ = learners.posterior_scores(infos, ex, sigs, 0.1, None)
    assert set(full) == set(ref)
    for s in sigs:
        succ, fail = learners.attempts(infos, ex, s)
        items = set(infos[s].items)
        n = sum(1 for e in ex if e.signature == s and items <= e.present)
        assert len(succ) + len(fail) == n and all(x <= set(infos[s].pool) for x in succ + fail)
    out = learners.world_scores(rec, 6)
    assert set(out["scores"]) == set(learners.LEARNERS) - {"gplearn"} and out["epsilon"] == 0.1
    assert out["diagnostics"]["agreement"]["hard_jobs"] == out["diagnostics"]["agreement"]["hard_agree"]
    assert all({"disjunctive", "disjunctive_soft"} <= set(j) for j in out["diagnostics"]["dnf_jobs"])


def test_posterior_full_equals_a_brute_force_posterior():
    from hypergraph_agent.agents.walker import RecipeInfo
    from itertools import combinations
    infos = {"r": RecipeInfo("r", 40, (30,), POOL, None)}
    subsets = [frozenset(c) for k in range(1, len(POOL) + 1) for c in combinations(POOL, k)]
    truth = {3, 7, 9, 11}
    for eps, slip in ((0.0, 0.0), (0.1, 0.1)):
        rng = np.random.default_rng(3)
        ex = []
        for t in range(30):
            present = frozenset(c for c in POOL if rng.random() < 0.6)
            present |= {30} if rng.random() < 0.9 else set()
            ok = truth <= present and 30 in present and not rng.random() < slip
            ex.append(traces.CraftExample(1, t, "r", present, ok))
        got, _ = learners.posterior_scores(infos, ex, ["r"], eps, None)
        weights = []
        for h in subsets:  # uniform prior over every nonempty subset times the slip likelihood
            lik = 1.0
            for e in ex:
                if 30 not in e.present:
                    lik *= 0.0 if e.success else 1.0  # an absent item input explains a failure
                    continue
                p1 = (1 - eps) * (h <= e.present)
                lik *= p1 if e.success else 1 - p1
            weights.append(lik)
        z = sum(weights)
        for b in POOL:
            want = sum(w for w, h in zip(weights, subsets) if b in h) / z
            assert got[("r", b)] == pytest.approx(want, abs=1e-12)
    prior, _ = learners.posterior_scores(infos, [], ["r"], 0.0, None)
    assert all(v == pytest.approx(32 / 63) for v in prior.values())  # above one half


# -------------------------------------------------------------- collection

def test_default_sets_reproduce_x2_streams_and_agents():
    g = {"name": "g", "kind": "generated", "observe_items": "all", "namespace": "rl_test_dev",
         "base_seed": 10_000, "seeds": [0], "n_worlds": 2, "episodes_per_world": 3}
    a, oa = x2col.stream_for(g, 1)
    s = col.validate({"run": {"name": "rl-t", "allocation": "dev", "ledger": "x"},
                      "ledger": {"allocations": {"dev": 1}, "adaptive_cap": 1, "reporting_cap": 1},
                      "sets": [dict(g)]})["sets"][0]
    b, ob = col.stream_for(s, 1)
    assert oa == ob and all(a.task(i).task_key == b.task(i).task_key and a.task(i).budget == b.task(i).budget
                            for i in oa)
    for policy in col.POLICIES:
        played = []
        for agent in (x2col.evidence_agent(policy, x2col.WALKER_DEFAULTS, ("g", 1, policy, "k")),
                      col.evidence_agent(policy, col.WALKER_DEFAULTS, ("g", 1, policy, "k"), 0.0)):
            env, m = RecipeQuestEnv(), meter()
            for i in ob:
                ex = Executor(env, m, "exploration", torch.Generator().manual_seed(i))
                agent.run_episode(ex, b.task(i), i)
            played.append([lg.steps for w in agent.worlds.values() for lg in w.evidence.logs])
        assert played[0] and played[0] == played[1], policy  # the same actions, episode by episode


def test_variant_sets_change_requirement_sizes_and_slips(tmp_path):
    cfg = col.load(ROOT / "configs" / "rule_learners" / "rl_dev.yaml")
    ooc = next(s for s in cfg["sets"] if s["name"] == "ooc")
    stream, _ = col.stream_for(ooc, 0)
    sizes = {len(r.true_base) for w in range(ooc["n_worlds"]) for r in stream.world(w).recipes}
    assert sizes <= {4, 5} and all(len(r.pool) == 8 for r in stream.world(0).recipes)
    slip = next(s for s in cfg["sets"] if s["name"] == "slip")
    stream, order = col.stream_for(slip, 0)
    assert {stream.task(i).failure_prob for i in order} == {0.1}
    _, _, recs, _ = rl_records(tmp_path, policies=("maximal",), episodes=16)
    v3 = an.variant_check(recs)
    assert v3["holds"] and set(v3["sizes"]["ooc"]) <= {4, 5} and v3["slips"].get("slip", 0) > 0


def test_config_validation():
    raw = yaml.safe_load((ROOT / "configs" / "rule_learners" / "rl_dev.yaml").read_text())
    for key, value, err in ((("sets", 0, "namespace"), "rl_slip", ValueError),
                            (("sets", 0, "namespace"), "x2_gen_dev", ValueError),
                            (("sets", 0, "observe_items"), "goal_only", ValueError),
                            (("sets", 0, "arity_weights"), [0.5, 0.5], ValueError),
                            (("sets", 0, "failure_prob"), 1.0, ValueError),
                            (("sets", 0, "n_world"), 2, KeyError),
                            (("walker", "omit"), 0.5, KeyError)):
        bad = copy.deepcopy(raw)
        target = bad
        for part in key[:-1]:
            target = target[part]
        target[key[-1]] = value
        with pytest.raises(err):
            col.validate(bad)


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_share_one_ledger_declaration(path):
    cfg = col.load(path)
    first = col.load(CONFIGS[0])
    assert cfg["ledger"] == first["ledger"] and cfg["run"]["ledger"] == "runs/RL-ledger.json"
    dev = cfg["run"]["allocation"] == "dev"
    assert all(s["namespace"].startswith("rl_") and s["namespace"].endswith("_dev") == dev
               for s in cfg["sets"])
    assert cfg["run"]["requires_flag"] == (not dev) and {s["name"] for s in cfg["sets"]} == {"slip", "ooc"}


def test_measurement_plan_is_covered_by_its_declaration():
    cfg = col.load(ROOT / "configs" / "rule_learners" / "rl_main.yaml")
    p = col.plan(cfg)
    led = cfg["ledger"]
    assert p["n_runs"] == 30 and all(r["worlds"] == 12 and r["tasks"] == 192 for r in p["runs"])
    assert led["allocations"]["rl"] >= p["worst_case_adaptive"] + p["largest_run"]["adaptive"]
    assert led["adaptive_cap"] == led["allocations"]["rl"] + led["allocations"]["dev"]
    dev = col.plan(col.load(ROOT / "configs" / "rule_learners" / "rl_dev.yaml"))
    assert led["reporting_cap"] >= p["worst_case_reporting"] + p["largest_run"]["reporting"] + 5000
    assert dev["worst_case_adaptive"] <= led["allocations"]["dev"] and dev["worst_case_reporting"] <= 5000


def test_collect_refuses_a_ledger_declared_otherwise(tmp_path, capsys):
    raw = yaml.safe_load((ROOT / "configs" / "rule_learners" / "rl_dev.yaml").read_text())
    raw["run"].update(runs_dir=str(tmp_path / "runs"), ledger=str(tmp_path / "ledger.json"))
    for s in raw["sets"]:
        s.update(n_worlds=1, episodes_per_world=1)
    path = tmp_path / "rl.yaml"
    path.write_text(yaml.safe_dump(raw))
    SessionLedger(tmp_path / "ledger.json", {"dev": 10_000, "rl": 1}, 10_001, 5).save()  # another declaration
    assert main(["collect", "--config", str(path), "--dry-run"]) == 0
    assert col.plan(col.load(path))["ledger_file_matches"] is False
    assert main(["collect", "--config", str(path)]) == 2 and "refused" in capsys.readouterr().out
    assert not (tmp_path / "runs").exists() or not any((tmp_path / "runs").glob("rl-dev2-*"))


def test_collect_resume_seeds_and_allowance(tmp_path):
    cfg = tiny_cfg(tmp_path, policies=("maximal",), episodes=2)
    ledger = SessionLedger(None, {"dev": 10_000})
    first = col.collect(cfg, ledger, stamp="a", only_sets=["slip"])
    again = col.collect(cfg, ledger, stamp="b", resume=True, only_sets=["slip"])
    assert len(first) == 1 and again[0]["skipped"] and again[0]["run_id"] == first[0]["run_id"]
    assert col.collect(cfg, ledger, stamp="c", only_seeds=[7]) == []
    with pytest.raises(col.AllowanceShort):
        col.collect(cfg, SessionLedger(None, {"dev": 5}, reporting_cap=5), stamp="d")
    man = json.loads((tmp_path / "runs" / first[0]["run_id"] / "manifest.json").read_text())
    conf = json.loads((tmp_path / "runs" / first[0]["run_id"] / "config.json").read_text())
    assert man["config_file"] == "<dict>" and conf["study"] == "rl" and conf["set"]["failure_prob"] == 0.1


# ---------------------------------------------------------------- analysis

def test_decision_rule_four_outcomes():
    rng = np.random.default_rng(0)
    up = list(0.1 + 0.01 * rng.standard_normal(30))
    assert an.decision(up, 0.05, names=("dnf", "posterior"))["decision"] == "dnf better"
    assert an.decision(up, 0.05, lower_is_better=True, names=("dnf", "posterior"))["decision"] == \
        "posterior better"
    assert an.decision(list(0.001 * rng.standard_normal(30)), 0.05)["decision"] == "equivalent"
    assert an.decision(list(0.3 * rng.standard_normal(30)), 0.05)["decision"] == "inconclusive"
    assert an.decision([], 0.05)["decision"] is None


def test_world_metrics_use_x2_scoring_and_gplearn_where_enabled(tmp_path, monkeypatch):
    _, _, recs, _ = rl_records(tmp_path, policies=("focused_sample",), episodes=4)
    rec = next(r for r in recs if r["set"] == "ooc")
    monkeypatch.setitem(classical.GPLEARN, "population_size", 40)
    monkeypatch.setitem(classical.GPLEARN, "generations", 2)
    row = an.world_metrics(rec, 4, gplearn=True)
    assert "gplearn" in row["metrics"] and set(row["metrics"]) == set(learners.LEARNERS)
    for m, v in row["metrics"].items():
        assert 0.0 <= v["f1"] <= 1.0 and v["act_ratio"] >= 1.0, m
        assert math.isnan(v["auroc"]) or 0.0 <= v["auroc"] <= 1.0, m
    assert row["metrics"]["dnf"] == row["metrics"]["dnf_rc2"] == row["metrics"]["dnf_soft"]
    assert row["metrics"]["posterior"] == row["metrics"]["posterior_eps"]  # no slips in this set
    assert an.world_metrics(rec, 99) is None
    primary_cell = dict(rec, strategy="random_omit")  # gplearn cells are (set, source, checkpoint)
    assert an.gplearn_here(primary_cell, 16) and not an.gplearn_here(primary_cell, 4)
    assert not an.gplearn_here(rec, 16) and not an.gplearn_here(dict(primary_cell, set="slip"), 16)


def test_benchmark_sets_get_one_checkpoint_and_no_maximal_cell():
    for src in an.SOURCES:
        rec = {"set": "bench_crafter", "strategy": src}
        assert an.checkpoints_for(rec, an.CHECKPOINTS) == ((an.BENCH_K,) if src != "maximal" else ())
        assert an.checkpoints_for(dict(rec, set="gen_obs"), an.CHECKPOINTS) == an.CHECKPOINTS
    assert set(an.BENCH_SETS) == set(col.X2_BENCH_SETS) and not set(an.BENCH_SETS) & set(an.SETS)


def test_v4_compares_posterior_and_random_with_x2s_summary():
    metrics = {m: {"auroc": 0.75, "f1": 0.5, "act_ratio": 1.2, "plan_success": float("nan")}
               for m in ("posterior", "random", "dnf")}
    per = {("gen_obs", "focused_sample", 16): {"gen_obs:s2:w0": {"seed": 2, "metrics": metrics}},
           ("ooc", "focused_sample", 16): {"ooc:s0:w0": {"seed": 0, "metrics": metrics}}}
    world = {"gen_obs:s2:w0": {"metrics": copy.deepcopy(metrics)}}
    x2 = {"runs": {"chosen": ["a", "b"]}, "per_world": {"gen_obs:test:focused_sample@16": world}}
    v4 = an.x2_agreement(per, x2, ["a"])
    assert v4["holds"] is True and v4["values"] == 8 and v4["worlds_missing"] == 0
    world["gen_obs:s2:w0"]["metrics"]["random"]["f1"] += 1e-6
    assert an.x2_agreement(per, x2, ["a"])["holds"] is False
    assert an.x2_agreement(per, x2, ["c"])["holds"] is None  # X2's summary covers other runs
    assert an.x2_agreement(per, None, ["a"])["holds"] is None
    assert an.x2_agreement({("gen_obs", "maximal", 4): {"gen_obs:s0:w9": {"seed": 0, "metrics": metrics}}},
                           x2, ["a"])["holds"] is False  # a world X2's summary lacks


def test_parallel_worlds_give_the_sequential_metrics(tmp_path):
    _, _, recs, _ = rl_records(tmp_path, policies=("focused_sample",), episodes=3)
    tasks = [(rec, (3,), ()) for rec in recs]
    seq = [an._world_rows(t) for t in tasks]
    finished = []
    par = an._rows_parallel(tasks, 2, lambda i, rows: finished.append(i))
    assert all(p is not None for p in par) and sorted(finished) == list(range(len(tasks)))
    for a, b in zip(seq, par):
        assert json.dumps([(k, r["metrics"]) for k, r in a], sort_keys=True) == \
            json.dumps([(k, r["metrics"]) for k, r in b], sort_keys=True)


def test_end_to_end_reads_x2_logs_and_reports(tmp_path, monkeypatch):
    x2_tiny(tmp_path, bench=True)
    cfg, res, _, _ = rl_records(tmp_path, policies=col.POLICIES, episodes=4)
    assert all(r["complete"] for r in res) and len(res) == 6
    x2recs, _ = col.load_x2_evidence(tmp_path / "runs", "x2-dev-")
    assert x2recs and {r["set"] for r in x2recs} == {"gen_obs", "bench_crafter"}
    assert all(r["failure_prob"] == 0 for r in x2recs)
    monkeypatch.setitem(an.PRIMARY, "k", 4)
    monkeypatch.setattr(an, "BENCH_K", 4)
    out = tmp_path / "out"
    base = ["rank", "--runs", str(tmp_path / "runs"), "--prefix", "rl-dev-", "--x2-prefix", "x2-dev-",
            "--out", str(out), "--expected-worlds", "1", "--checkpoints", "2", "4"]
    args = base + ["--cache", str(tmp_path / "cache")]
    assert main(args) == 0
    s = json.loads((out / "rl_benchmark.json").read_text())
    assert {k.split(":")[0] for k in s["table"]} == {"gen_obs", "slip", "ooc", "bench_crafter"}
    assert {k for k in s["table"] if k.startswith("bench")} == {"bench_crafter:focused_sample@4",
                                                                "bench_crafter:random_omit@4"}
    assert s["validity"]["V1"]["holds"] and s["validity"]["V2"]["holds"]
    v3 = s["validity"]["V3"]  # four episodes of one world may contain no slip at all
    assert set(map(int, v3["sizes"]["ooc"])) <= {4, 5} and set(map(int, v3["sizes"]["slip"])) <= {1, 2, 3}
    assert v3["holds"] == (v3["slips"].get("slip", 0) > 0) and "bench_crafter" not in v3["sizes"]
    assert s["validity"]["V4"]["holds"] is None  # X2's committed summary covers other runs
    assert s["complete"] and s["verdict"] == (s["primary_decision"]["decision"] if v3["holds"] else "invalid")
    assert set(s["secondary"]) == {t[0] for t in an.SECONDARY} and len(s["secondary"]) == 8
    assert all(t["holds"] in (True, False) for t in s["secondary"].values())
    assert set(s["descriptive"]) == {t[0] for t in an.DESCRIPTIVE}
    assert not any("holds" in t for t in s["descriptive"].values())
    text = (out / "rl_benchmark.md").read_text() + (out / "rl_benchmark.json").read_text()
    assert "Secondary tests" in text and "Descriptive contrasts" in text
    assert "<=[" not in text  # no recipe signature (no name of a converted world) in the report
    for name in ("rl", "x2"):
        assert len(s["runs"][name]["provenance"]) == 1
    # the cache folder may never be X2's
    assert main(base + ["--cache", str(tmp_path / "runs" / "x2_cache")]) == 2


def test_rank_cache_resume_and_code_stamp(tmp_path):
    _, _, recs, _ = rl_records(tmp_path, policies=("maximal",), episodes=2)
    a = an.rank_benchmark(recs, checkpoints=(2,), cache_dir=tmp_path / "c", expected_worlds=1)
    b = an.rank_benchmark(recs, checkpoints=(2,), cache_dir=tmp_path / "c", expected_worlds=1, resume=True)
    assert a["rows_reused"] == 0 and b["rows_reused"] == len(recs)
    assert json.dumps(a["table"], sort_keys=True) == json.dumps(b["table"], sort_keys=True)
    assert a["provenance"]["cache_key"] == b["provenance"]["cache_key"]
    assert a["validity"]["V2"]["holds"]  # brute-force evidence: every exact learner at chance
    f = next((tmp_path / "c" / a["provenance"]["cache_key"] / "rows").glob("*.json"))
    d = json.loads(f.read_text())
    d["value"]["row"]["code"] = "0" * 64
    f.write_text(json.dumps(d))
    with pytest.raises(an.CodeMismatch):
        an.rank_benchmark(recs, checkpoints=(2,), cache_dir=tmp_path / "c", resume=True)


def _copy_run(src: pathlib.Path, name: str, evidence: bool, **manifest) -> None:
    d = src.parent / name
    d.mkdir()
    man = json.loads((src / "manifest.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    man.update(run_id=name, **manifest)
    if evidence:  # another seed: a run of its own, with evidence
        cfg["seed"] = man["seed"] = 9
        (d / "evidence").mkdir()
        for f in (src / "evidence").glob("*.json"):
            (d / "evidence" / f.name).write_bytes(f.read_bytes())
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "config.json").write_text(json.dumps(cfg))


def test_rl_runs_are_selected_like_x2s_and_mixed_code_is_refused(tmp_path):
    x2_tiny(tmp_path, episodes=2)
    _, res, _, _ = rl_records(tmp_path, policies=("maximal",), episodes=2)
    src = tmp_path / "runs" / res[0]["run_id"]
    _copy_run(src, "rl-dev-crashed", False, status="running", stop_reason=None,
              start_time="2000-01-01T00:00:00")
    recs, sel = col.load_evidence(tmp_path / "runs", "rl-dev-")
    assert sel["reruns_used"] == [{"incomplete": "rl-dev-crashed", "rerun": res[0]["run_id"]}]
    assert len(sel["provenance"]) == 1 and all(r["complete"] for r in recs)
    man = json.loads((src / "manifest.json").read_text())
    _copy_run(src, "rl-dev-othercode", True, source={**man["source"], "loaded_code_sha256": "0" * 64})
    assert len(col.load_evidence(tmp_path / "runs", "rl-dev-")[1]["provenance"]) == 2
    assert main(["rank", "--runs", str(tmp_path / "runs"), "--prefix", "rl-dev-", "--x2-prefix", "x2-dev-",
                 "--out", str(tmp_path / "o"), "--cache", str(tmp_path / "c")]) == 2


# ------------------------------------------------------------------ hygiene

def test_study_rl_files_have_short_lines_and_no_dashes():
    files = sorted((ROOT / "src" / "hypergraph_agent" / "rules").glob("*.py")) + CONFIGS
    files += [ROOT / "tests" / "test_rule_learners.py", ROOT / "docs" / "studies" / "RL_rule_learners.md"]
    banned = (chr(0x2014), chr(0x2013))
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert len(line) <= 110, f"{f.name}:{i}"
            assert not any(d in line for d in banned), f"{f.name}:{i}"
            assert not re.search(r"\s--\s", line), f"{f.name}:{i}"
