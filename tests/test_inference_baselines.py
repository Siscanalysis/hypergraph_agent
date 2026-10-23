"""Study X2: adapters, learners, metrics, collection and the offline benchmark."""

import copy
import hashlib
import json
import pathlib
import re

import numpy as np
import pytest
import torch
import yaml

from hypergraph_agent.agents.walker import Walker
from hypergraph_agent.envs.public_schema import CRAFT
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.reference_solver import reference_solve
from hypergraph_agent.evaluation.markov_study import (
    info_from_record, log_from_record, posterior_marginals, world_records,
)
from hypergraph_agent.evaluation.walkers import play
from hypergraph_agent.inference_baselines import benchmark as bm
from hypergraph_agent.inference_baselines import collection as col
from hypergraph_agent.inference_baselines import (
    arms_worker, learners, macq_adapter, msgi_adapter, provenance, traces,
)
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.training.budget import SessionLedger

from helpers import meter

ROOT = pathlib.Path(__file__).resolve().parents[1]
CONFIGS = sorted((ROOT / "configs" / "inference_baselines").glob("*.yaml"))


def gen_set(observe="all", n_worlds=1, episodes=8, namespace="x2_test_dev"):
    return {"name": "gen", "kind": "generated", "observe_items": observe, "namespace": namespace,
            "base_seed": 10_000, "seeds": [0], "n_worlds": n_worlds, "episodes_per_world": episodes}


def records(policy="focused_sample", observe="all", n_worlds=1, episodes=8, seed=0, s=None):
    s = s or gen_set(observe, n_worlds, episodes)
    stream, order = col.stream_for(s, seed)
    agent = col.evidence_agent(policy, col.WALKER_DEFAULTS, ("gen", seed, policy, "x2"))
    rows = play(agent, policy, stream, order, meter(100_000), seed, None, "exploration", {})
    recs = world_records(agent, stream, order, rows,
                         {"run_id": f"r-{policy}", "seed": seed, "strategy": policy, "set": s["name"],
                          "set_kind": s["kind"], "observe_items": s["observe_items"]})
    for rec in recs:
        rec["infos"] = {i["signature"]: info_from_record(i) for i in rec["infos"]}
        rec["logs"] = [log_from_record(x) for x in rec["logs"]]
        rec["complete"] = True
        rec["world_label"] = f"{s['name']}:s{seed}:{rec['world_key']}"
    return recs, rows


def benchmark_data_present(name: str) -> bool:
    from hypergraph_agent.benchmarks import loaders
    try:
        loaders.load(name)
    except Exception:
        return False
    return True


# ---------------------------------------------------------------- policies

def test_logged_random_omit_acts_like_random_omit_and_keeps_logs():
    stream, order = col.stream_for(gen_set(episodes=4), 0)
    tasks = [stream.task(i) for i in order]

    def run(agent):
        m, env, out = meter(), RecipeQuestEnv(), []
        for i, t in enumerate(tasks):
            ex = Executor(env, m, "exploration", torch.Generator().manual_seed(i))
            out.append(agent.run_episode(ex, t, i))
        return out

    plain = run(Walker("random_omit", np.random.default_rng(3), omit_prob=0.663))
    logged = col.LoggedRandomOmit(np.random.default_rng(3), 0.663)
    rows = run(logged)
    assert [r["primitive_length"] for r in rows] == [r["primitive_length"] for r in plain]
    assert len(next(iter(logged.worlds.values())).evidence.logs) == 4


# ------------------------------------------------------------------ traces

def test_traces_need_observable_items_and_keep_the_state_sequence():
    recs, _ = records(observe="goal_only", episodes=4)
    rec = recs[0]
    assert not bm.observable(rec, rec["logs"], rec["infos"])
    with pytest.raises(traces.ItemsUnobserved):
        for lg in rec["logs"]:
            traces.observed_states(lg, rec["infos"])
    recs, _ = records(observe="all", episodes=4)
    rec = recs[0]
    for lg in rec["logs"]:
        states = traces.observed_states(lg, rec["infos"])
        assert all(a.after == b.before for a, b in zip(states, states[1:]))
        steps = traces.successful_steps(lg, rec["infos"])
        assert all(s.changed for s in steps) and all(a.after == b.before for a, b in zip(steps, steps[1:]))
    ex = traces.craft_examples(rec["logs"], rec["infos"])
    truth = rec["truth"]["requirements"]
    assert ex and all(e.success == (set(truth[e.signature]) <= e.present
                                    and set(rec["infos"][e.signature].items) <= e.present) for e in ex)


# ---------------------------------------------------------------- learners

def test_factorized_posterior_equals_the_joint_posterior_with_items_observable():
    recs, _ = records(observe="all", episodes=6)
    rec = recs[0]
    sigs = bm.scored_sigs(rec["infos"], rec["logs"])
    joint, diag = posterior_marginals(rec["infos"], rec["logs"], np.random.default_rng(0))
    assert diag["sampled_recipes"] == 0 and diag["prior_recipes"] == 0
    ex = traces.craft_examples(rec["logs"], rec["infos"])
    fact, fdiag = learners.factorized_marginals(rec["infos"], ex, sigs)
    assert fdiag["contradiction"] == 0
    assert all(abs(fact[e] - joint.get(e, 0.0)) < 1e-9 for e in fact)


def test_sam_style_learner_equals_macq_observer_and_is_safe():
    recs, _ = records(observe="all", episodes=8)
    rec = recs[0]
    steps = [traces.successful_steps(lg, rec["infos"]) for lg in rec["logs"]]
    sam = learners.sam_preconditions(steps)
    obs = macq_adapter.observer_preconditions(rec["logs"], rec["infos"])
    assert sam and sam == obs
    for sig, pre in sam.items():  # safe: every true requirement is inside the learned precondition
        assert set(rec["truth"]["requirements"][sig]) <= pre


def test_msgi_adapter_recovers_a_conjunction_and_applies_msgi_rules():
    feats = ["a", "b", "c", "d", "e", "f"]
    rng = np.random.default_rng(1)
    X = [[bool(v) for v in rng.random(6) < 0.5] for _ in range(40)]
    y = [r[0] and r[1] for r in X]
    pre = msgi_adapter.infer_precondition(X, y, feats)
    assert pre.status == "tree" and pre.terms == [{"a": 1, "b": 1}]
    assert msgi_adapter.infer_precondition(X[:3], [True] * 3, feats).status == "always"
    assert msgi_adapter.infer_precondition(X[:3], [False] * 3, feats).status == "never"
    assert msgi_adapter.infer_precondition([], [], feats).status == "never"


def test_vendored_msgi_files_match_their_recorded_checksums():
    src = (ROOT / "third_party" / "msgi" / "SOURCE.md").read_text()
    sizes = dict(re.findall(r"^\| `([^`]+)` \| `[^`]+` \| (\d+) \|$", src, re.M))
    shas = dict(re.findall(r"^- `([^`]+)`: `([0-9a-f]{64})`$", src, re.M))
    assert len(sizes) == 3 and set(sizes) == set(shas)
    for rel, size, sha in ((r, sizes[r], shas[r]) for r in sizes):
        data = (ROOT / "third_party" / "msgi" / rel).read_bytes()
        assert len(data) == int(size) and hashlib.sha256(data).hexdigest() == sha


def test_arms_runs_in_a_worker_with_a_fixed_hash_seed():
    recs, _ = records(observe="all", episodes=4)
    rec = recs[0]
    job = macq_adapter.arms_job("a", rec["logs"], rec["infos"])
    seen = []
    out = macq_adapter.run_arms([job, {**job, "id": "b"}, {**job, "id": "c", "traces": []}], workers=2,
                                on_result=lambda r: seen.append(r["id"]))
    assert out["a"]["status"] in ("ok", "iteration_cap") and out["a"]["precond"] == out["b"]["precond"]
    assert out["a"]["code"] == provenance.code_sha256() and sorted(seen) == ["a", "b", "c"]
    assert out["a"]["weights_moved"] > 0  # the released pairing would have moved some weights
    assert out["c"]["status"] == "no_traces"
    pre = macq_adapter.arms_preconditions(out["a"], rec["infos"])
    assert set(pre) <= set(rec["infos"])
    assert macq_adapter.arms_preconditions({"status": "timeout"}, rec["infos"]) == {}


def test_arms_weights_follow_their_clauses(monkeypatch):
    import macq.extract.arms as arms_mod
    from nnf import Or, Var
    from nnf import false as nnffalse
    recs, _ = records(observe="all", episodes=4)
    rec = recs[0]
    job = macq_adapter.arms_job("w", rec["logs"], rec["infos"])
    arms_worker.install()
    calls = {"step3": [], "step4": []}
    step3, step4 = arms_mod.ARMS.step3, arms_mod.ARMS.step4

    def spy3(*args, **kw):
        calls["step3"].append(args)
        return step3(*args, **kw)

    def spy4(max_sat, decode):
        calls["step4"].append(([list(c) for c in max_sat.soft], list(max_sat.wght), dict(decode)))
        return step4(max_sat, decode)

    spy3._x2_pairing = True  # keeps install() from wrapping the spy again
    monkeypatch.setattr(arms_mod.ARMS, "step3", staticmethod(spy3))
    monkeypatch.setattr(arms_mod.ARMS, "step4", staticmethod(spy4))
    res = arms_worker.run_job(job)
    assert res["status"] == "ok" and calls["step3"] and len(calls["step3"]) == len(calls["step4"])
    rates = arms_mod.ARMS._calculate_support_rates
    for (c, aw, iw, th, i3d, pld, *_), (soft, wght, decode) in zip(calls["step3"], calls["step4"]):
        weights = [aw] * len(c.action) + [iw] * len(c.info) + rates(list(c.info3.values()), th, i3d) \
            + rates(list(c.plan.values()), th, pld)
        want: dict = {}
        for con, w in zip(c.action + c.info + list(c.info3) + list(c.plan), weights):
            if con != nnffalse:
                want[con] = min(w, want.get(con, w))
        assert len(soft) == len(want)
        for clause, w in zip(soft, wght):  # every soft clause carries the weight step3 gave it
            assert want[Or({Var(decode[abs(x)], x > 0) for x in clause})] == w


def test_arms_step_limit_keeps_the_model_learned_so_far(monkeypatch):
    recs, _ = records(observe="all", episodes=4)
    rec = recs[0]
    job = macq_adapter.arms_job("cap", rec["logs"], rec["infos"])
    job["settings"] = {**job["settings"], "upper_bound": 99}  # no action is ever fully learned
    crafts = {a for t in job["traces"] for a in t["actions"] if a.startswith("craft_")}
    models = []
    for cap in (2, 3):
        monkeypatch.setattr(arms_worker, "ARMS_ITERATIONS", cap)
        res = arms_worker.run_job(job)
        assert res["status"] == "iteration_cap" and res["iterations"] == cap
        assert crafts and crafts <= set(res["precond"])  # a precondition for every craft
        models.append(res["precond"])
    assert models[0] == models[1]


def test_provenance_hash_is_the_run_manifest_code_hash():
    from hypergraph_agent.training import run
    assert provenance.code_sha256(fresh=True) == run._code_hash()
    assert provenance.msgi_sha256() and set(provenance.versions()) >= {"macq", "nnf", "python-sat", "torch"}


# ----------------------------------------------------------------- metrics

def test_set_scores_inferred_sets_and_set_metrics():
    infos = {"r": type("I", (), {"pool": (1, 2, 3, 4)})()}
    sc, mod = learners.set_scores(infos, ["r"], {"r": frozenset({1, 9})})
    assert sc == {("r", 1): 1.0, ("r", 2): 0.0, ("r", 3): 0.0, ("r", 4): 0.0} and mod == {"r"}
    sc, mod = learners.set_scores(infos, ["r"], {})
    assert set(sc.values()) == {0.0} and mod == set()
    assert bm.inferred_set({1: 0.2, 2: 0.2, 3: 0.1, 4: 0.0}, (1, 2, 3, 4)) == {1, 2}
    assert bm.inferred_set({1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0}, (1, 2, 3, 4)) == {1, 2, 3, 4}
    assert bm.inferred_set({1: 0.5, 2: 0.9, 3: 0.4999, 4: 0.0}, (1, 2, 3, 4)) == {1, 2}
    m = bm.set_metrics(frozenset({1, 2}), {2, 3})
    assert m == {"precision": 0.5, "recall": 0.5, "f1": 0.5}
    assert bm.set_metrics(frozenset(), {2})["precision"] == 0.0


def test_acting_cost_is_optimal_on_the_truth_and_brute_force_on_full_pools():
    recs, rows = records(policy="maximal", observe="all", episodes=6)
    rec = recs[0]
    truth = {s: frozenset(v) for s, v in rec["truth"]["requirements"].items()}
    exact = bm.acting_metrics(rec, truth)
    assert exact == {"act_ratio": 1.0, "plan_success": 1.0}
    full = bm.acting_metrics(rec, {})
    assert full["plan_success"] == 1.0 and full["act_ratio"] >= 1.0
    # the brute-force plan is what maximal executed: its rows have the same lengths
    opt = sum(r["reference_length"] for r in rows)
    assert full["act_ratio"] == pytest.approx(sum(r["primitive_length"] for r in rows) / opt)
    # omitting one required candidate of every recipe: the node's plans fail where they use a recipe
    wrong = {s: frozenset(sorted(v)[1:]) or frozenset(sorted(set(rec["infos"][s].pool) - v)[:1])
             for s, v in truth.items()}
    bad = bm.acting_metrics(rec, wrong)
    assert bad["plan_success"] < 1.0 and bad["act_ratio"] > 1.0


def test_optimal_length_matches_the_reference_solver():
    stream, order = col.stream_for(gen_set(episodes=6, n_worlds=2), 0)
    for i in order:
        t = stream.task(i)
        sigs = [r.signature(t.profile) for r in t.rules]
        infos, truth = {}, {}
        for r in t.rules:
            s = r.signature(t.profile)
            infos[s] = type("I", (), {"effect": r.effect, "items": r.item_inputs, "pool": r.pool,
                                      "known": None})()
            truth[s] = r.true_base
        assert bm.optimal_length(sigs, infos, truth, frozenset(t.initial_true), t.goal) == \
            reference_solve(t).length


def test_decision_rule_and_baseline_selection():
    rng = np.random.default_rng(0)
    assert bm.decision(list(0.1 + 0.01 * rng.standard_normal(36)))["decision"] == "posterior better"
    assert bm.decision(list(-0.1 + 0.01 * rng.standard_normal(36)))["decision"] == "baseline better"
    eq = bm.decision(list(0.001 * rng.standard_normal(36)))
    assert eq["decision"] == "equivalent" and eq["margin"] == 0.03
    assert bm.decision(list(0.2 * rng.standard_normal(36)))["decision"] == "inconclusive"
    rows = [{"metrics": {"msgi": {"auroc": 0.7}, "observer": {"auroc": 0.7}, "arms": {"auroc": 0.6}}}]
    assert bm.select_best(rows)["selected"] == "msgi"
    rows[0]["metrics"]["observer"]["auroc"] = 0.71
    assert bm.select_best(rows)["selected"] == "observer"
    assert bm.select_best([])["selected"] is None


def test_parallel_worlds_give_the_sequential_metrics():
    recs, _ = records(observe="goal_only", episodes=2)
    recs += records(policy="random_omit", observe="all", episodes=4)[0]
    tasks = [(rec, (2,), {}, None) for rec in recs]
    seq = [bm._world_rows(t) for t in tasks]
    finished = []
    par = bm._rows_parallel(tasks, 2, lambda i, rows: finished.append(i))
    assert all(p is not None for p in par) and sorted(finished) == list(range(len(tasks)))
    assert all(r["code"] == provenance.code_sha256() for rows in par for _, r in rows)
    for a, b in zip(seq, par):
        assert json.dumps([(k, r["metrics"]) for k, r in a], sort_keys=True) == \
            json.dumps([(k, r["metrics"]) for k, r in b], sort_keys=True)


# -------------------------------------------------------- config and plans

@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_configs_validate_and_share_one_ledger_declaration(path):
    raw = yaml.safe_load(path.read_text())
    cfg = col.validate(raw, str(path))
    first = yaml.safe_load(CONFIGS[0].read_text())
    assert cfg["ledger"] == first["ledger"] and cfg["run"]["ledger"] == "runs/X2-ledger.json"
    dev = cfg["run"]["allocation"] == "dev"
    assert all(s["namespace"].endswith("_dev") == dev for s in cfg["sets"])
    assert cfg["run"]["requires_flag"] == (not dev)


def test_config_validation_rejects_unknown_keys_and_misused_namespaces():
    raw = yaml.safe_load((ROOT / "configs" / "inference_baselines" / "dev.yaml").read_text())
    bad = copy.deepcopy(raw)
    bad["sets"][0]["namespace"] = "x2_gen"
    with pytest.raises(ValueError):
        col.validate(bad)
    bad = copy.deepcopy(raw)
    bad["walker"]["omit"] = 0.5
    with pytest.raises(KeyError):
        col.validate(bad)
    bad = copy.deepcopy(raw)
    bad["sets"][0]["n_world"] = 2
    with pytest.raises(KeyError):
        col.validate(bad)


def test_measurement_plan_is_covered_by_its_allocation():
    if not all(benchmark_data_present(n) for n in ("little_alchemy_2", "freeciv_classic")):
        pytest.skip("fetched benchmark data absent (benchmarks/little_alchemy/fetch.py, "
                    "benchmarks/freeciv/fetch.py)")
    cfg = col.load(ROOT / "configs" / "inference_baselines" / "x2_main.yaml")
    p = col.plan(cfg)
    assert p["n_runs"] == 45 and p["allocation_covers_worst_case"]
    assert p["worst_case_adaptive"] <= cfg["ledger"]["adaptive_cap"] - cfg["ledger"]["allocations"]["dev"]
    assert p["worst_case_reporting"] <= cfg["ledger"]["reporting_cap"]
    gen = [r for r in p["runs"] if r["set"] == "gen_obs"]
    assert len(gen) == 15 and all(r["worlds"] == 12 and r["tasks"] == 192 for r in gen)


def test_benchmark_stream_is_deterministic_and_asks_for_world_items():
    s = {"name": "b", "kind": "benchmark", "graph": "crafter", "worlds": "whole", "n_worlds": 2,
         "namespace": "x2_test_dev", "seeds": [0], "episodes_per_world": 3, "observe_items": "all",
         "conversion_seed": 100, "goal_sample_seed": 0}
    a, order = col.stream_for(s, 0)
    b, _ = col.stream_for(s, 0)
    assert order == list(range(6))
    for i in order:
        ta, tb = a.task(i), b.task(i)
        assert ta.task_key == tb.task_key and ta.goal in {t for t, _ in ta.world.levels}
        assert ta.observe_items == "all" and ta.world.world_key.startswith("B")
    assert a.task(0).world.world_key != a.task(3).world.world_key


# --------------------------------------------------------------- pipeline

def test_collection_and_benchmark_end_to_end(tmp_path):
    raw = yaml.safe_load((ROOT / "configs" / "inference_baselines" / "dev.yaml").read_text())
    raw["run"].update(runs_dir=str(tmp_path / "runs"), ledger=str(tmp_path / "ledger.json"))
    raw["policies"] = ["focused_sample", "maximal"]
    sets = [s for s in raw["sets"] if s["kind"] == "generated"]
    fetched = [g for g in ("little_alchemy_2", "freeciv_classic") if benchmark_data_present(g)]
    if "little_alchemy_2" in fetched:
        sets += [s for s in raw["sets"] if s["name"] == "bench_little_alchemy_2"]
    if "freeciv_classic" in fetched:
        sets.append({"name": "bench_freeciv_classic", "kind": "benchmark", "graph": "freeciv_classic",
                     "worlds": "per_goal", "n_worlds": 1, "goal_sample_seed": 7, "seeds": [0],
                     "namespace": "x2_bench_freeciv_classic_dev", "observe_items": "all"})
    for s in sets:
        s.update(n_worlds=1, episodes_per_world=4)
    raw["sets"] = sets
    cfg = col.validate(raw)
    ledger = SessionLedger(None, {"dev": 10_000})
    res = col.collect(cfg, ledger, stamp="t")
    assert all(r["complete"] for r in res) and len(res) == 2 * len(sets)
    reporting = sum(r["interactions"] for r in res if r["policy"] == "maximal")
    assert ledger.summary()["reporting_total"] == reporting
    worlds, sel = col.load_evidence(tmp_path / "runs", "x2-dev-")
    assert len(worlds) == len(res) and sel["ignored"] == [] and len(sel["provenance"]) == 1
    out = bm.rank_benchmark(worlds, checkpoints=(2, 4), expected_test_worlds=1)
    t = out["table"]
    # brute-force evidence: no information for any score that reads the logs
    for m in ("posterior", "pair_weight", "random", "sam", "observer", "msgi"):
        assert t["gen_obs:selection:maximal@4"][m]["auroc"]["estimate"] == pytest.approx(0.5)
    assert set(bm.ITEMS_ONLY).isdisjoint(t["gen_goal:selection:focused_sample@4"])
    assert set(bm.ITEMS_ONLY) <= set(t["gen_obs:selection:focused_sample@4"])
    assert out["primary_verdict"] is None  # no test worlds in this check
    st = out["diagnostics"]["gen_obs:selection:focused_sample@4"]["arms_status"]
    assert st["ok"] + st["iteration_cap"] == 1 and out["secondary"]["S3"]["holds"] is True
    text = json.dumps(out, default=str) + bm.markdown(out)
    for g in fetched:  # no element or recipe name of the fetched graphs in the report
        s = next(x for x in cfg["sets"] if x.get("graph") == g)
        cw = col.converted_world(s["graph"], s["worlds"], s["n_worlds"], s["conversion_seed"],
                                 s["goal_sample_seed"], 0)
        names = {n for _, n in cw.names if len(n) > 3}
        assert names and not [n for n in names if re.search(rf"\b{re.escape(n)}\b", text)]


def tiny_cfg(tmp_path, policies=("focused_sample",), episodes=2, omit=0.663):
    raw = yaml.safe_load((ROOT / "configs" / "inference_baselines" / "dev.yaml").read_text())
    raw["run"].update(runs_dir=str(tmp_path / "runs"), ledger=str(tmp_path / "ledger.json"))
    raw["policies"] = list(policies)
    raw["walker"]["omit_prob"] = omit
    raw["sets"] = [dict(raw["sets"][0], n_worlds=1, episodes_per_world=episodes)]
    return col.validate(raw)


def test_collect_resume_skips_completed_runs_and_refuses_a_short_allowance(tmp_path):
    cfg = tiny_cfg(tmp_path)
    ledger = SessionLedger(None, {"dev": 10_000})
    first = col.collect(cfg, ledger, stamp="a")
    again = col.collect(cfg, ledger, stamp="b", resume=True)
    assert again[0]["skipped"] and again[0]["run_id"] == first[0]["run_id"]
    assert not col.collect(tiny_cfg(tmp_path, omit=0.5), ledger, stamp="c", resume=True)[0].get("skipped")
    with pytest.raises(col.AllowanceShort):
        col.collect(cfg, SessionLedger(None, {"dev": 5}), stamp="d")
    man = json.loads((tmp_path / "runs" / first[0]["run_id"] / "manifest.json").read_text())
    assert man["config_file"] == "<dict>"  # the config's origin is recorded, not hashed
    assert "source" not in json.loads((tmp_path / "runs" / first[0]["run_id"] / "config.json").read_text())


def _fake_run(src: pathlib.Path, name: str, *, status="completed", start=None, set_name=None, code=None,
              evidence=True) -> pathlib.Path:
    d = src.parent / name
    d.mkdir()
    man = json.loads((src / "manifest.json").read_text())
    cfg = json.loads((src / "config.json").read_text())
    man.update(run_id=name, status=status, stop_reason="tasks_done" if status == "completed" else None)
    man["start_time"] = start or man["start_time"]
    if code:
        man["source"]["loaded_code_sha256"] = code
    if set_name:
        cfg["set"]["name"] = set_name
    (d / "manifest.json").write_text(json.dumps(man))
    (d / "config.json").write_text(json.dumps(cfg))
    if evidence:
        (d / "evidence").mkdir()
        for f in (src / "evidence").glob("*.json"):
            (d / "evidence" / f.name).write_bytes(f.read_bytes())
    return d


def test_load_evidence_replaces_crashed_runs_and_rank_refuses_mixed_code(tmp_path):
    from hypergraph_agent.inference_baselines.__main__ import main
    cfg = tiny_cfg(tmp_path)
    done = col.collect(cfg, SessionLedger(None, {"dev": 10_000}), stamp="b")[0]["run_id"]
    src = tmp_path / "runs" / done
    _fake_run(src, "x2-dev-crashed", status="running", start="2000-01-01T00:00:00", evidence=False)
    _fake_run(src, "x2-dev-lost", status="running", set_name="other", evidence=False)
    worlds, sel = col.load_evidence(tmp_path / "runs", "x2-dev-")
    assert sel["reruns_used"] == [{"incomplete": "x2-dev-crashed", "rerun": done}]
    assert sel["incomplete_chosen"] == ["x2-dev-lost"] and len(worlds) == 1 and worlds[0]["complete"]
    _fake_run(src, "x2-dev-othercode", set_name="third", code="0" * 64)
    assert len(col.load_evidence(tmp_path / "runs", "x2-dev-")[1]["provenance"]) == 2
    assert main(["rank", "--runs", str(tmp_path / "runs"), "--prefix", "x2-dev-",
                 "--out", str(tmp_path / "o"), "--cache", str(tmp_path / "c")]) == 2


def test_rank_cache_resume_and_selection_completeness(tmp_path, monkeypatch):
    recs, _ = records(observe="all", episodes=4)
    monkeypatch.setitem(bm.PRIMARY, "set", "gen")
    monkeypatch.setitem(bm.PRIMARY, "k", 4)
    a = bm.rank_benchmark(recs, checkpoints=(2, 4), cache_dir=tmp_path, expected_selection_worlds=1)
    assert a["best_baseline"]["focused_sample"]["selected"] in bm.OPEN_SOURCE
    assert a["rows_reused"] == 0 and a["arms_reused"] == 0
    assert a["provenance"]["code"] == provenance.code_sha256()
    b = bm.rank_benchmark(recs, checkpoints=(2, 4), cache_dir=tmp_path, resume=True,
                          expected_selection_worlds=2)
    assert b["rows_reused"] == len(recs) and b["arms_reused"] == b["arms_jobs"] > 0
    assert sum(b["arms_status_totals"].values()) == b["arms_jobs"] and "ARMS jobs by status" in bm.markdown(b)
    chk = bm.arms_seed_check(recs, a, [1])  # exploratory: ARMS under another string hash seed
    one = chk["seeds"]["1"]["by_source"]["focused_sample"]
    assert one["worlds"] == 1 and one["selected"] in bm.OPEN_SOURCE
    assert isinstance(one["arms_below_observer"], bool)

    def metrics(s):  # the cached run reproduces every value (the selected baseline differs by design)
        return json.dumps({c: {m: v for m, v in row.items() if m != "best"} for c, row in s["table"].items()},
                          sort_keys=True)

    assert metrics(a) == metrics(b)
    sel = b["best_baseline"]["focused_sample"]  # one selection world where two are expected: no selection
    assert sel["selected"] is None and sel["n_worlds"] == 1 and b["primary_verdict"] is None
    assert set(b["secondary"]) >= {"S1", "S2", "S3", "S4", "S5", "S6", "S7"}
    f = next((tmp_path / a["provenance"]["cache_key"] / "rows").glob("*.json"))
    d = json.loads(f.read_text())
    d["value"]["row"]["code"] = "0" * 64
    f.write_text(json.dumps(d))
    with pytest.raises(bm.CodeMismatch):
        bm.rank_benchmark(recs, checkpoints=(2, 4), cache_dir=tmp_path, resume=True)


# ------------------------------------------------------------------ hygiene

def test_study_x2_files_have_short_lines_and_no_dashes():
    files = list((ROOT / "src" / "hypergraph_agent" / "inference_baselines").glob("*.py")) + CONFIGS
    files += [ROOT / "tests" / "test_inference_baselines.py", ROOT / "third_party" / "msgi" / "SOURCE.md",
              ROOT / "docs" / "studies" / "X2_inference_baselines.md"]
    dash = chr(0x2014)
    for f in files:
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            assert len(line) <= 110, f"{f.name}:{i}"
            assert dash not in line, f"{f.name}:{i}"
