"""Study P arms: imperfect learned skills on the study L engine."""

import ast
import json
import pathlib
from collections import Counter, defaultdict
from dataclasses import replace

import numpy as np
import pytest

from hypergraph_agent.techtree.config import load_config, run_hash, variants
from hypergraph_agent.techtree.env import TechTreeEnv, agent_interface
from hypergraph_agent.techtree.explorers import Chunk, Episode, Explorer
from hypergraph_agent.techtree.generator import (
    TechStream, TechStreamConfig, TechWorldConfig, make_world, revealed_links,
)
from hypergraph_agent.techtree.layered import make_layered
from hypergraph_agent.techtree.skills import (
    SKILL_KINDS, SkillRecord, execution_shares, make_skilled, pair_mask, parse_arm, reliability_curve,
)

from helpers import meter
from test_techtree import ALLOWED_IMPORTS, TT, _soundness, boundary_violations, run

ROOT = pathlib.Path(__file__).resolve().parents[1]
P_CONFIGS = sorted((ROOT / "configs" / "skills").glob("*.yaml"))
WORLD = TechWorldConfig()  # study L's world: 3 primitives, levels 1-3 with three concepts each
MACRO = SKILL_KINDS.index("search_macro")


def stream(depth, seed=0, n_worlds=1, epw=6, budget=100, ns="unit"):
    return TechStream(TechStreamConfig(ns, seed, n_worlds, epw, replace(WORLD, reuse_depth=depth), (3,),
                                       "uniform", budget))


def play(agent, s, m=None, reveal=False):
    """Rows (with the world key) and tasks of a whole stream, played through the public interface."""
    env, m = TechTreeEnv(), m or meter(500_000)
    rows, tasks = [], []
    for w, e in s.order():
        t = s.task(w, e)
        if reveal:
            agent.reveal(t.world.world_key, revealed_links(t.world))
        rows.append({**run(agent, env, t, m), "world_key": t.world.world_key, "world_episode": e})
        tasks.append(t)
    return rows, tasks


def skilled(arm, seed=0, **params):
    return make_skilled(arm, np.random.default_rng(seed), params)


def by_world(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["world_key"]].extend(r["skill_exec"])
    return out


# ------------------------------------------------------------- the skill model
def test_arm_names_and_parameters():
    assert parse_arm("promote_gate_e30") == ("promote_gate", 0.3)
    assert parse_arm("remember_e00") == ("remember", 0.0) and parse_arm("promote_e05") == ("promote", 0.05)
    for bad in ("promote", "promote_e3", "promote_exx", "sham_promote_e30", "none_e10", "remember_e100"):
        with pytest.raises(KeyError):
            skilled(bad)
    with pytest.raises(KeyError):
        skilled("remember_e30", gate_n=2)
    for params in ({"half_practice": 0}, {"gate_m": 4, "gate_k": 3}, {"gate_m": 0}):
        with pytest.raises(ValueError):
            skilled("remember_e30", **params)
    assert type(skilled("oracle_library")) is Explorer  # study L's anchor, exact execution


def test_error_follows_the_learning_curve():
    agent = skilled("remember_e30", half_practice=10)
    for n, e in ((0, 0.3), (10, 0.15), (30, 0.075), (90, 0.03)):
        assert agent.error(SkillRecord(n=n)) == pytest.approx(e)
    assert skilled("promote_e00").error(SkillRecord(n=0)) == 0.0


@pytest.mark.parametrize("arm", ["remember_e30", "promote_e10", "promote_gate_e30"])
def test_presses_slip_at_the_rate_of_the_curve(arm):
    # each pressed position of an execution slips with probability e_n; executions stop at the
    # first slip, so slips minus the summed e_n over pressed positions is a martingale
    agent = skilled(arm, seed=1)
    rows, _ = play(agent, stream(2, n_worlds=2, epw=12))
    e0 = parse_arm(arm)[1]
    entries = [x for r in rows for x in r["skill_exec"]]
    rate = [e0 / (1 + n / 10) for _, _, n, _, _, _ in entries]
    expected = sum(e * p for e, (_, _, _, _, p, _) in zip(rate, entries))
    var = sum(e * (1 - e) * p for e, (_, _, _, _, p, _) in zip(rate, entries))
    slips = sum(x[3] == 0 for x in entries)
    assert slips > 20 and abs(slips - expected) < 4 * var ** 0.5, (slips, expected, var)
    early = [x for x in entries if x[2] < 5]
    late = [x for x in entries if x[2] >= 20]
    per_press = [sum(x[3] == 0 for x in g) / sum(x[4] for x in g) for g in (early, late)]
    assert per_press[0] > per_press[1]  # practice lowers the error


def _episode(agent, task):
    agent.reveal(task.world.world_key, revealed_links(task.world))
    penv, token_for = agent_interface(TechTreeEnv())
    ep = Episode(penv, token_for(task), meter())
    return ep, agent.knowledge(ep.spec)


def test_a_slip_presses_another_primitive_stops_and_records_no_failure():
    t = stream(0).task(0, 0)
    agent = skilled("remember_e30")
    ep, k = _episode(agent, t)
    c = t.world.at(1)[0]
    seq = k.known[c]
    agent.draws = lambda wk, x, n: (np.array([1.0, 0.0])[:n], np.zeros(n))  # the second press slips
    agent._execute(k, ep, Chunk("replay", seq, target=c), Counter())
    assert ep.trace[0] == seq[0] and ep.trace[1] != seq[1] and len(ep.trace) == 2
    assert ep.obs.available[ep.trace[1]] and c not in ep.held
    assert agent.log[-1][2:5] == [0, 0, 2] and c not in k.bad  # slip, two presses, no failure clause
    agent.draws = lambda wk, x, n: (np.ones(n), np.zeros(n))  # no slip: typed exactly, fires
    ep.step(ep.spec.wait_action)
    agent._execute(k, ep, Chunk("replay", seq, target=c), Counter())
    assert c in ep.held and agent.log[-1][2:5] == [1, 1, 2] and agent.records[t.world.world_key][c].n == 2


def test_an_exact_typing_without_parents_still_records_a_true_failure():
    # depth 0: a flat level-2 concept whose sequence contains no level-1 combo fires nothing on the way
    w, x = next((w, x) for w in (make_world(s, WORLD) for s in range(40)) for x in w.at(2)
                if not {w.sequences[x][i:i + 2] for i in range(3)} & {w.sequences[c] for c in w.at(1)})
    t = replace(stream(0).task(0, 0), world=w, goal=w.at(3)[0])
    agent = skilled("remember_e30")
    ep, k = _episode(agent, t)
    agent.draws = lambda wk, c, n: (np.ones(n), np.zeros(n))
    agent._execute(k, ep, Chunk("replay", k.known[x], target=x), Counter())
    assert ep.trace == list(w.sequences[x]) and x not in ep.held and len(k.bad[x]) == 1
    assert not set(w.parents[x]) <= k.bad[x][0]


def test_controller_draws_are_common_across_arms_and_levels():
    agents = [skilled(a, seed=4) for a in ("remember_e10", "promote_e30", "promote_gate_e30")]
    assert len({a.ctrl_seed for a in agents}) == 1
    # the agent's own generator is not consumed: the next draw equals that of a study L agent
    ref = make_layered("promote", np.random.default_rng(4))
    assert agents[0].rng.integers(2 ** 31) == ref.rng.integers(2 ** 31)
    blocks = [[a.draws("Wabc", 2, 4)[0] for _ in range(5)] for a in agents]
    assert all(np.array_equal(x, y) for b in blocks[1:] for x, y in zip(blocks[0], b))
    # a press that slips at e0 = 0.1 also slips at e0 = 0.3 (same draw, larger threshold)
    lo, hi = (a.error(SkillRecord(n=3)) for a in agents[:2])
    u = np.concatenate(blocks[0])
    assert lo < hi and set(np.flatnonzero(u < lo)) <= set(np.flatnonzero(u < hi))


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_zero_error_reproduces_study_l(depth):
    s = stream(depth, seed=depth, n_worlds=2, epw=6)
    keys = ("primitive_length", "fired", "success", "replay_steps", "search_steps", "macro_steps")
    for base in ("remember", "promote"):
        rows_l, _ = play(make_layered(base, np.random.default_rng(7)), s)
        rows_p, _ = play(make_skilled(f"{base}_e00", np.random.default_rng(7), {"half_practice": 3}), s)
        assert [[r[x] for x in keys] for r in rows_l] == [[r[x] for x in keys] for r in rows_p], base
        assert all(r["executions_slip"] == 0 and r["practice_steps"] == 0 for r in rows_p)


@pytest.mark.parametrize("arm", ["remember_e30", "promote_e30", "promote_gate_e30", "promote_gate_e60"])
def test_evidence_stays_sound_under_imperfect_presses(arm):
    for depth in (0, 2):
        agent = skilled(arm, seed=depth)
        rows, tasks = play(agent, stream(depth, seed=depth, n_worlds=1, epw=6))
        _soundness(agent, tasks)
        assert sum(r["executions_slip"] for r in rows) > 0


# ------------------------------------------------------------- accounting
def test_every_press_is_charged_and_attributed():
    m = meter(500_000)
    agent = skilled("promote_gate_e30", seed=2)
    rows, _ = play(agent, stream(2, seed=2, n_worlds=2, epw=6), m)
    assert m.used == sum(r["primitive_length"] for r in rows)
    practice = sum(r["practice_steps"] for r in rows)
    assert practice > 0 and m.by_purpose["skill_practice"] == practice
    assert m.by_purpose["exploration"] == m.used - practice
    for r in rows:
        assert r["primitive_length"] == (r["replay_steps"] + r["search_steps"] + r["practice_steps"]
                                         + r["idle_steps"] + int(r["success"]))
        assert r["exec_steps"] == r["replay_steps"] + r["macro_steps"] + r["practice_steps"]
        assert r["exec_steps"] == sum(x[4] for x in r["skill_exec"])
        assert r["failed_exec_steps"] <= r["exec_steps"]
        assert r["executions"] == len(r["skill_exec"]) == (r["executions_ok"] + r["executions_slip"]
                                                           + r["executions_cut"])


# ------------------------------------------------------------------ the gate
def _admission_replay(entries, m, k):
    """Recompute each skill's record from its executions (checking that they are counted in order);
    also count the uses as a macro atom that came before admission."""
    recs = defaultdict(lambda: {"n": 0, "recent": [], "admitted": False})
    early = 0
    for c, _, n, outcome, _, kind in entries:
        r = recs[c]
        assert n == r["n"]
        early += kind == MACRO and not r["admitted"]
        r["n"] += 1
        if outcome >= 0:
            r["recent"] = (r["recent"] + [outcome == 1])[-k:]
            r["admitted"] = r["admitted"] or sum(r["recent"]) >= m
    return recs, early


@pytest.mark.parametrize("arm,m,k", [("promote_gate_e30", 2, 3), ("promote_gate_e10", 3, 4),
                                     ("promote_gate_e00", 2, 3)])
def test_gate_admits_after_m_of_the_last_k_and_atoms_are_admitted(arm, m, k):
    agent = skilled(arm, seed=3, gate_m=m, gate_k=k)
    rows, _ = play(agent, stream(2, seed=3, n_worlds=2, epw=6))
    for wk, entries in by_world(rows).items():
        recs, early = _admission_replay(entries, m, k)
        assert early == 0
        for c, r in recs.items():
            rec = agent.records[wk][c]
            assert (rec.n, rec.recent, rec.admitted) == (r["n"], r["recent"], r["admitted"])
            if parse_arm(arm)[1] == 0:
                assert rec.admitted == (sum(x[3] == 1 for x in entries if x[0] == c) >= m)
    assert sum(r["macro_steps"] for r in rows) > 0


def test_promote_uses_unadmitted_atoms_and_never_practises():
    agent = skilled("promote_e30", seed=3)
    rows, _ = play(agent, stream(2, seed=3, n_worlds=2, epw=6))
    assert sum(r["practice_steps"] for r in rows) == 0
    assert sum(_admission_replay(entries, 2, 3)[1] for entries in by_world(rows).values()) > 0


def test_practice_happens_only_when_no_admitted_macro_is_open():
    agent = skilled("promote_gate_e30", seed=5)
    search = agent._search
    seen = Counter()

    def spy(k, ep, level):
        wk = ep.spec.world_key
        open_ = k.open_mask(level, ep.held, agent.usable(k, ep))
        admitted = pair_mask(k, level, agent.atoms(k, wk)) & open_
        full = k.macro_mask(level, k.known) & open_
        chunk = search(k, ep, level)
        seen[chunk.kind] += 1
        if chunk.kind == "practice":
            assert not admitted.any() and full.any() and not agent.admitted(wk, chunk.target)
        elif admitted.any():
            assert chunk.kind == "search_macro" and admitted[chunk.code]
        else:
            assert chunk.kind != "search_macro"
        return chunk

    agent._search = spy
    play(agent, stream(2, seed=5, n_worlds=2, epw=6))
    assert seen["practice"] > 0 and seen["search_macro"] > 0


# ------------------------------------------------------------ row summaries
def test_execution_shares_and_reliability_curve():
    rows = [{"variant": "d2", "arm": "a", "primitive_length": 10, "exec_steps": 6, "practice_steps": 2,
             "failed_exec_steps": 3, "executions": 4, "executions_slip": 1,
             "skill_exec": [[0, 1, 0, 1, 2, 0], [0, 1, 1, 0, 1, 0], [3, 2, 0, -1, 1, 1], [0, 1, 2, 1, 2, 2]]},
            {"variant": "d2", "arm": "a", "primitive_length": 10, "exec_steps": 0, "practice_steps": 0,
             "failed_exec_steps": 0, "executions": 0, "executions_slip": 0, "skill_exec": []},
            {"variant": "d2", "arm": "reference", "primitive_length": 7}]
    sh = execution_shares(rows)
    assert sh == {"d2": {"a": {"exec_share": 0.3, "practice_share": 0.1, "failed_share": 0.15,
                               "slip_rate": 0.25}}}
    curve = reliability_curve(rows, bins=(0, 2))
    assert curve == {"d2": {"a": {1: [{"n_from": 0, "executions": 2, "ok_rate": 0.5},
                                      {"n_from": 2, "executions": 1, "ok_rate": 1.0}]}}}
    json.dumps({"shares": sh, "curve": curve})  # stored as JSON by the analysis


# --------------------------------------------------------------- boundary
def test_skills_module_respects_the_agent_boundary():
    source = (TT / "skills.py").read_text()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        assert set(mods) <= ALLOWED_IMPORTS | {"layered"}, mods
    assert not boundary_violations(source)


def test_skills_lines_are_short():
    files = [TT / "skills.py", pathlib.Path(__file__), *P_CONFIGS]
    long = [f"{f.name}:{n}" for f in files for n, line in enumerate(f.read_text().splitlines(), 1)
            if len(line) > 110]
    assert not long, long


# ---------------------------------------------------------- configs, runner
@pytest.mark.parametrize("path", P_CONFIGS, ids=lambda p: p.name)
def test_p_configs_load_and_plan(path):
    from hypergraph_agent.techtree.study import plan, reporting_arms
    cfg = load_config(path)
    out = plan(cfg)
    assert cfg["run"]["study"] == "P" and cfg["run"]["ledger"] == "runs/P-ledger.json"
    for v in out["variants"]:
        assert v["per_run_cap"] == v["worlds_per_seed"] * v["episodes_per_world"] * v["episode_budget"]
        for arm in v["arms"]:
            assert arm in reporting_arms("P") or make_skilled(arm, np.random.default_rng(0), cfg["agent"])
    if cfg["stream"]["namespace"].endswith("_dev"):
        assert cfg["run"]["allocation"] == "dev" and not cfg["run"]["requires_flag"]
    else:
        assert cfg["run"]["requires_flag"] and cfg["run"]["allocation"] != "dev"
        assert out["caps"]["adaptive_total"] == cfg["ledger"]["allocations"][cfg["run"]["allocation"]]


def test_p_configs_share_ledger_world_and_agent():
    assert [p.name for p in P_CONFIGS] == ["p_dev.yaml", "p_main.yaml"]
    dev, main = (load_config(ROOT / "configs" / "skills" / n) for n in ("p_dev.yaml", "p_main.yaml"))
    from hypergraph_agent.techtree.study import plan
    assert dev["ledger"] == main["ledger"] and dev["world"] == main["world"] and dev["agent"] == main["agent"]
    led = main["ledger"]
    assert led["adaptive_cap"] == sum(led["allocations"].values())
    reporting = [plan(c)["caps"]["reporting_total"] for c in (main, dev)]
    assert led["reporting_cap"] >= sum(reporting)
    assert main["stream"]["namespace"] == "techtree_p" and set(dev["arms"]) <= set(main["arms"])
    assert [v["name"] for v in variants(dev)] == [v["name"] for v in variants(main)]
    # agent parameters enter the run hash
    v = variants(main)[0]
    other = {**main, "agent": {**main["agent"], "gate_m": 3}}
    assert run_hash(main, v, "promote_gate_e30", 0) != run_hash(other, v, "promote_gate_e30", 0)


def test_p_measurement_config_needs_the_flag():
    from hypergraph_agent.techtree.__main__ import main
    path = str(ROOT / "configs" / "skills" / "p_main.yaml")
    assert main(["--config", path, "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        main(["--config", path])


def test_runner_plays_study_p_and_charges_each_kind(tmp_path):
    from hypergraph_agent.techtree.analysis import analyse, collect_runs
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    p = tmp_path / "p.yaml"
    p.write_text(f"""
run: {{name: p-e2e, study: P, runs_dir: {(tmp_path / 'runs').as_posix()}}}
stream: {{namespace: unit_dev, n_worlds: 1, episodes_per_world: 4, goal_levels: [3]}}
agent: {{half_practice: 10, gate_m: 2, gate_k: 3}}
variants: [{{name: d2, world: {{reuse_depth: 2}}}}]
arms: [remember_e30, promote_gate_e30, oracle_library, reference]
analysis:
  late_from: 2
  tests:
    - {{type: paired, name: gate, metric: steps_late, a: promote_gate_e30, b: remember_e30, variant: d2,
       expect: less}}
""")
    ledger = SessionLedger(None, {"dev": 50_000}, 50_000, 50_000)
    res = run_study(load_config(p), ledger, str(tmp_path / "runs"))
    assert [r["n"] for r in res] == [4, 4, 4, 4]
    runs = collect_runs(tmp_path / "runs", "p-e2e")
    rows = {r["arm"]: r["rows"] for r in runs}
    rep = ("oracle_library", "reference")
    assert ledger.state["reporting_total"] == sum(x["primitive_length"] for a in rep for x in rows[a])
    assert ledger.state["adaptive_total"] == sum(x["primitive_length"] for a in rows if a not in rep
                                                 for x in rows[a])
    by_run = {m["meta"]["arm"]: m["by_purpose"] for m in ledger.state["runs"].values()}
    assert by_run["promote_gate_e30"].get("skill_practice", 0) == sum(
        x["practice_steps"] for x in rows["promote_gate_e30"])
    assert all("skill_exec" in x for x in rows["remember_e30"])
    assert "skill_exec" not in rows["oracle_library"][0]
    res = analyse(load_config(p), runs)
    assert not res["incomplete_cells"] and res["tests"][0]["n_worlds"] == 1
    json.dumps(res)
    assert set(res["row_summaries"]) == {"execution_shares", "reliability_curve"}
    assert set(res["row_summaries"]["execution_shares"]["d2"]) == {"remember_e30", "promote_gate_e30"}
    table = res["tables"]["d2"]
    assert table["remember_e30"]["exec_steps"] > 0 and "practice_steps" in table["promote_gate_e30"]
    assert "exec_steps" not in table["oracle_library"] and "exec_steps" not in table["reference"]


# ------------------------------------------------- analysis extensions (generic)
def test_row_summaries_only_for_studies_that_declare_them():
    from hypergraph_agent.techtree.analysis import row_summaries
    rows = [{"variant": "v", "arm": "remember_e30", "primitive_length": 4, "exec_steps": 2,
             "practice_steps": 0, "failed_exec_steps": 1, "executions": 1, "executions_slip": 1,
             "skill_exec": [[0, 1, 0, 0, 2, 0]]}]
    assert row_summaries("U", rows) == {} and row_summaries("L", rows) == {}
    out = row_summaries("P", rows)
    assert out["execution_shares"]["v"]["remember_e30"]["failed_share"] == 0.25
    assert out["reliability_curve"]["v"]["remember_e30"][1][0]["ok_rate"] == 0.0


def test_interaction_low_cell_may_use_other_arms():
    from hypergraph_agent.techtree.analysis import cells_of, run_test
    worlds = ["w0", "w1", "w2"]
    values = {("hi", "a"): [10, 12, 14], ("hi", "b"): [20, 20, 20], ("hi", "c"): [15, 15, 15],
              ("hi", "d"): [18, 19, 20], ("lo", "a"): [30, 30, 30], ("lo", "b"): [25, 24, 23]}
    metrics = {cell: {w: {"steps_late": x} for w, x in zip(worlds, xs)} for cell, xs in values.items()}
    t = {"type": "interaction", "name": "x", "metric": "steps_late", "a": "a", "b": "b", "a_lo": "c",
         "b_lo": "d", "variant_hi": "hi", "variant_lo": "hi", "expect": "less"}
    res = run_test(t, metrics, [], 500, 0)
    assert res["per_world"] == {"w0": -7.0, "w1": -4.0, "w2": -1.0} and res["paired_by_world"]
    assert res["estimate"] == pytest.approx(-4.0) and res["pass"] is True
    assert cells_of(t, {}) == [("hi", "a"), ("hi", "b"), ("hi", "c"), ("hi", "d")]
    # without the new keys the low cell reads a and b, as before
    old = {k: v for k, v in t.items() if k not in ("a_lo", "b_lo")} | {"variant_lo": "lo"}
    res = run_test(old, metrics, [], 500, 0)
    assert res["per_world"] == {"w0": -15.0, "w1": -14.0, "w2": -13.0}
    assert cells_of(old, {}) == [("hi", "a"), ("hi", "b"), ("lo", "a"), ("lo", "b")]


def test_interaction_low_cell_keys_are_validated(tmp_path):
    def config(test):
        p = tmp_path / "c.yaml"
        p.write_text("run: {name: x}\narms: [pooled, isolated, oracle]\n"
                     "variants: [{name: a}, {name: b}]\n"
                     f"analysis: {{tests: [{test}]}}\n")
        return p
    base = "type: interaction, name: t, metric: steps_late, a: pooled, b: isolated"
    load_config(config(f"{{{base}, a_lo: oracle, variant_hi: a, variant_lo: a}}"))
    load_config(config(f"{{{base}, variant_hi: a, variant_lo: b}}"))
    with pytest.raises(ValueError):  # an arm that is not configured
        load_config(config(f"{{{base}, b_lo: nope, variant_hi: a, variant_lo: b}}"))
    with pytest.raises(ValueError):  # a cell compared with itself
        load_config(config(f"{{{base}, a_lo: pooled, variant_hi: a, variant_lo: a}}"))
    with pytest.raises(KeyError):  # the new keys belong to interaction tests only
        load_config(config("{type: paired, name: t, metric: steps_late, a: pooled, b: isolated, "
                           "a_lo: oracle, variant: a}"))
