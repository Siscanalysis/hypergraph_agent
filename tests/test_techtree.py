"""TechTree environment, generator, reference, study U agents, boundary and runner."""

import ast
import pathlib
from dataclasses import replace

import numpy as np
import pytest

from hypergraph_agent.techtree.config import load_config
from hypergraph_agent.techtree.env import PublicEnv, TaskToken, TechTreeEnv, agent_interface, public_view
from hypergraph_agent.techtree.explorers import Episode, Explorer, make_explorer
from hypergraph_agent.techtree.generator import (
    TechStream, TechStreamConfig, TechWorldConfig, make_world, needs_unlock, reference_solve,
)

from helpers import meter

ROOT = pathlib.Path(__file__).resolve().parents[1]
TT = ROOT / "src" / "hypergraph_agent" / "techtree"
TT_CONFIGS = sorted((ROOT / "configs" / "unlock").glob("*.yaml")) \
    + sorted((ROOT / "configs" / "layers").glob("*.yaml"))

L_WORLD = TechWorldConfig()  # 3 primitives, levels 1-3, three concepts each, combos of 2
SMALL = TechWorldConfig(n_primitives=2, n_levels=2, concepts_per_level=(2, 2), combo_length=2)


def u_world(k=2, P=3, slots=2, levels=1):
    per = (2,) if levels == 1 else (3, 2)
    return TechWorldConfig(n_primitives=P, n_slots=slots, n_unlocks=1, unlock_level=levels, n_levels=levels,
                           concepts_per_level=per, combo_length=k)


def run(agent, env, task, m):
    """Play one task through the public interface agents receive."""
    penv, token_for = agent_interface(env)
    return agent.run_episode(penv, token_for(task), m)


def stream(world, goal_levels=(1,), rule="uniform", vis="announced", signal=False, n_worlds=1, epw=6,
           budget=100, seed=0, ns="unit"):
    return TechStream(TechStreamConfig(ns, seed, n_worlds, epw, world, goal_levels, rule, budget, vis,
                                       signal))


# ------------------------------------------------------------- generator
@pytest.mark.parametrize("rule", ["linear", "doubling"])
def test_lengths_parents_and_compositions(rule):
    cfg = TechWorldConfig(length_rule=rule, reuse_depth=2, concepts_per_level=(3, 3, 3))
    for seed in range(5):
        w = make_world(seed, cfg)
        for c, lv in enumerate(w.levels):
            assert len(w.sequences[c]) == cfg.length(lv)
            if lv >= 2:
                a, b = w.parents[c]
                assert a != b and sorted((w.levels[a], w.levels[b])) == sorted(cfg.parent_levels(lv))
                assert w.compositional[c] and w.sequences[c] == w.sequences[a] + w.sequences[b]
        for lv in range(1, 4):
            assert len({w.sequences[c] for c in w.at(lv)}) == len(w.at(lv))
            assert lv == 1 or len({w.parents[c] for c in w.at(lv)}) == len(w.at(lv))


@pytest.mark.parametrize("depth", [0, 1])
def test_flat_concepts_are_no_composition_of_two_others(depth):
    cfg = replace(L_WORLD, reuse_depth=depth)
    for seed in range(8):
        w = make_world(seed, cfg)
        seqs = list(w.sequences)
        for c, lv in enumerate(w.levels):
            assert w.compositional[c] == (2 <= lv <= depth + 1)
            if lv >= 2 and not w.compositional[c]:
                s = seqs[c]
                n = len(seqs)
                assert all(s != seqs[a] + seqs[b] for a in range(n) for b in range(n) if a != b)


def test_worlds_are_coupled_across_reuse_depth():
    for seed in range(5):
        w0, w1, w2 = (make_world(seed, replace(L_WORLD, reuse_depth=d)) for d in (0, 1, 2))
        assert w0.keys == w1.keys == w2.keys and w0.parents == w1.parents == w2.parents
        assert [w0.sequences[c] for c in w0.at(1)] == [w2.sequences[c] for c in w2.at(1)]
        assert [w1.sequences[c] for c in w1.at(2)] == [w2.sequences[c] for c in w2.at(2)]


def test_unlock_layout():
    for seed in range(10):
        w = make_world(seed, u_world(k=3, P=4, slots=3))
        (key, slot), = w.unlocks
        entry, = w.entries
        assert key != entry and 4 <= slot < 7
        assert all(s < 4 for s in w.sequences[key])
        assert sum(s >= 4 for s in w.sequences[entry]) == 1 and slot in w.sequences[entry]
        assert needs_unlock(w, entry) and not needs_unlock(w, key)


def test_invalid_configs_are_rejected():
    with pytest.raises(ValueError):
        TechWorldConfig(concepts_per_level=(3, 3))
    with pytest.raises(ValueError):
        TechWorldConfig(n_primitives=2, concepts_per_level=(5, 3, 3))
    with pytest.raises(ValueError):
        TechWorldConfig(n_unlocks=1, n_slots=0)
    with pytest.raises(ValueError):
        TechWorldConfig(n_primitives=4, combo_length=4)  # level-3 window space too large
    with pytest.raises(ValueError):
        TechStreamConfig("unit", 0, world=L_WORLD, goal_rule="unlock")


# ------------------------------------------------------------- dynamics
def _press(env, seq):
    obs = None
    for a in seq:
        obs, *_ = env.step(a)
    return obs


def test_gating_fixpoint_and_trace_clearing():
    w = make_world(3, replace(L_WORLD, reuse_depth=2))
    goal = w.at(3)[0]
    t = stream(replace(L_WORLD, reuse_depth=2), goal_levels=(3,)).task(0, 0)
    t = replace(t, world=w, goal=goal)
    env = TechTreeEnv()
    env.reset(options={"task": t})
    a, b = w.parents[goal]
    # typing the goal's sequence fires its whole chain within the presses (compositional world)
    obs = _press(env, w.sequences[goal])
    assert all(obs.held[x] for x in w.ancestors(goal))
    env.reset(options={"task": t})
    lvl2 = next(x for x in w.parents[goal] if w.levels[x] == 2)
    # without its level-1 parents held, a level-2 concept cannot fire from a fresh trace
    p1, p2 = w.parents[lvl2]
    seq = w.sequences[lvl2]
    obs = _press(env, (env.public_spec.wait_action,) + seq)
    assert obs.held[lvl2]  # compositional: its parents fire along the way
    env.reset(options={"task": t})
    half = len(w.sequences[p1])
    obs = _press(env, seq[:half] + (env.public_spec.wait_action,) + seq[half:])
    assert obs.held[p1] and not obs.held[lvl2]  # wait cleared the trace


def test_flat_concept_needs_its_parents():
    # depth 0: every level >= 2 concept is flat; pick one whose sequence contains no parent combo
    w, x = next((w, x) for w in (make_world(s, L_WORLD) for s in range(30)) for x in w.at(2)
                if not any(w.sequences[p] in {w.sequences[x][i:i + 2] for i in range(3)}
                           for p in w.parents[x]))
    t = replace(stream(L_WORLD, goal_levels=(2,)).task(0, 0), world=w, goal=x)
    env = TechTreeEnv()
    env.reset(options={"task": t})
    obs = _press(env, w.sequences[x])
    assert not obs.held[x]
    for p in w.parents[x]:
        _press(env, (env.public_spec.wait_action,) + w.sequences[p])
    obs = _press(env, w.sequences[x])
    assert obs.held[x]


def test_submit_success_and_deadline():
    s = stream(L_WORLD, goal_levels=(1,), budget=4)
    t = s.task(0, 0)
    env = TechTreeEnv()
    env.reset(options={"task": t})
    spec = env.public_spec
    _, r, term, _, info = env.step(spec.submit_action)
    assert r == 0 and not term
    for a in t.world.sequences[t.goal]:
        env.step(a)
    _, r, term, _, info = env.step(spec.submit_action)
    assert r == 1 and term and info["termination"] == "success"
    env.reset(options={"task": t})
    for _ in range(4):
        _, r, term, _, info = env.step(spec.wait_action)
    assert term and r == 0 and info["termination"] == "deadline"


@pytest.mark.parametrize("vis", ["announced", "silent"])
def test_unlocks_by_visibility(vis):
    s = stream(u_world(), rule="unlock", vis=vis)
    t = s.task(0, 0)
    w = t.world
    (key, slot), = w.unlocks
    env = TechTreeEnv()
    obs, _ = env.reset(options={"task": t})
    spec = env.public_spec
    assert obs.available[slot] == (vis == "silent")
    if vis == "announced":
        with pytest.raises(ValueError):
            env.step(slot)
    else:
        obs = _press(env, w.sequences[t.goal])  # inactive slot: the entry cannot fire
        assert not obs.held[t.goal]
    _press(env, (spec.wait_action,) + w.sequences[key])
    obs = _press(env, w.sequences[t.goal])
    assert obs.available[slot] and obs.held[key] and obs.held[t.goal]


def test_intermediate_signal():
    s = stream(u_world(k=3, P=4, slots=2), rule="unlock", signal=True)
    t = s.task(0, 0)
    w = t.world
    (key, _), = w.unlocks
    env = TechTreeEnv()
    obs, _ = env.reset(options={"task": t})
    assert obs.progress is False
    seq = w.sequences[key]
    obs, *_ = env.step(seq[0])
    assert obs.progress is True
    obs, *_ = env.step(seq[1])
    assert obs.progress is True
    obs, *_ = env.step(seq[2])  # completed: the key is held, no longer a proper prefix
    assert obs.held[key]
    env2 = TechTreeEnv()
    obs, _ = env2.reset(options={"task": replace(t, signal=False)})
    assert obs.progress is None


# ------------------------------------------------------------- reference
def _brute_force(task) -> int:
    """Independent breadth-first search over full states (held set, trace)."""
    w = task.world
    P, S = w.config.n_primitives, w.config.n_primitives + w.config.n_slots
    keys = w.slot_keys()
    maxlen = max(len(s) for s in w.sequences)
    frontier, seen, d = [(frozenset(), ())], set(), 0
    while frontier:
        d += 1
        nxt = []
        for held, tr in frontier:
            for a in range(S):
                t = (tr + (a if a < P or keys.get(a) in held else None,))[-maxlen:]
                h, changed = set(held), True
                while changed:
                    changed = False
                    for c, s in enumerate(w.sequences):
                        if c not in h and all(p in h for p in w.parents[c]) and t[-len(s):] == s:
                            h.add(c)
                            changed = True
                if task.goal in h:
                    return d + 1
                st = (frozenset(h), t)
                if st not in seen:
                    seen.add(st)
                    nxt.append(st)
        frontier = nxt
    raise AssertionError("goal unreachable")


@pytest.mark.parametrize("world", [SMALL, replace(SMALL, reuse_depth=1), u_world(), u_world(k=1, P=3),
                                   u_world(k=2, P=2, slots=2, levels=2)],
                         ids=lambda c: f"{c.n_levels}-{c.reuse_depth}-{c.n_unlocks}-{c.combo_length}")
def test_reference_is_exact_and_executable(world):
    rule = "unlock" if world.n_unlocks else "uniform"
    for seed in range(4):
        s = stream(world, goal_levels=(world.n_levels,), rule=rule, seed=seed, epw=2)
        for e in range(2):
            t = s.task(0, e)
            ref = reference_solve(t)
            assert ref.exact and ref.length == _brute_force(t)
            env = TechTreeEnv()
            env.reset(options={"task": t})
            for a in ref.plan:
                _, r, term, _, _ = env.step(a)
            assert term and r == 1


# ---------------------------------------------------------------- agents
def _soundness(agent, tasks):
    """Discoveries are exact, candidate preconditions contain the truth, failure
    clauses are true, and no unknown concept's sequence is eliminated where it
    could have fired."""
    for t in tasks:
        w = t.world
        k = agent.worlds.get(w.world_key)
        if k is None:
            continue
        keys = w.slot_keys()
        for c, seq in k.known.items():
            assert seq == w.sequences[c]
            if c not in k.revealed:
                assert set(w.parents[c]) | {keys[s] for s in seq if s in keys} <= k.cands[c]
            for h in k.bad.get(c, []):
                assert not (set(w.parents[c]) | {keys[s] for s in seq if s in keys}) <= h
        for c in range(len(w.levels)):
            if c in k.known:
                continue
            lv, code = w.levels[c], k.code(w.sequences[c])
            assert not k.elim[lv][code], (c, w.sequences[c])
            if k.silent and k.cond[lv][code] >= 0:
                assert not w.requirements(c) - {c} <= k.contexts[k.cond[lv][code]]


U_CASES = [(k, vis, sig) for k in (1, 2, 3) for vis in ("announced", "silent") for sig in (False, True)]


@pytest.mark.parametrize("k,vis,sig", U_CASES)
@pytest.mark.parametrize("arm", ["pooled", "isolated"])
def test_u_explorers_are_sound_and_solve(k, vis, sig, arm):
    cfg = u_world(k=k, P=3 if k < 3 else 3, slots=2)
    s = stream(cfg, rule="unlock", vis=vis, signal=sig, n_worlds=2, epw=5, budget=150, seed=k)
    tasks = [s.task(w, e) for w, e in s.order()]
    agent = make_explorer(arm, np.random.default_rng(0))
    m, env = meter(100_000), TechTreeEnv()
    rows = [run(agent, env, t, m) for t in tasks]
    _soundness(agent, tasks)
    for i in range(0, len(rows), 5):
        assert rows[i + 4]["success"]  # the last episode of every world is solved by replay
        assert rows[i + 4]["primitive_length"] <= 2 * k + 3


def test_isolated_tests_one_window_at_a_time():
    s = stream(u_world(k=3), rule="unlock", epw=1, budget=60)
    t = s.task(0, 0)
    agent = make_explorer("isolated", np.random.default_rng(0))
    env, m = TechTreeEnv(), meter()
    seen = []
    step = env.step

    def spy(a):
        seen.append(a)
        return step(a)

    env.step = spy
    run(agent, env, t, m)
    wait = len(env.public_spec.action_keys) - 2
    runs = "".join("|" if a == wait else "x" for a in seen).split("|")
    # between two clears at most one 3-press window (plus a replay of a known key)
    assert max(len(r) for r in runs[:-1]) <= 6 and seen.count(wait) >= 5


def test_blind_never_presses_a_slot_and_fails_unlock_goals():
    for vis in ("announced", "silent"):
        s = stream(u_world(), rule="unlock", vis=vis, epw=3)
        agent = make_explorer("blind", np.random.default_rng(0))
        env, m = TechTreeEnv(), meter()
        pressed = []
        step = env.step
        env.step = lambda a: (pressed.append(a), step(a))[1]
        rows = [run(agent, env, s.task(0, e), m) for e in range(3)]
        assert not any(r["success"] for r in rows)
        assert not any(env.public_spec.n_base <= a < env.public_spec.n_symbols for a in pressed)


def test_memory_amortizes_and_nomem_forgets():
    s = stream(u_world(k=3), rule="unlock", epw=4, budget=200)
    tasks = [s.task(0, e) for e in range(4)]
    rows = {}
    for arm in ("pooled", "nomem"):
        agent, env, m = make_explorer(arm, np.random.default_rng(0)), TechTreeEnv(), meter()
        rows[arm] = [run(agent, env, t, m) for t in tasks]
        if arm == "nomem":
            assert not agent.worlds
    assert rows["pooled"][0]["primitive_length"] == rows["nomem"][0]["primitive_length"]
    later = {arm: sum(r["primitive_length"] for r in rows[arm][1:]) for arm in rows}
    assert later["pooled"] < later["nomem"]
    assert all(r["discoveries_new"] == 0 for r in rows["pooled"][1:])


def test_oracle_replays_supplied_links():
    from hypergraph_agent.techtree.generator import revealed_links
    for vis in ("announced", "silent"):
        s = stream(u_world(k=3), rule="unlock", vis=vis, n_worlds=3, epw=2)
        agent, env, m = make_explorer("oracle", np.random.default_rng(0)), TechTreeEnv(), meter()
        for w, e in s.order():
            t = s.task(w, e)
            agent.reveal(t.world.world_key, revealed_links(t.world))
            r = run(agent, env, t, m)
            ref = reference_solve(t).length
            assert r["success"] and ref <= r["primitive_length"] <= ref + 1
            assert r["search_steps"] == 0


def test_random_agent_uses_only_available_actions():
    s = stream(u_world(), rule="unlock", epw=2, budget=40)
    agent, env, m = make_explorer("random", np.random.default_rng(0)), TechTreeEnv(), meter()
    rows = [run(agent, env, s.task(0, e), m) for e in range(2)]
    assert sum(r["primitive_length"] for r in rows) == m.used


# ----------------------------------------------------- information boundary
def _canary_pair():
    s0 = stream(replace(L_WORLD, reuse_depth=0), goal_levels=(3,))
    s2 = stream(replace(L_WORLD, reuse_depth=2), goal_levels=(3,))
    a, b = s0.task(0, 0), s2.task(0, 0)
    return a, b


def test_public_view_ignores_hidden_structure():
    a, b = _canary_pair()
    assert a.world.sequences != b.world.sequences and a.world.compositional != b.world.compositional
    assert public_view(a) == public_view(b)
    w = a.world
    c = replace(a, world=replace(w, parents=tuple(reversed(w.parents)), unlocks=((0, 3),), entries=(1,)))
    assert public_view(c) == public_view(a)


def test_identical_public_views_give_identical_first_decisions():
    chunks = []
    for t in _canary_pair():
        agent, (penv, token_for) = Explorer(np.random.default_rng(0)), agent_interface(TechTreeEnv())
        ep = Episode(penv, token_for(t), meter())
        k = agent.knowledge(ep.spec)
        chunks.append(agent._decide(k, ep))
    assert chunks[0] == chunks[1]


AGENT_MODULES = ("explorers.py", "layered.py")
ALLOWED_IMPORTS = {"__future__", "collections", "dataclasses", "numpy", "env", "explorers"}


def test_agent_modules_import_nothing_private():
    for name in AGENT_MODULES:
        tree = ast.parse((TT / name).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]]
            else:
                continue
            assert set(mods) <= ALLOWED_IMPORTS, (name, mods)


FORBIDDEN_CALLS = {"getattr", "setattr", "delattr", "vars", "dir", "__import__", "eval", "exec", "compile",
                   "globals", "locals"}


def boundary_violations(source: str) -> list[str]:
    """Introspection an agent module may not use: the reflective builtins (called or referenced),
    dunder attributes, frame and traceback attributes (f_*, tb_*) and private attributes of any
    object other than ``self``."""
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and node.id in FORBIDDEN_CALLS:
            out.append(f"{node.lineno}: {node.id}")
        elif isinstance(node, ast.Attribute):
            a, own = node.attr, isinstance(node.value, ast.Name) and node.value.id == "self"
            if (a.startswith("__") and a.endswith("__")) or a.startswith(("f_", "tb_")) \
                    or (a.startswith("_") and not own):
                out.append(f"{node.lineno}: .{a}")
    return out


def test_agent_modules_use_no_introspection():
    for name in AGENT_MODULES:
        assert not boundary_violations((TT / name).read_text()), name


def test_boundary_check_catches_known_escapes():
    closure = "def f(env):\n    return getattr(env.reset, '__clo' + 'sure__')"
    frame = "import sys\ndef f():\n    return sys._getframe(1).f_locals['task']"
    for src in (closure, frame, "def f(e):\n    return e._reset", "def f(o):\n    return o.__dict__",
                "def f(o):\n    g = vars\n    return g(o)"):
        assert boundary_violations(src), src


def test_info_is_public_only():
    t = _canary_pair()[0]
    env = TechTreeEnv()
    _, info0 = env.reset(options={"task": t})
    _, _, _, _, info = env.step(0)
    assert info0 == {} and set(info) == {"termination", "fired"}


# --------------------------------------------------------- configs, runner
@pytest.mark.parametrize("path", TT_CONFIGS, ids=lambda p: p.name)
def test_every_techtree_config_loads_and_plans(path):
    from hypergraph_agent.techtree.study import plan
    cfg = load_config(path)
    out = plan(cfg)
    assert out["runs"] > 0 and out["caps"]["adaptive_total"] + out["caps"]["reporting_total"] > 0
    if not cfg["stream"]["namespace"].endswith("_dev"):
        assert cfg["run"]["requires_flag"] and cfg["run"]["allocation"] != "dev"


def test_config_rules(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x, studdy: U}\n")
    with pytest.raises(KeyError):
        load_config(bad)
    ns = tmp_path / "ns.yaml"
    ns.write_text("run: {name: x}\nstream: {namespace: techtree_u}\n")
    with pytest.raises(ValueError):
        load_config(ns)
    var = tmp_path / "var.yaml"
    var.write_text("run: {name: x}\nvariants: [{name: a, env: {}}]\n")
    with pytest.raises(KeyError):
        load_config(var)


def test_measurement_configs_need_the_flag():
    from hypergraph_agent.techtree.__main__ import main
    mains = [p for p in TT_CONFIGS if "dev" not in p.name]
    assert mains
    for p in mains:
        assert main(["--config", str(p), "--dry-run"]) == 0
        with pytest.raises(SystemExit):
            main(["--config", str(p)])


def test_runner_and_analysis_end_to_end(tmp_path):
    import time
    from hypergraph_agent.techtree.analysis import analyse, collect_runs
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    cfg_path = tmp_path / "t.yaml"
    cfg_path.write_text(f"""
run: {{name: tt-e2e, study: U, seeds: [0, 1], runs_dir: {(tmp_path / 'runs').as_posix()}}}
world: {{n_primitives: 3, n_slots: 2, n_unlocks: 1, n_levels: 1, concepts_per_level: [2], combo_length: 2}}
stream: {{namespace: unit_dev, n_worlds: 2, episodes_per_world: 4, goal_levels: [1], goal_rule: unlock,
          episode_budget: 60}}
variants:
  - {{name: ann, stream: {{unlock_visibility: announced}}}}
  - {{name: sil, stream: {{unlock_visibility: silent}}, arms: [pooled, isolated]}}
arms: [pooled, isolated, oracle, reference]
analysis:
  tests:
    - {{type: paired, name: iso_vs_pooled, metric: first_success_steps, a: isolated, b: pooled, variant: ann,
       expect: greater}}
    - {{type: ratio, name: ratio, metric: first_success_steps, a: isolated, b: pooled, variant: ann,
       band: [1, 10]}}
    - {{type: equivalence, name: eq, metric: steps_late, a: isolated, b: pooled, variant: ann, margin_abs: 1}}
    - {{type: interaction, name: ix, metric: first_success_steps, a: isolated, b: pooled, variant_hi: sil,
       variant_lo: ann, expect: greater}}
    - {{type: within_margin, name: wm, metric: steps_late, a: pooled, others: [isolated, oracle],
       variants: [ann], margin_rel: 1.0}}
    - {{type: all_success, name: ref_ok, arm: reference}}
    - {{type: at_least_reference, name: oracle_ok, arm: oracle}}
  verdict: {{rule: all, require: [iso_vs_pooled, ref_ok]}}
""")
    cfg = load_config(cfg_path)
    ledger = SessionLedger(None, {"dev": 100_000}, 100_000, 100_000)
    runs_dir = tmp_path / "runs"
    results = run_study(cfg, ledger, str(runs_dir))
    assert len(results) == 2 * (4 + 2)
    runs = collect_runs(runs_dir, "tt-e2e")
    rows = [row for r in runs for row in r["rows"]]
    assert len(rows) == 2 * 6 * 8 and all(r["complete"] for r in runs)
    rep = ("oracle", "reference")
    assert ledger.state["reporting_total"] == sum(r["primitive_length"] for r in rows if r["arm"] in rep)
    assert ledger.state["adaptive_total"] == sum(r["primitive_length"] for r in rows if r["arm"] not in rep)
    res = analyse(cfg, runs)
    by = {t["name"]: t for t in res["tests"]}
    assert by["ref_ok"]["pass"] and by["oracle_ok"]["pass"] and not res["incomplete_cells"]
    assert by["iso_vs_pooled"]["n_worlds"] == 4 and by["ix"]["paired_by_world"] and by["ratio"]["pass"]
    assert res["verdict"]["verdict"] in ("pass", "fail") and set(res["tables"]) == {"ann", "sil"}
    # an incomplete run blocks every test and verdict that reads its cell
    victim = next(d for d in runs_dir.iterdir() if d.name.startswith("tt-e2e-pooled-ann-s1"))
    lines = (victim / "eval.jsonl").read_text().splitlines()
    (victim / "eval.jsonl").write_text("\n".join(lines[:5]) + "\n")
    res = analyse(cfg, collect_runs(runs_dir, "tt-e2e"))
    by = {t["name"]: t for t in res["tests"]}
    assert "ann/pooled" in res["incomplete_cells"] and by["iso_vs_pooled"]["pass"] is None
    assert by["ref_ok"]["pass"] and res["verdict"]["verdict"] == "no verdict"
    # a deterministic rerun (resume) repeats only that run; the analysis uses it and lists it
    time.sleep(1.1)
    again = run_study(cfg, ledger, str(runs_dir), resume=True)
    assert [(r["arm"], r["variant"], r["seed"]) for r in again] == [("pooled", "ann", 1)]
    res = analyse(cfg, collect_runs(runs_dir, "tt-e2e"))
    assert not res["incomplete_cells"] and len(res["reruns_used"]) == 1
    assert res["verdict"]["verdict"] != "no verdict"
    # a second complete run of the same cell is refused
    time.sleep(1.1)
    run_study(load_config(cfg_path), ledger, str(runs_dir))
    with pytest.raises(ValueError):
        analyse(cfg, collect_runs(runs_dir, "tt-e2e"))


def test_verdict_rules():
    from hypergraph_agent.techtree.analysis import verdict
    spec = {"rule": "precedence", "primary": "P", "falsification": "F", "conditions": ["C"],
            "validity": ["V"]}

    def v(p=True, f=False, c=True, ok=True, blocked=()):
        tests = {"P": {"pass": p}, "F": {"pass": f}, "C": {"pass": c}, "V": {"pass": ok}}
        for n in blocked:
            tests[n]["incomplete"] = ["x/y"]
        return verdict(spec, tests)["verdict"]

    assert v() == "pass" and v(c=False) == "confounded" and v(f=True) == "negligible"
    assert v(p=False, f=True) == "falsified" and v(p=False) == "inconclusive"
    assert v(ok=False) == "fail (validity)" and v(ok=False, blocked=("V",)) == "no verdict"
    assert v(blocked=("P",)) == "no verdict" and v(blocked=("C",)) == "no verdict"
    assert v(p=False, blocked=("C",)) == "inconclusive"
    every = {"rule": "all", "require": ["P", "V"]}
    assert verdict(every, {"P": {"pass": True}, "V": {"pass": True}})["verdict"] == "pass"
    assert verdict(every, {"P": {"pass": False}, "V": {"pass": True}})["verdict"] == "fail"
    blocked = {"P": {"pass": None, "incomplete": ["a/b"]}, "V": {"pass": True}}
    assert verdict(every, blocked)["verdict"] == "no verdict"


def test_duplicate_runs_are_refused_unless_deterministic_reruns():
    from hypergraph_agent.techtree.analysis import select_runs
    a = {"run_id": "r1", "variant": "v", "arm": "x", "seed": 0, "run_hash": "h", "start": "1",
         "code": ["sha", "commit"], "complete": False, "rows": []}
    b = {**a, "run_id": "r2", "start": "2", "complete": True}
    chosen, reruns = select_runs([b, a])
    assert [r["run_id"] for r in chosen] == ["r2"] and reruns[0]["replaced"] == ["r1"]
    for bad in ({**a, "complete": True}, {**a, "run_hash": "other"}, {**a, "code": ["other", "commit"]},
                {**a, "code": ["sha", "other"]}):
        with pytest.raises(ValueError):
            select_runs([bad, b])


def test_agents_hold_nothing_private():
    t = _canary_pair()[0]
    penv, token_for = agent_interface(TechTreeEnv())
    token = token_for(t)
    assert isinstance(token, TaskToken) and not [a for a in dir(token) if not a.startswith("__")]
    assert {a for a in dir(penv) if not a.startswith("__")} == {"_reset", "_step", "_spec", "reset", "step",
                                                               "public_spec"}
    for f in (penv._reset, penv._step, penv._spec):
        assert not [a for a in dir(f) if not a.startswith("__")]
    with pytest.raises(AttributeError):
        penv.env = TechTreeEnv()
    obs, info = penv.reset(token)
    assert info == {} and isinstance(penv, PublicEnv)
    with pytest.raises(KeyError):
        penv.reset(token)  # a token opens one episode


def test_reference_and_oracle_are_identical_across_signal_and_visibility():
    from hypergraph_agent.techtree.generator import revealed_links
    from hypergraph_agent.techtree.study import reference_episode
    for k in (1, 2, 3):
        outcomes = {}
        for vis in ("announced", "silent"):
            for sig in (False, True):
                s = stream(u_world(k=k, P=4, slots=3), rule="unlock", vis=vis, signal=sig, n_worlds=3, epw=2)
                oracle, env = make_explorer("oracle", np.random.default_rng(0)), TechTreeEnv()
                out = []
                for w, e in s.order():
                    t = s.task(w, e)
                    ref = reference_solve(t)
                    oracle.reveal(t.world.world_key, revealed_links(t.world))
                    r = run(oracle, env, t, meter())
                    penv, token_for = agent_interface(env)
                    rr = reference_episode(penv, token_for(t), meter(), ref)
                    out.append((t.task_key, ref.plan, r["primitive_length"], r["fired"],
                                rr["primitive_length"], rr["success"]))
                outcomes[(vis, sig)] = out
        assert len({repr(o) for o in outcomes.values()}) == 1, k


def test_pooled_and_isolated_act_identically_for_one_press_composites():
    for vis in ("announced", "silent"):
        s = stream(u_world(k=1, P=4, slots=3), rule="unlock", vis=vis, n_worlds=2, epw=3)
        tasks = [s.task(w, e) for w, e in s.order()]
        rows = {}
        for arm in ("pooled", "isolated"):
            agent, env, m = make_explorer(arm, np.random.default_rng(5)), TechTreeEnv(), meter()
            rows[arm] = [(r["primitive_length"], r["fired"]) for r in (run(agent, env, t, m) for t in tasks)]
        assert rows["pooled"] == rows["isolated"]


def test_namespace_allocation_and_cli_subset_rules(tmp_path):
    from hypergraph_agent.techtree.__main__ import main
    alloc = tmp_path / "alloc.yaml"
    alloc.write_text("run: {name: x, allocation: u_main}\nstream: {namespace: techtree_dev}\n")
    with pytest.raises(ValueError):
        load_config(alloc)
    arms = tmp_path / "arms.yaml"
    arms.write_text("run: {name: x}\narms: [pooled]\nvariants: [{name: a, arms: [random]}]\n")
    with pytest.raises(ValueError):
        load_config(arms)
    verdict = tmp_path / "verdict.yaml"
    verdict.write_text("run: {name: x}\nanalysis: {verdict: {rule: all, require: [nope]}}\n")
    with pytest.raises(KeyError):
        load_config(verdict)
    main_cfg = next(p for p in TT_CONFIGS if "dev" not in p.name)
    with pytest.raises(SystemExit):
        main(["--config", str(main_cfg), "--allow-measurement", "--arms", "reference"])


def test_measurement_runs_are_not_repeated(tmp_path):
    import json
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    p = tmp_path / "m.yaml"
    p.write_text(f"""
run: {{name: tt-meas, study: U, allocation: m, requires_flag: true,
      runs_dir: {(tmp_path / 'runs').as_posix()}}}
world: {{n_primitives: 3, n_slots: 2, n_unlocks: 1, n_levels: 1, concepts_per_level: [2], combo_length: 2}}
stream: {{namespace: unit_measure, n_worlds: 1, episodes_per_world: 2, goal_levels: [1], goal_rule: unlock}}
arms: [pooled, isolated]
""")
    cfg, runs_dir = load_config(p), tmp_path / "runs"
    ledger = SessionLedger(None, {"m": 10_000}, 10_000, 10_000)
    assert len(run_study(cfg, ledger, str(runs_dir))) == 2
    with pytest.raises(RuntimeError):
        run_study(cfg, ledger, str(runs_dir))  # complete runs exist: refuse without --resume
    assert run_study(cfg, ledger, str(runs_dir), resume=True) == []
    # an incomplete run made by other code cannot be resumed (the analysis would refuse the rerun)
    victim = next(d for d in runs_dir.iterdir() if d.name.startswith("tt-meas-pooled"))
    (victim / "eval.jsonl").write_text("")
    man = json.loads((victim / "manifest.json").read_text())
    man["source"]["loaded_code_sha256"] = "other"
    (victim / "manifest.json").write_text(json.dumps(man))
    with pytest.raises(RuntimeError):
        run_study(cfg, ledger, str(runs_dir), resume=True)


def test_techtree_lines_are_short():
    files = [*TT.glob("*.py"), ROOT / "tests" / "test_techtree.py", ROOT / "tests" / "test_layered.py"]
    long = [f"{f.name}:{n}" for f in files for n, line in enumerate(f.read_text().splitlines(), 1)
            if len(line) > 110]
    assert not long, long


def test_sign_test_and_bootstrap():
    from hypergraph_agent.techtree.analysis import bootstrap, sign_test
    assert sign_test(np.array([1.0] * 10)) == pytest.approx(2 / 2 ** 10)
    assert sign_test(np.array([1.0, -1.0, 0.0])) == 1.0
    d = np.arange(20, dtype=float)
    lo, hi = bootstrap(d, 2000, 0, 0.95)
    assert lo < d.mean() < hi


def test_world_metrics_tolerate_rows_without_discovery_counts():
    from hypergraph_agent.techtree.analysis import world_metrics
    base = {"variant": "v", "world_key": "W", "primitive_length": 5, "reference_length": 5, "success": True,
            "world_steps_before": 0, "fired": []}
    rows = [{**base, "arm": "pooled", "world_episode": 0, "discoveries_new": 2},
            {**base, "arm": "random", "world_episode": 0}]
    m = world_metrics(rows, 1)
    pooled = m[("v", "pooled")]["W"]
    assert pooled["discoveries_new"] == 2 and "discoveries_in_replay" not in pooled
    assert "discoveries_new" not in m[("v", "random")]["W"]
