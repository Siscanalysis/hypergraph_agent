"""External benchmarks: checksums, loaders, conversions and statistics.

Benchmarks whose data must be fetched (``python benchmarks/<name>/fetch.py``)
are skipped with a message when their files are absent.
"""

import importlib.util
import json
import sys
from dataclasses import replace

import pytest

from hypergraph_agent.benchmarks import loaders, sources, stats
from hypergraph_agent.benchmarks.graph import Recipe, RecipeGraph, Unlock, make_graph
from hypergraph_agent.benchmarks.to_recipequest import (
    ConversionError, recipequest_world, sink_goals,
)
from hypergraph_agent.benchmarks.to_techtree import (
    concept_dag, concept_stats, techtask, techworld_concepts, techworld_from_graph,
)
from hypergraph_agent.envs.generator import PROFILE_KNOWN, PROFILE_UNKNOWN, PROFILES, TaskConfig, make_world
from hypergraph_agent.envs.public_schema import public_view
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.envs.reference_solver import bfs_shortest, reference_solve
from hypergraph_agent.envs.vocabulary import BASE_TYPES, ITEM_TYPES, TYPE_NAMES
from hypergraph_agent.techtree.env import TechTreeEnv
from hypergraph_agent.techtree.generator import TechWorldConfig, make_world as make_techworld
from hypergraph_agent.techtree.generator import reference_solve as tech_reference

COMMITTED = ("crafter", "psketch_craft", "msgi_mining")
FETCHED = ("little_alchemy", "freeciv")


def need(benchmark):
    if not loaders.available(benchmark):
        pytest.skip(f"benchmarks/{benchmark} data absent: run python benchmarks/{benchmark}/fetch.py")


# ------------------------------------------------------------- checksums
@pytest.mark.parametrize("benchmark", loaders.BENCHMARKS)
def test_manifest_is_complete(benchmark):
    m = sources.manifest(benchmark)
    assert m["benchmark"] == benchmark and m["files"]
    for f in m["files"]:
        assert f["status"] in ("committed", "fetched") and len(f["sha256"]) == 64 and f["url"]
        if f["status"] == "fetched":
            assert f["path"].startswith("data/"), "fetched files go to the untracked data/ folder"
    if any(f["status"] == "fetched" for f in m["files"]):
        assert (sources.folder(benchmark) / "fetch.py").exists()
    for doc in ("SOURCE.md", "stats.json"):
        assert (sources.folder(benchmark) / doc).exists()


@pytest.mark.parametrize("benchmark", loaders.BENCHMARKS)
def test_committed_files_verify(benchmark):
    status = sources.verify(benchmark, "committed")
    assert all(s == "ok" for s in status.values()), status


WITH_FETCHED = tuple(b for b in loaders.BENCHMARKS
                     if any(f["status"] == "fetched" for f in sources.manifest(b)["files"]))


@pytest.mark.parametrize("benchmark", WITH_FETCHED)
def test_fetched_files_verify(benchmark):
    status = sources.verify(benchmark, "fetched")
    present = {p: s for p, s in status.items() if s != "missing"}
    if not present:
        pytest.skip(f"benchmarks/{benchmark} data absent: run python benchmarks/{benchmark}/fetch.py")
    assert all(s == "ok" for s in present.values()), present


def test_fetch_only_benchmarks_commit_no_source_data():
    for b in FETCHED:
        assert all(f["status"] == "fetched" for f in sources.manifest(b)["files"])


def test_gitignore_keeps_downloaded_benchmark_data_out():
    gitignore = sources.BENCHMARKS_DIR.parent / ".gitignore"
    if sources.BENCHMARKS_DIR.name != "benchmarks" or not gitignore.exists():
        pytest.skip("benchmarks folder relocated through HYPERGRAPH_BENCHMARKS_DIR")
    assert "benchmarks/*/data/" in gitignore.read_text(encoding="utf-8").splitlines()


def test_checksums_accept_exact_bytes_only(tmp_path):
    src = sources.folder("crafter") / "data.yaml"
    data = src.read_bytes()
    assert b"\r\n" not in data
    assert sources.sha256(src) == sources.manifest("crafter")["files"][0]["sha256"]
    crlf = tmp_path / "data.yaml"
    crlf.write_bytes(data.replace(b"\n", b"\r\n"))
    assert sources.sha256(crlf) != sources.sha256(src)


def test_missing_data_raises_a_clear_error(tmp_path, monkeypatch):
    for b in FETCHED:
        (tmp_path / b).mkdir()
        (tmp_path / b / "manifest.json").write_text(json.dumps(sources.manifest(b)))
    monkeypatch.setattr(sources, "BENCHMARKS_DIR", tmp_path)
    with pytest.raises(sources.DataMissing, match="fetch.py"):
        loaders.load("little_alchemy_2")
    with pytest.raises(sources.DataMissing, match="fetch.py"):
        loaders.load("freeciv_classic")


def test_msgi_graph_reproduces_from_the_source_pickle():
    pkl = sources.folder("msgi_mining") / "data" / "full_mining.pkl"
    if not pkl.exists():
        pytest.skip("run python benchmarks/msgi_mining/fetch.py to check the extraction")
    assert sources.verify("msgi_mining", "fetched")["data/full_mining.pkl"] == "ok"
    path = sources.folder("msgi_mining") / "extract.py"
    spec = importlib.util.spec_from_file_location("msgi_extract", path)
    mod = importlib.util.module_from_spec(spec)
    dont_write = sys.dont_write_bytecode
    sys.dont_write_bytecode = True  # no __pycache__ inside benchmarks/
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.dont_write_bytecode = dont_write
    text = mod.dumps(mod.document(mod.load_graphs(pkl)))
    committed = (sources.folder("msgi_mining") / "mining_graph.json").read_text(encoding="utf-8")
    assert committed.replace("\r\n", "\n") == text


# ---------------------------------------------------------------- loaders
def test_crafter_graph():
    g = loaders.load("crafter")
    assert len(g.derived) == 17 and len(g.recipes) == 25 and len(g.base) == 10
    (r,) = g.recipes_for("iron_pickaxe")
    assert r.requires == ("coal", "furnace", "iron", "table", "wood")
    assert {r.requires for r in g.recipes_for("table")} == {("mat:grass", "wood"), ("mat:path", "wood"),
                                                             ("mat:sand", "wood")}
    assert g.layers()["diamond"] == 8
    ach = loaders.crafter_achievements()
    assert len(ach) == 22 and ach["place_stone"] == "placed_stone" and ach["eat_cow"] is None
    assert all(e is None or e in g.elements for e in ach.values())


def test_psketch_graph():
    g = loaders.load("psketch_craft")
    assert len(g.derived) == 11
    assert g.recipes_for("bed")[0].requires == ("grass", "plank", "workshop1")
    assert g.recipes_for("gem")[0].requires == ("axe", "src:gem")
    assert g.kind_of("workshop0") == "facility"
    assert set(loaders.psketch_goals()) <= set(g.derived) and len(loaders.psketch_goals()) == 10


def test_msgi_graph():
    g = loaders.load("msgi_mining")
    assert len(g.derived) == 23 and {"Cut wood", "Get stone", "Get string"} <= g.base
    assert {r.requires for r in g.recipes_for("Light furnace")} == {
        ("Make firewood", "obj:furnace"), ("Get coal", "obj:furnace")}
    assert g.layers()["Make electrum bracelet"] == 8


def test_little_alchemy_graphs():
    need("little_alchemy")
    g1, g2 = loaders.load("little_alchemy_1"), loaders.load("little_alchemy_2")
    assert len(g1.elements) == 540 and g1.base == frozenset({"air", "earth", "fire", "water"})
    assert len(g2.elements) == 720 and len(g2.base) == 4 and len(g2.unlocks) == 9
    assert {u.kind for u in g2.unlocks} == {"progress", "k_of_n"}
    assert all(len(r.requires) <= 2 for g in (g1, g2) for r in g.recipes)
    assert all(e in g2.layers() for e in g2.derived)


def test_freeciv_graphs():
    need("freeciv")
    for name in ("freeciv_classic", "freeciv_civ2civ3"):
        g = loaders.load(name)
        assert len(g.base) + len(g.derived) == 87 and len(g.base) == 7
        assert all(1 <= len(r.requires) <= 3 for r in g.recipes)
        assert len(g.unlocks) > 100 and all(u.kind == "all" for u in g.unlocks)
    g = loaders.load("freeciv_classic")
    assert g.recipes_for("Iron Working")[0].requires == ("Bronze Working", "Warrior Code")


def test_freeciv_ruleset_parser():
    text = ('[advance_a]\nname = _("A")\nreq1 = "None"\nreq2 = "None"\n'
            '[advance_b]\nname = _("?tech:B")\nreq1 = "A"\nreq2 = "None"\n'
            '[advance_c]\nname = _("C")\nreq1 = "B"\nreq2 = "Never"\n'
            '[unit_x]\nname = _("X")\nreqs =\n    { "type", "name", "range"\n      "Tech", "B", "Player"\n'
            '      "Building", "Barracks", "City"\n    }\n')
    secs = loaders.ruleset_sections(text)
    assert [s for s, _ in secs] == ["advance_a", "advance_b", "advance_c", "unit_x"]
    assert loaders._field(secs[1][1], "name") == "B"
    assert loaders._reqs(secs[3][1]) == [{"type": "Tech", "name": "B", "range": "Player"},
                                         {"type": "Building", "name": "Barracks", "range": "City"}]


# ------------------------------------------------------------ neutral graph
def toy_graph():
    recipes = [Recipe("ab", ("a", "b")), Recipe("abc", ("ab", "c")), Recipe("abc", ("a", "bc")),
               Recipe("bc", ("b", "c")), Recipe("x", ("x", "a")), Recipe("a", ("b", "c"))]
    unlocks = [Unlock("star", "k_of_n", ("ab", "bc", "abc"), 2), Unlock("late", "progress", (), 7)]
    return make_graph("toy", {"a", "b", "c"}, recipes, unlocks)


def test_make_graph_cleans_and_levels():
    g = toy_graph()
    assert not g.recipes_for("a") and not g.recipes_for("x")  # base product and self loop dropped
    assert any("own product" in n for n in g.notes) and any("base element" in n for n in g.notes)
    lv = g.layers()
    assert lv["ab"] == lv["bc"] == 1 and lv["abc"] == 2 and lv["star"] == 2 and lv["late"] == 3
    assert "star" not in g.layers(use_unlocks=False)


def test_neutral_json_round_trip():
    for g in [toy_graph(), loaders.load("crafter"), loaders.load("msgi_mining")]:
        back = RecipeGraph.from_dict(json.loads(g.to_json()))
        assert back == g


def test_validation_rejects_malformed_graphs():
    with pytest.raises(ValueError):
        RecipeGraph("bad", ("a",), frozenset({"a"}), (Recipe("b", ("a",)),)).validate()
    with pytest.raises(ValueError):
        make_graph("bad", {"a"}, [Recipe("b", ("a",))], [Unlock("c", "majority", ("a",), 1)])


# ---------------------------------------------------- RecipeQuest conversion
def converted_worlds():
    out = []
    for name in ("crafter", "psketch_craft"):
        g = loaders.load(name)
        out.append(recipequest_world(g, sink_goals(g), seed=1))
    g = loaders.load("msgi_mining")
    out += [recipequest_world(g, [goal]) for goal in ("Make electrum bracelet", "Bake pork",
                                                      "Craft gold diamond necklace")]
    return out


def replay(task, plan):
    env = RecipeQuestEnv()
    env.reset(seed=0, options={"task": task})
    idx = env.public_spec.action_index()
    reward = 0.0
    for key in plan:
        _, reward, terminated, _, _ = env.step(idx[key])
        if terminated:
            break
    return reward


def test_converted_worlds_respect_recipequest_semantics():
    for cw in converted_worlds():
        w = cw.world
        items = {t for t, _ in w.levels}
        assert items <= set(ITEM_TYPES) and len(items) <= 16
        sigs = {p: set() for p in PROFILES}
        for r in w.recipes:
            assert r.effect in items and set(r.item_inputs) <= items
            assert 1 <= len(r.true_base) <= w.config.max_true
            assert set(r.true_base) <= set(r.pool) <= set(BASE_TYPES) and len(r.pool) == w.config.pool_size
            for p in PROFILES:
                assert r.signature(p) not in sigs[p]
                sigs[p].add(r.signature(p))
            assert all(w.level_of(i) < w.level_of(r.effect) or len(w.recipes_for(r.effect)) > 1
                       for i in r.item_inputs)
        for t, name in cw.names:
            assert (t in items) == (name in cw.items)


def test_converted_worlds_are_solvable_by_the_reference_solver():
    for cw in converted_worlds():
        for i, item in enumerate(cw.items):
            for profile in (PROFILE_KNOWN, PROFILE_UNKNOWN):
                task = cw.task(item, seed=i, cfg=TaskConfig(profile=profile, n_distractor_base=0))
                ref = reference_solve(task)
                assert ref.status == "optimal" and ref.length <= task.budget
                assert replay(task, ref.plan) == 1.0
        small = cw.task(cw.items[0])
        assert bfs_shortest(small)[0] == reference_solve(small).length


def test_converted_tasks_hide_true_base():
    for cw in converted_worlds():
        task = cw.task(cw.items[-1], cfg=TaskConfig(profile=PROFILE_UNKNOWN, n_distractor_base=0))
        spec = public_view(task)
        assert all(r.known_base is None and r.pool for r in spec.rules)
        # canary: change a hidden subset; the public view must not change
        r0 = task.rules[0]
        other = (min(set(r0.pool) - set(r0.true_base)),)
        changed = replace(task, rules=(replace(r0, true_base=other),) + task.rules[1:])
        assert changed.rules[0].true_base != r0.true_base
        assert public_view(changed) == spec
        # element names from the data never reach the public view (names that are
        # also vocabulary words, such as "stone", cannot be told apart and are skipped)
        text, vocab = repr(spec), " ".join(TYPE_NAMES)
        assert not any(name in text for _, name in cw.names if name not in vocab)


def test_conversion_promotes_and_refuses_as_documented():
    g = loaders.load("crafter")
    cw = recipequest_world(g, sink_goals(g))
    assert cw.promoted == ("wood",) and len(cw.items) == 16
    assert all(len(r.true_base) >= 1 for r in cw.world.recipes)
    with pytest.raises(ConversionError, match="items"):
        recipequest_world(loaders.load("msgi_mining"), sink_goals(loaders.load("msgi_mining")))
    with pytest.raises(ConversionError, match="base element"):
        recipequest_world(g, ["mat:tree"])
    keep_all = recipequest_world(g, sink_goals(g), max_alternatives=None)
    assert len(keep_all.world.recipes) > len(cw.world.recipes)


def test_conversion_is_deterministic_and_seeded():
    g = loaders.load("psketch_craft")
    a = recipequest_world(g, sink_goals(g), seed=3)
    assert a == recipequest_world(g, sink_goals(g), seed=3)
    b = recipequest_world(g, sink_goals(g), seed=4)
    assert a.world.recipes != b.world.recipes


def acyclic(world) -> bool:
    deps = {}
    for r in world.recipes:
        deps.setdefault(r.effect, set()).update(r.item_inputs)
    state = {}

    def visit(x):
        if state.get(x) == 1:
            return False
        if state.get(x) == 2:
            return True
        state[x] = 1
        ok = all(visit(y) for y in deps.get(x, ()))
        state[x] = 2
        return ok

    return all(visit(x) for x in deps)


def test_cycles_are_not_closed():
    recipes = [Recipe("x", ("a", "y")), Recipe("y", ("b", "x")), Recipe("y", ("a", "b")),
               Recipe("x", ("a", "b"))]
    g = make_graph("cyc", {"a", "b"}, recipes)
    cw = recipequest_world(g, ["x", "y"], max_alternatives=None)
    assert acyclic(cw.world) and len(cw.world.recipes) == 3
    assert any("would need" in d for d in cw.dropped)
    for cw in converted_worlds():
        assert acyclic(cw.world)


# ----------------------------------------------------- TechTree conversion
def two_parent_graph():
    recipes = [Recipe("ab", ("a", "b")), Recipe("bc", ("b", "c")), Recipe("ac", ("a", "c")),
               Recipe("ab_c", ("ab", "c")), Recipe("bc_a", ("a", "bc")), Recipe("ab_c_b", ("ab_c", "b")),
               Recipe("odd", ("ab", "bc"))]
    return make_graph("pairs", {"a", "b", "c"}, recipes)


def test_techworld_from_a_two_parent_graph():
    g = two_parent_graph()
    tc = techworld_from_graph(g, "linear", combo_length=1)
    w = tc.world
    assert tc.report["fitted_derived"] == 6 and "odd" not in tc.names
    for c, lv in enumerate(w.levels):
        assert len(w.sequences[c]) == w.config.length(lv)
        if lv >= 2:
            a, b = w.parents[c]
            assert w.sequences[c] == w.sequences[a] + w.sequences[b]
            assert sorted((w.levels[a], w.levels[b])) == sorted(w.config.parent_levels(lv))
    for goal in tc.names:
        task = techtask(tc, goal)
        ref = tech_reference(task)
        assert ref.status == "optimal"
        env = TechTreeEnv()
        env.reset(options={"task": task})
        reward = 0.0
        for a in ref.plan:
            _, reward, terminated, _, _ = env.step(a)
        assert reward == 1.0


def test_techworld_refuses_graphs_without_the_shape():
    with pytest.raises(ConversionError):
        techworld_from_graph(loaders.load("crafter"), "linear", combo_length=2)


def test_techworld_from_little_alchemy():
    need("little_alchemy")
    tc = techworld_from_graph(loaders.load("little_alchemy_2"), "linear", combo_length=1)
    assert tc.report["levels"] >= 3
    task = techtask(tc, tc.names[-1], budget=200)
    assert tech_reference(task).status == "optimal"


def test_concept_stats_match_techtree_reuse_depth():
    cfg = TechWorldConfig(n_primitives=3, n_levels=3, concepts_per_level=(3, 3, 3), combo_length=2,
                          length_rule="linear", reuse_depth=2)
    cs = concept_stats(techworld_concepts(make_techworld(5, cfg)))
    assert cs["reuse_depth_max"] == 2 and cs["other_parent_shares"] == {"level_1": 1.0}
    cs0 = concept_stats(techworld_concepts(make_techworld(5, replace(cfg, reuse_depth=0))))
    assert cs0["reuse_depth_max"] == 0
    dag = concept_dag(two_parent_graph())
    assert concept_stats(dag)["reuse_depth_max"] == 2


# ------------------------------------------------------------ statistics
@pytest.mark.parametrize("benchmark", loaders.BENCHMARKS)
def test_statistics_reproduce(benchmark):
    if benchmark in FETCHED:
        need(benchmark)
    committed = json.loads((sources.folder(benchmark) / "stats.json").read_text(encoding="utf-8"))
    assert json.loads(stats.dumps(stats.benchmark_stats(benchmark))) == committed


def test_generated_family_statistics_reproduce():
    committed = json.loads((sources.BENCHMARKS_DIR / "generated_families.json").read_text(encoding="utf-8"))
    assert json.loads(stats.dumps({"generated_by": committed["generated_by"],
                                   "families": stats.family_stats()})) == committed


def test_world_graph_matches_generated_worlds():
    from hypergraph_agent.envs.generator import WorldConfig
    w = make_world(11, WorldConfig())
    g = stats.world_graph(w, "gen")
    assert len(g.derived) == 16 and len(g.recipes) == len(w.recipes)
    assert max(g.layers().values()) == 8
