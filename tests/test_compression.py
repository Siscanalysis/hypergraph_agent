"""Study CA (docs/studies/CA_compression_atoms.md).

Frozen behaviour first: without the ``latent`` option every TechTree config of studies U, L, U2, P
and PG loads to the same dictionary, generates the same worlds, tasks, reference plans and run hashes,
and every one of its arms acts as before; study X's TechTree streams and the benchmark families are
unchanged too (digests computed with the committed code before the option existed). Then the latent
worlds, the optional config key, the compressors, description lengths, the CA arms and their
information boundary, the configs (one ledger declaration, caps, flags, fresh namespaces), an
end-to-end run with the analysis and its row summaries, and the hygiene of the study's files."""

import ast
import hashlib
import importlib.util
import json
import pathlib
import re
from dataclasses import replace

import numpy as np
import pytest
import yaml

from hypergraph_agent.techtree.compression import (
    CORPORA, DEFAULT_PARAMS, bpe_fragments, chain_symbols, check_params, description_lengths, encode,
    layer_summary, library_summary, make_compression, min_tokens, parse_sexpr, spearman, stitch_fragments,
)
from hypergraph_agent.techtree.config import load_config, run_hash, stream_config, variant_arms, variants
from hypergraph_agent.techtree.env import TechTreeEnv, public_view
from hypergraph_agent.techtree.generator import (
    TechStream, TechStreamConfig, TechWorldConfig, debug_view, make_world, reference_solve, revealed_links,
)

from helpers import meter
from test_techtree import ALLOWED_IMPORTS, _brute_force, _soundness, boundary_violations, run

# the compressor of study CA, an optional package imported only where used; the tests that need it skip
# without it
needs_stitch = pytest.mark.skipif(importlib.util.find_spec("stitch_core") is None,
                                  reason="stitch-core (study CA's compressor) is not installed")
ROOT = pathlib.Path(__file__).resolve().parents[1]
TT = ROOT / "src" / "hypergraph_agent" / "techtree"
CONFIGS = sorted((ROOT / "configs" / "compression").glob("*.yaml"))
DOC = ROOT / "docs" / "studies" / "CA_compression_atoms.md"
FILES = [TT / "compression.py", pathlib.Path(__file__), *CONFIGS]
L_WORLD = TechWorldConfig()  # study L's layers: 3 primitives, three levels of three concepts
LATENT = replace(L_WORLD, reuse_depth=2, latent=2)


def rng(seed=0):
    return np.random.default_rng(seed)


def stream(world, seed=0, n_worlds=2, epw=4, budget=100, ns="ca_unit_dev", goal_levels=(3,)):
    return TechStream(TechStreamConfig(ns, seed, n_worlds, epw, world, goal_levels, "uniform", budget))


def play(arm, st, params=None, seed=0):
    agent, env, m = make_compression(arm, rng(seed), params), TechTreeEnv(), meter(1_000_000)
    rows, tasks = [], []
    for w, e in st.order():
        t = st.task(w, e)
        if arm == "oracle_library":
            agent.reveal(t.world.world_key, revealed_links(t.world))
        rows.append(run(agent, env, t, m))
        tasks.append(t)
    return agent, rows, tasks


# ------------------------------------------------------- frozen worlds are unchanged
# sha256 prefixes of every frozen TechTree config's worlds, tasks (with reference lengths and plans)
# and run hashes, of study X's TechTree test streams and of the benchmark families, computed with the
# TechTree code of 7ffa394 (unchanged up to ec5b442), before the latent option existed
FROZEN = {
    "layers/l1_main.yaml": ("f4e1b20ace1a44af", "1fb02956e33a22e6", "f0b89f32c361dc23"),
    "layers/l2_main.yaml": ("0ac40943e8628357", "d9a5c7adbe171b0f", "40aae09012335406"),
    "layers/l_dev.yaml": ("2bc85c694ad4ec35", "e16aea3776b654ad", "43ad35d782b9c0fb"),
    "unlock/u_main.yaml": ("5f6e01a70bac0b63", "00cf31e1892461a3", "198466bb9ddf6755"),
    "unlock/u_dev.yaml": ("219aebb7ee75fe0f", "a7cffaf821be5253", "0a3c5cd8c7d9d035"),
    "unlock2/u2_main.yaml": ("bf7378c4c39e0ab5", "7b6cdce5603ea7c8", "d120566272a8e0a2"),
    "unlock2/u2_dev.yaml": ("92408464d0bfb56d", "ffef6e57c5e25fd5", "9ba398948d7e2587"),
    "skills/p_main.yaml": ("346e6f8183b1c280", "bc7e915ef142f7e5", "48768a03146feda5"),
    "skills/p_dev.yaml": ("e81de919e4a242d1", "19eb8195ce0a431a", "dfb67a436108ef6a"),
    "policygraph_tt/pg_u_main.yaml": ("bad1949152a5341b", "d5494624d411e10c", "3ac0af30c7810dcd"),
    "policygraph_tt/pg_l_main.yaml": ("7297db4896ef38e3", "e03785012b0a9426", "1776041f9eaf9c7c"),
    "policygraph_tt/pg_dev.yaml": ("d8e2eed2f83b7d77", "ca37125d967c4690", "24a9b614a469dc51"),
}
FROZEN_X = {"tt_l2_d2_main.yaml": "372672e5624742c2", "tt_l2_d2_dev.yaml": "e6218864cd637096",
            "tt_u_k3_main.yaml": "bf76221a9b66394b", "tt_u_k3_dev.yaml": "56cb1059fbe23ff2"}
FROZEN_BENCH = {"TechTree-L-d0": "ee6907aa54f6862f", "TechTree-L-d1": "9772a2cb973f198b",
                "TechTree-L-d2": "fdb01a199e1487df", "TechTree-U-k3": "fc1ef3d24a6d2259"}
# sha256 prefixes of every TechTree config of studies U, L, U2, P and PG as loaded, and of the rows of
# every arm of every variant on one world of 4 episodes (first seed, the runner's agent seeds, supplied
# links and extra fields; per-episode timing dropped), computed with the code of ec5b442 (no TechTree
# file changed since 7ffa394), before the latent option existed
FROZEN_TT_CONFIGS = {  # config: (loaded dictionary, rows of every (variant, arm))
    "configs/layers/l1_main.yaml": ("3a50fcc22578f693d883783d", "1f1544e573a636e7e0329d96"),
    "configs/layers/l2_main.yaml": ("89a6ea08c0b8bada1221121c", "d596acac4a477b7a00d615df"),
    "configs/layers/l_dev.yaml": ("30960a955abdda6532cade52", "ff91aa7a23ae2a94d69034ac"),
    "configs/policygraph_tt/pg_dev.yaml": ("b723cb98bd0fd71c6909c6d1", "21dea20214c157d98897b9ae"),
    "configs/policygraph_tt/pg_l_main.yaml": ("e4cb07ab78c369d1445881e4", "bb21417968f2c93c36e82e9a"),
    "configs/policygraph_tt/pg_u_main.yaml": ("eb0a70cd09a3c882d4a5f40e", "667fb5eebcc2fcfeba8ef8b6"),
    "configs/skills/p_dev.yaml": ("466202567220500600e959d2", "041a6864ac894b23d0c2b90b"),
    "configs/skills/p_main.yaml": ("986fc66c8418688308fd1f50", "97f548d7964cc538c5ce6ab0"),
    "configs/unlock/u_dev.yaml": ("d8bee2c09f0e7724221001f5", "4a7b2432eac89c1a495361e0"),
    "configs/unlock/u_main.yaml": ("ca224283a4c766de4e8305ae", "83797f8f047e46b56f62340d"),
    "configs/unlock2/u2_dev.yaml": ("a4a23f83a0a997b6addbaf54", "c39c0d378df1ff9b58a75a67"),
    "configs/unlock2/u2_main.yaml": ("61e8a6c44480925600b6d4eb", "b1e873368a156380a721efab"),
}


def _world_fields(w):
    return [w.world_key, w.seed, w.attempt, list(w.keys), list(w.levels), [list(s) for s in w.sequences],
            [list(p) for p in w.parents], list(w.compositional), [list(u) for u in w.unlocks],
            list(w.entries)]


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()[:16]


def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()[:24]


def test_latent_off_keeps_every_frozen_world_task_plan_and_run_hash():
    for name, (worlds_d, tasks_d, hashes_d) in FROZEN.items():
        cfg = load_config(ROOT / "configs" / name)
        worlds, tasks, hashes = [], [], []
        for v in variants(cfg):
            for seed in cfg["run"]["seeds"]:
                scfg, _ = stream_config(cfg, v, seed)
                st = TechStream(scfg)
                made = [st.world(w) for w in range(scfg.n_worlds)]
                assert all(not w.fragments and not w.fragment_of for w in made)
                worlds += [[v["name"], seed, _world_fields(w)] for w in made]
                for w, e in st.order():
                    t = st.task(w, e)
                    ref = reference_solve(t)
                    tasks.append([v["name"], seed, t.task_key, t.goal, t.budget, w, e, ref.length,
                                  list(ref.plan)])
                hashes += [[v["name"], a, seed, run_hash(cfg, v, a, seed)] for a in variant_arms(cfg, v)]
        assert (_digest(worlds), _digest(tasks), _digest(hashes)) == (worlds_d, tasks_d, hashes_d), name


def _frozen_agent_rows(cfg, v, arm, seed):
    """Rows of one arm on one world of 4 episodes, played as the runner plays it."""
    from hypergraph_agent.envs.generator import derive_seed
    from hypergraph_agent.techtree import study as S
    from hypergraph_agent.techtree.config import merged
    from hypergraph_agent.training.budget import BudgetMeter, SessionLedger
    scfg, reveal = stream_config(cfg, v, seed)
    st = TechStream(replace(scfg, n_worlds=1, episodes_per_world=4))
    tasks = [st.task(w, e) for w, e in st.order()]
    study = cfg["run"]["study"]
    kind = "reporting" if arm in S.reporting_arms(study) else "adaptive"
    m = BudgetMeter(SessionLedger(None, {"dev": 10 ** 9}, 10 ** 9, 10 ** 9), "golden", "dev",
                    4 * scfg.episode_budget, kind=kind)
    agent_rng = np.random.default_rng(derive_seed("techtree_agent", study, seed) % 2 ** 32)
    agent = None if arm == "reference" else S.make_agent(study, arm, agent_rng, cfg["agent"])
    w, s = merged(cfg, v)
    extra = {"arm": arm, "variant": v["name"], "seed": seed, **S.factors(w, s)}
    rows = S.play(agent, arm, tasks, [reference_solve(t) for t in tasks], m, None, extra, reveal)
    return [{k: x for k, x in r.items() if k != "episode_s"} for r in rows]


@pytest.mark.parametrize("rel", sorted(FROZEN_TT_CONFIGS))
def test_frozen_techtree_configs_load_the_same_and_every_arm_acts_as_before(rel):
    cfg = load_config(ROOT / rel)
    loaded, acted = FROZEN_TT_CONFIGS[rel]
    assert _sha({k: x for k, x in cfg.items() if k != "_source"}) == loaded
    assert "latent" not in cfg["world"] and all("latent" not in v.get("world", {}) for v in variants(cfg))
    seed = cfg["run"]["seeds"][0]
    cases = {f"{rel}:{v['name']}:{arm}": _sha(_frozen_agent_rows(cfg, v, arm, seed))
             for v in variants(cfg) for arm in variant_arms(cfg, v)}
    assert _sha(cases) == acted


def test_latent_off_keeps_study_x_streams_and_the_benchmark_families():
    from hypergraph_agent.baselines.sources import evaluation_source
    from hypergraph_agent.benchmarks import stats
    from hypergraph_agent.envs.generator import derive_seed
    for name, digest in FROZEN_X.items():
        x = yaml.safe_load((ROOT / "configs" / "baselines" / name).read_text())
        sp, out = x["test"], []
        for seed in sp["seeds"]:
            src = evaluation_source("techtree", x["setting"], sp["namespace"], sp["base_seed"], seed,
                                    sp["n_worlds"], sp["episodes_per_world"])
            while (block := src.next_block()) is not None:
                out += [[seed, it.task.task_key, it.world_key, it.task.goal, _world_fields(it.task.world)]
                        for it in block]
        assert _digest(out) == digest, name
    for fam, wcfg in stats.TECH_FAMILIES.items():
        ws = [_world_fields(make_world(derive_seed("benchmark_family", fam, i), wcfg)) for i in range(50)]
        assert _digest(ws) == FROZEN_BENCH[fam], fam


def test_no_frozen_config_mentions_the_latent_option():
    for d in ("unlock", "layers", "unlock2", "skills", "policygraph_tt", "baselines"):
        for p in (ROOT / "configs" / d).glob("*.yaml"):
            assert "latent" not in p.read_text(), p


# ------------------------------------------------------------------- latent worlds
def test_latent_worlds_hide_shared_fragments_in_compositions():
    for seed in range(40):
        w = make_world(seed, LATENT)
        combos = {w.sequences[c] for c in w.at(1)}
        assert len(w.fragments) == 2 and len(set(w.fragments)) == 2
        assert not set(w.fragments) & set(w.sequences) and not set(w.fragments) & combos
        assert all(len(f) == 2 and max(f) < 3 for f in w.fragments)
        uses = [j for j in w.fragment_of if j >= 0]
        assert sorted(set(uses)) == [0, 1] and all(uses.count(j) >= 2 for j in (0, 1))
        for x, lv in enumerate(w.levels):
            if lv == 1:
                assert w.fragment_of[x] == -1
                continue
            frag, (p0, p1) = w.fragments[w.fragment_of[x]], w.parents[x]
            spelled = {w.sequences[p0] + frag, frag + w.sequences[p1]}
            assert w.sequences[x] in spelled and w.compositional[x]
            if lv == 3:  # the kept parent is the level-2 one, in its concatenation position
                kept = p0 if w.levels[p0] == 2 else p1
                assert w.sequences[x] == (w.sequences[p0] + frag if kept == p0 else frag + w.sequences[p1])


def test_latent_worlds_are_coupled_with_plain_worlds_and_across_depths():
    coupled = 0
    for seed in range(30):
        plain = make_world(seed, replace(L_WORLD, reuse_depth=2))
        d1, d2 = (make_world(seed, replace(LATENT, reuse_depth=d)) for d in (1, 2))
        same = (plain.keys == d2.keys and plain.parents == d2.parents and plain.attempt == d2.attempt
                and [plain.sequences[c] for c in plain.at(1)] == [d2.sequences[c] for c in d2.at(1)])
        coupled += same
        if d1.attempt == d2.attempt:
            assert d1.fragments == d2.fragments
            assert [d1.sequences[c] for c in d1.at(2)] == [d2.sequences[c] for c in d2.at(2)]
            assert all(d1.fragment_of[c] == -1 and not d1.compositional[c] for c in d1.at(3))
    assert coupled >= 28


def test_flat_concepts_of_latent_worlds_are_no_two_piece_concatenation():
    for seed in range(30):
        w = make_world(seed, replace(LATENT, reuse_depth=1))
        pieces = list(w.sequences) + list(w.fragments)
        for x in w.at(3):
            s = w.sequences[x]
            assert all(s != a + b for i, a in enumerate(pieces) for j, b in enumerate(pieces) if i != j)


def test_latent_option_is_validated():
    for bad in (dict(length_rule="doubling"), dict(reuse_depth=0), dict(latent=4), dict(latent=-1),
                dict(latent=True)):
        with pytest.raises(ValueError):
            TechWorldConfig(**{**dict(reuse_depth=2, latent=2), **bad})
    with pytest.raises(ValueError):
        TechWorldConfig(n_primitives=3, n_slots=2, n_unlocks=1, n_levels=2, concepts_per_level=(3, 3),
                        reuse_depth=1, latent=1)
    assert TechWorldConfig(latent=0) == TechWorldConfig()


def test_reference_is_exact_on_latent_worlds_and_the_public_view_hides_the_option():
    small = TechWorldConfig(n_primitives=2, n_levels=2, concepts_per_level=(2, 2), reuse_depth=1, latent=1)
    for seed in range(4):
        st = stream(small, seed=seed, n_worlds=1, epw=2, goal_levels=(2,))
        for e in range(2):
            t = replace(st.task(0, e), goal=st.world(0).at(2)[e % 2])
            ref = reference_solve(t)
            assert ref.exact and ref.length == _brute_force(t)
            env = TechTreeEnv()
            env.reset(options={"task": t})
            for a in ref.plan:
                _, r, term, _, _ = env.step(a)
            assert term and r == 1
    a, b = stream(replace(L_WORLD, reuse_depth=2)).task(0, 0), stream(LATENT).task(0, 0)
    assert a.world.sequences != b.world.sequences and public_view(a) == public_view(b)
    assert "latent fragment 0" in debug_view(b.world) and "latent" not in debug_view(a.world)


def test_config_accepts_latent_only_as_an_optional_world_key(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("run: {name: x, study: CA}\nvariants: [{name: a, world: {reuse_depth: 2, latent: 2}}]\n")
    cfg = load_config(p)
    assert "latent" not in cfg["world"] and stream_config(cfg, variants(cfg)[0], 0)[0].world.latent == 2
    p.write_text("run: {name: x, study: CA}\nworld: {latnet: 2}\n")
    with pytest.raises(KeyError):
        load_config(p)
    from hypergraph_agent.techtree.study import STUDY_AGENTS
    assert STUDY_AGENTS["CA"] == ("compression", "make_compression")


# --------------------------------------------------------------------- compressors
def test_chains_parse_and_inline_earlier_abstractions():
    assert encode((1, 0, 2)) == "(a1 (a0 (a2 e)))"
    assert parse_sexpr("(a1 (a0 e))") == ["a1", ["a0", "e"]]
    bodies = {"fn_0": "(a1 (a2 #0))", "fn_1": "(a0 (fn_0 #0))", "fn_2": "(fn_1 (a2 e))", "fn_3": "(a1 e)",
              "fn_4": "(a2 fn_3)", "fn_5": "(#0 (a1 e))"}
    expect = {"fn_0": (1, 2), "fn_1": (0, 1, 2), "fn_2": (0, 1, 2, 2), "fn_3": (1,), "fn_4": (2, 1)}
    for name, syms in expect.items():
        assert chain_symbols(parse_sexpr(bodies[name]), bodies) == syms
    assert chain_symbols(parse_sexpr(bodies["fn_5"]), bodies) is None


@needs_stitch
def test_stitch_finds_a_planted_fragment_and_is_deterministic():
    noise = [(0, 1, 2, 2, 1), (2, 0, 0, 1, 2), (1, 1, 0, 2, 0), (2, 2, 0, 1, 1)]
    planted = [(1, 0, 2, 0, 0), (2, 1, 0, 2, 0), (0, 2, 1, 0, 2, 0), (1, 0, 2, 0, 1)]
    frags = stitch_fragments(noise + planted, 50, 6)
    assert (1, 0, 2, 0) in frags and all(2 <= len(f) <= 6 for f in frags)
    assert stitch_fragments(noise + planted, 50, 6) == frags
    assert stitch_fragments([(0, 1)], 50, 6) == [] and stitch_fragments([(0, 1), (0, 1)], 1, 6) == [(0, 1)]
    from hypergraph_agent.techtree.compression import stitch_library
    found, effort = stitch_library(noise + planted, 50, 6)
    assert found == frags and len(frags) <= effort <= 50  # effort: abstractions before one-press drops
    assert stitch_library(noise + planted, 2, 6)[1] == 2  # the cap binds


def test_bpe_merges_the_most_frequent_pair_until_no_pair_repeats():
    corpus = [(0, 1, 0, 1, 2), (0, 1, 2), (2, 2)]
    assert bpe_fragments(corpus, 50, 6) == [(0, 1), (0, 1, 2)]
    assert bpe_fragments(corpus, 1, 6) == [(0, 1)]
    assert bpe_fragments([(1, 0), (0, 1)], 50, 6) == []  # every pair occurs once
    assert bpe_fragments([(1, 0), (1, 0), (0, 1), (0, 1)], 1, 6) == [(0, 1)]  # ties: the smaller pair
    assert bpe_fragments([(0, 0, 0, 0)] * 2, 50, 3) == [(0, 0)]  # longer tokens than max_len dropped
    from hypergraph_agent.techtree.compression import bpe_library
    assert bpe_library(corpus, 50, 6) == ([(0, 1), (0, 1, 2)], 2) and bpe_library(corpus, 1, 6)[1] == 1


def test_description_lengths():
    table = np.array([[0, 1, 2, 2], [1, 1, 1, 1], [2, 0, 1, 0]])
    toks = [(0, 1), (2, 2), (1, 1, 1), (0, 1, 2, 2)]
    assert list(description_lengths(table, [])) == [4, 4, 4]
    assert list(description_lengths(table, toks)) == [1, 2, 3]
    assert list(description_lengths(table, [(0, 1), (2, 2)])) == [2, 4, 3]
    assert min_tokens((2, 0, 1, 0), [(0, 1), (1, 0)]) == 3 and min_tokens((5,), []) == 1


def test_spearman_and_ranks():
    assert spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 3], [1, 2, 3, 4]) == pytest.approx(0.9486833, abs=1e-6)
    assert spearman([1, 2], [1, 2]) is None and spearman([1, 1, 1], [1, 2, 3]) is None


# ------------------------------------------------------------------------- the arms
def test_factory_parameters_and_reporting_arms():
    for arm in ("promote_stitch", "promote_bpe", "enum_prim", "enum_atoms", "enum_frag", "oracle_library",
                "pooled", "remember", "promote", "sham_promote", "random"):
        make_compression(arm, rng())
    with pytest.raises(KeyError):
        make_compression("promote_zip", rng())
    assert check_params(None) == DEFAULT_PARAMS and set(CORPORA) == {"firings", "tests", "episodes"}
    for bad in ({"corpus": "raw"}, {"stitch_iterations": 0}, {"bpe_merges": True}):
        with pytest.raises(ValueError):
            check_params(bad)
    with pytest.raises(KeyError):
        check_params({"iterations": 3})
    from hypergraph_agent.techtree.study import reporting_arms
    assert reporting_arms("CA") == ("random", "oracle", "oracle_library", "reference")


@pytest.mark.parametrize("world", [replace(L_WORLD, reuse_depth=0), replace(L_WORLD, reuse_depth=2)],
                         ids=["d0", "d2"])
def test_layer_zero_enumerator_and_pooled_are_remember(world):
    st = stream(world, n_worlds=2, epw=4)
    rows = {arm: [(r["primitive_length"], r["fired"]) for r in play(arm, st)[1]]
            for arm in ("remember", "pooled", "enum_prim")}
    assert rows["remember"] == rows["pooled"] == rows["enum_prim"]


@needs_stitch
@pytest.mark.parametrize("arm", ["promote_stitch", "promote_bpe", "enum_atoms", "enum_frag"])
@pytest.mark.parametrize("world", [replace(L_WORLD, reuse_depth=0), LATENT], ids=["d0", "latent_d2"])
def test_compression_arms_keep_sound_evidence(arm, world):
    agent, rows, tasks = play(arm, stream(world, n_worlds=1, epw=4))
    _soundness(agent, tasks)
    assert all(r["primitive_length"] <= 100 for r in rows)


@needs_stitch
def test_firings_corpus_holds_what_fired_and_failed_replays():
    agent, rows, tasks = play("promote_stitch", stream(LATENT, n_worlds=1, epw=6), {"corpus": "firings"})
    w = tasks[0].world
    corpus = agent.corpus[w.world_key]
    fired = sum(len(r["fired"]) for r in rows)
    successes = sum(r["ca_corpus_success"] for r in rows)
    assert successes == fired and len(corpus) == successes + sum(r["ca_corpus_failure"] for r in rows)
    assert set(corpus) <= set(w.sequences)  # firings and failed replays are concepts' sequences
    assert rows[0]["ca_library"] == [] and rows[-1]["ca_corpus_size"] == len(corpus)


@needs_stitch
def test_tests_and_episodes_corpora():
    st = stream(LATENT, n_worlds=1, epw=3)
    a_tests, r_tests, tasks = play("promote_stitch", st, {"corpus": "tests"})
    a_fired, r_fired, _ = play("promote_stitch", st, {"corpus": "firings"})
    wk = tasks[0].world.world_key
    assert sum(r["ca_corpus_failure"] for r in r_tests) > sum(r["ca_corpus_failure"] for r in r_fired)
    assert len(a_tests.corpus[wk]) > len(a_fired.corpus[wk])
    k, concepts = a_tests.worlds[wk], set(tasks[0].world.sequences)
    failed = [s for s in a_tests.corpus[wk] if s not in concepts]  # level = length / 2 in these worlds
    assert failed and all(k.elim[len(s) // 2][k.code(s)] for s in failed)
    a_ep, r_ep, _ = play("promote_stitch", st, {"corpus": "episodes"})
    assert len(a_ep.corpus[wk]) == len(r_ep)  # one press sequence per episode (no wait pressed)
    assert [len(s) for s in a_ep.corpus[wk]] == [r["primitive_length"] - r["success"] for r in r_ep]


@needs_stitch
def test_library_changes_only_between_episodes_and_sets_the_macro_candidates():
    agent, rows, tasks = play("promote_stitch", stream(replace(L_WORLD, reuse_depth=2), n_worlds=1, epw=6))
    wk = tasks[0].world.world_key
    lib = [tuple(f) for f in rows[-1]["ca_library"]]
    assert lib == agent.fragments() and agent.versions[wk] <= len(rows)
    k = agent.worlds[wk]
    for level in (2, 3):
        n = k.lengths[level - 1]
        want = {a + b for a in lib for b in lib if a != b and len(a) + len(b) == n}
        got = {k.digits(level, int(c)) for c in np.flatnonzero(agent.pair_mask(k, level))}
        assert got == want
    assert all(r["ca_compress_cap"] == 50 and 0 <= r["ca_compress_effort"] <= 50 for r in rows)
    assert rows[0]["ca_compress_effort"] == 0 and rows[-1]["ca_compress_effort"] >= len(lib)


def test_enumerator_tiers_follow_description_length():
    st = stream(replace(L_WORLD, reuse_depth=2), n_worlds=1, epw=3)
    agent, rows, tasks = play("enum_atoms", st)
    k = agent.worlds[tasks[0].world.world_key]
    dl = agent.dl(k, 3)
    expect = description_lengths(k.table[3], [s for s in k.known.values() if 2 <= len(s) <= 6])
    assert (dl == expect).all() and dl.max() == 6 and dl.min() >= 1
    assert all(r["ca_layer"] == 1 for r in rows)


@needs_stitch
def test_oracle_reports_description_lengths_and_acts_as_study_ls_oracle():
    from hypergraph_agent.techtree.layered import make_layered
    st = stream(LATENT, n_worlds=2, epw=3)
    _, rows, tasks = play("oracle_library", st)
    env, m, plain = TechTreeEnv(), meter(1_000_000), make_layered("oracle_library", rng())
    for r, t in zip(rows, tasks):
        plain.reveal(t.world.world_key, revealed_links(t.world))
        other = run(plain, env, t, m)
        assert {key: r[key] for key in other} == other
        w = t.world
        dl0, dl1, dl2, dl_residue = r["ca_goal_dl"]
        lower = [s for c, s in enumerate(w.sequences) if w.levels[c] < w.levels[t.goal]]
        assert dl0 == 6 and dl1 == min_tokens(w.sequences[t.goal], lower) == 3 and 2 <= dl2 <= dl1
        assert dl_residue == min_tokens(w.sequences[t.goal], set(lower) | set(w.fragments)) == 2
        assert sorted(tuple(s) for _, s in r["ca_world_sequences"]) == sorted(w.sequences)
        assert r["ca_world_dl"]["3"][0] == 6 and r["ca_world_dl"]["2"][1] == 3
        assert r["ca_world_dl"]["3"][3] == 2 and r["ca_compress_errors"] == 0


def test_parent_residues_are_the_hidden_fragments_at_depth_two():
    from hypergraph_agent.techtree.compression import parent_residues
    for seed in range(30):
        w = make_world(seed, LATENT)
        assert parent_residues(dict(enumerate(w.sequences)), dict(enumerate(w.parents))) == set(w.fragments)
        plain = make_world(seed, replace(L_WORLD, reuse_depth=2))  # compositions leave concepts only
        assert not parent_residues(dict(enumerate(plain.sequences)), dict(enumerate(plain.parents)))
    seqs = {0: (0, 1), 1: (1, 2), 2: (0, 1, 2, 2), 3: (2, 2, 1, 2)}  # 2 keeps 0, 3 keeps 1
    assert parent_residues(seqs, {0: (), 1: (), 2: (0, 1), 3: (0, 1)}) == {(2, 2)}


# --------------------------------------------------------------- information boundary
def test_compression_module_passes_the_source_check():
    src = (TT / "compression.py").read_text()
    assert not boundary_violations(src)
    top = [a.name for n in ast.parse(src).body if isinstance(n, ast.Import) for a in n.names]
    assert "stitch_core" not in top  # imported where used, so the package imports without it
    # layered is itself an agent module under the same rule; stitch_core is the compressor under test
    # (the justification is in docs/studies/CA_compression_atoms.md, Implementation)
    allowed = ALLOWED_IMPORTS | {"layered", "stitch_core"}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.Import):
            mods = [a.name.split(".")[0] for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            mods = [(node.module or "").split(".")[0]]
        else:
            continue
        assert set(mods) <= allowed, mods


def test_compression_arms_never_read_hidden_world_fields():
    """No attribute named like a hidden field of a TechWorld is read from anything but ``self``."""
    tree = ast.parse((TT / "compression.py").read_text())
    attrs = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)
             and not (isinstance(n.value, ast.Name) and n.value.id == "self")}
    hidden = {"fragments", "fragment_of", "compositional", "sequences", "parents", "unlocks", "entries"}
    assert not attrs & hidden, attrs & hidden


# --------------------------------------------------------------------------- configs
def ledger_block(path):
    raw = yaml.safe_load(path.read_text())
    led = raw["ledger"]
    return led["allocations"], led["adaptive_cap"], led["reporting_cap"]


def test_ca_configs_share_one_ledger_and_cover_their_allocations():
    from hypergraph_agent.techtree.study import plan
    assert [p.name for p in CONFIGS] == ["ca_dev.yaml", "ca_dev_tests.yaml", "ca_main.yaml"]
    assert len({repr(ledger_block(p)) for p in CONFIGS}) == 1
    alloc, adaptive_cap, reporting_cap = ledger_block(CONFIGS[0])
    assert set(alloc) == {"dev", "ca_main"} and adaptive_cap == sum(alloc.values())
    totals = {p.name: plan(load_config(p))["caps"] for p in CONFIGS}
    assert totals["ca_main.yaml"]["adaptive_total"] == alloc["ca_main"]
    assert all(totals[n]["adaptive_total"] <= alloc["dev"] for n in ("ca_dev.yaml", "ca_dev_tests.yaml"))
    assert sum(t["reporting_total"] for t in totals.values()) <= reporting_cap
    for p in CONFIGS:
        cfg = load_config(p)
        assert cfg["run"]["study"] == "CA" and cfg["run"]["ledger"] == "runs/CA-ledger.json"
        for v in variants(cfg):
            for arm in variant_arms(cfg, v):
                if arm != "reference":
                    make_compression(arm, rng(), cfg["agent"])
        namespaces = {s["namespace"] for s in plan(cfg)["variants"]}
        if "dev" in p.name:
            assert cfg["run"]["allocation"] == "dev" and namespaces == {"ca_dev"}
            assert not cfg["run"]["requires_flag"]
        else:
            assert cfg["run"]["requires_flag"] and cfg["run"]["allocation"] == "ca_main"
            assert namespaces == {"techtree_ca"}


def test_development_passes_mirror_the_measurement_config():
    dev, dev_tests, main = (yaml.safe_load(p.read_text()) for p in CONFIGS)
    # first pass: every setting and arm, with the corpus of firings only
    for key in ("ledger", "world", "variants", "arms"):
        assert dev[key] == main[key], key
    assert dev["agent"] == {**main["agent"], "corpus": "firings"} and main["agent"]["corpus"] == "tests"
    assert dev["analysis"]["tests"] == main["analysis"]["tests"] and dev["analysis"]["verdict"] is None
    assert dev["stream"] == {**main["stream"], "namespace": "ca_dev"} == dev_tests["stream"]
    # second pass: the measurement's corpus, the arms whose actions depend on it, the same worlds
    for key in ("ledger", "agent", "world"):
        assert dev_tests[key] == main[key], key
    assert dev_tests["arms"] == ["promote_stitch", "promote_bpe", "enum_frag"]
    settings = [(v["name"], v["world"]) for v in main["variants"]]
    assert [(v["name"], v["world"]) for v in dev_tests["variants"]] == settings
    assert dev_tests["run"]["seeds"] == dev["run"]["seeds"]


@needs_stitch
@pytest.mark.parametrize("arm", ["remember", "promote", "sham_promote", "enum_atoms", "oracle_library"])
def test_arms_without_compression_ignore_the_corpus(arm):
    st = stream(LATENT, n_worlds=1, epw=4)
    keep = ("primitive_length", "fired", "success", "search_steps", "macro_steps")
    rows = [[{k: r[k] for k in keep} for r in play(arm, st, {"corpus": c})[1]] for c in CORPORA]
    assert rows[0] == rows[1] == rows[2]


def test_measurement_config_needs_the_flag():
    from hypergraph_agent.techtree.__main__ import main
    p = ROOT / "configs" / "compression" / "ca_main.yaml"
    assert main(["--config", str(p), "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        main(["--config", str(p)])


def test_ca_namespaces_are_fresh():
    others = [p for p in (ROOT / "configs").rglob("*.yaml") if p.parent.name != "compression"]
    for ns in ("techtree_ca", "ca_dev"):
        assert not any(re.search(rf"\b{ns}\b", p.read_text()) for p in others), ns


# ------------------------------------------------------------- runner and analysis
@needs_stitch
def test_runner_and_analysis_end_to_end(tmp_path):
    from hypergraph_agent.techtree.analysis import analyse, collect_runs
    from hypergraph_agent.techtree.study import run_study
    from hypergraph_agent.training.budget import SessionLedger
    p = tmp_path / "ca.yaml"
    p.write_text(f"""
run: {{name: ca-e2e, study: CA, seeds: [0], runs_dir: {(tmp_path / 'runs').as_posix()}}}
agent: {{corpus: firings, stitch_iterations: 50, bpe_merges: 50}}
stream: {{namespace: unit_dev, n_worlds: 1, episodes_per_world: 4, goal_levels: [3], episode_budget: 100}}
variants:
  - {{name: d2, world: {{reuse_depth: 2}}}}
  - {{name: latent_d2, world: {{reuse_depth: 2, latent: 2}}}}
arms: [remember, promote, promote_stitch, promote_bpe, enum_atoms, enum_frag, oracle_library, reference]
analysis:
  late_from: 2
  tests:
    - {{type: paired, name: p, metric: steps_late, a: promote_stitch, b: promote, variant: latent_d2,
       expect: less}}
    - {{type: all_success, name: ref_ok, arm: reference}}
    - {{type: all_success, name: oracle_ok, arm: oracle_library}}
  verdict: {{rule: all, require: [ref_ok, oracle_ok]}}
""")
    cfg = load_config(p)
    ledger = SessionLedger(None, {"dev": 100_000}, 100_000, 100_000)
    assert len(run_study(cfg, ledger, str(tmp_path / "runs"))) == 16
    runs = collect_runs(tmp_path / "runs", "ca-e2e")
    rows = [row for r in runs for row in r["rows"]]
    reporting = sum(r["primitive_length"] for r in rows if r["arm"] in ("oracle_library", "reference"))
    assert ledger.state["reporting_total"] == reporting
    assert ledger.state["adaptive_total"] == sum(r["primitive_length"] for r in rows) - reporting
    res = analyse(cfg, runs)
    assert not res["incomplete_cells"] and res["verdict"]["verdict"] == "pass"
    layers, libs = res["row_summaries"]["layers"], res["row_summaries"]["libraries"]
    assert set(layers["world_dl"]) == {"d2", "latent_d2"} and layers["world_dl"]["d2"]["3"][1] == 2.0
    assert set(layers["learners"]["latent_d2"]) == {"remember", "enum_atoms", "enum_frag"}
    assert {p_["layer"] for p_ in layers["points"]} == {0, 1, 2} and "pooled" in layers["spearman"]
    assert set(libs["d2"]) == {"promote_stitch", "promote_bpe", "enum_atoms", "enum_frag"}
    assert libs["latent_d2"]["enum_atoms"]["size"] == 0


def test_row_summaries_on_hand_made_rows():
    base = {"variant": "v", "seed": 0, "reference_length": 7, "fired": []}
    rows = []
    for wk, dl, costs in (("W1", 2.0, [50, 10, 10, 10]), ("W2", 3.0, [100, 60, 10, 10])):
        world = {"2": [4.0, 2.0, 2.0], "3": [6.0, dl, 2.0]}
        for e, n in enumerate(costs):
            rows.append({**base, "arm": "oracle_library", "world_key": wk, "world_episode": e,
                         "primitive_length": 7, "success": True, "ca_world_dl": world,
                         "ca_world_sequences": [[1, [0, 1]], [2, [0, 1, 2, 2]], [3, [0, 1, 2, 2, 1, 1]]]})
            rows.append({**base, "arm": "enum_atoms", "world_key": wk, "world_episode": e,
                         "primitive_length": n, "success": n < 100, "ca_library": [[0, 1], [2, 2], [1, 0]],
                         "ca_corpus_size": 9, "ca_compress_errors": 0})
    out = layer_summary(rows)
    assert out["world_dl"]["v"]["3"] == [6.0, 2.5, 2.0]
    learner = out["learners"]["v"]["enum_atoms"]
    assert learner["dl"] == 2.5 and learner["first_success_steps"] == (50 + 160) / 2
    assert learner["steps_late"] == 10.0 and out["spearman"]["enum_atoms"]["n"] == 2
    lib = library_summary(rows)["v"]["enum_atoms"]
    assert lib["size"] == 3 and lib["concept_share"] == pytest.approx(1 / 3)
    assert lib["shared_share"] == pytest.approx(1 / 3) and lib["other_share"] == pytest.approx(1 / 3)
    for r in rows:
        if r["arm"] == "enum_atoms":
            r.update({"ca_compress_effort": 50 if r["world_episode"] % 2 else 7, "ca_compress_cap": 50})
    assert library_summary(rows)["v"]["enum_atoms"]["cap_share"] == pytest.approx(0.5)


# ------------------------------------------------------------------------- hygiene
def test_ca_files_have_short_lines_no_dashes_and_no_local_paths():
    texts = {f: f.read_text(encoding="utf-8") for f in FILES + ([DOC] if DOC.exists() else [])}
    long = [f"{f.name}:{n}" for f, text in texts.items()
            for n, line in enumerate(text.splitlines(), 1) if len(line) > 110]
    assert not long, long
    dashes = (chr(0x2014), chr(0x2013), " -" + "- ")  # em dash, en dash, spaced double hyphen
    assert not [f.name for f, text in texts.items() if any(d in text for d in dashes)]
    local = re.compile(r"\b[A-Za-z]:[\\/]|/" + "Users/|/" + "home/")  # drive letters and home folders
    assert not [f.name for f, text in texts.items() if local.search(text)]
