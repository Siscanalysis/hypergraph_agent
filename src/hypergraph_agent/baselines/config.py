"""Strict YAML configs of study X (``configs/baselines/``).

A config describes one setting (an environment family and its cell), the
learners trained on its training namespace, the repository's agents, the
evaluation streams and the analysis. Files are merged onto ``DEFAULTS``; unknown
keys raise. ``setting.world``, ``setting.tasks``, ``setting.stream``,
``agent_params`` and a learner's ``params`` and ``rnd`` are passed to the
generator, the agents or the library and validated there.

Namespace rule: the development allocations (``DEV_ALLOCATIONS``) may only use
namespaces ending in ``_dev``; any other combination needs
``run.requires_flag: true``, and the CLI then refuses to run without
``--allow-measurement``. So development and tuning never touch measurement
worlds or allocations.

Learner entries: ``id`` (unique), ``group`` (the learner as analysed, by
default the algorithm; tuning candidates of one learner share it), ``algo``
(``learners.ALGOS``), ``params`` (library keyword arguments; absent ones keep the
library defaults), ``trial`` (one learner episode per world block),
``rnd`` (null, or the RND reward wrapper's settings), ``step_penalty`` (reward
shaping, default 0) and ``modes`` (action selection at evaluation,
``deterministic`` and/or ``stochastic``).

Evaluation-only configs: with ``run.checkpoints_from`` (the path of another
config) a config trains nothing; it takes its learners and training block from
that config and evaluates the checkpoints of that config's completed training
runs on its own test stream (``test.budgets``: the checkpoint budgets to
evaluate, default every training budget), next to the repository's agents. Its
best learner is the one the other config selects. ``tuning_record`` is
inherited. Study X's benchmark test sets are such configs, with the external
loader ``baselines.benchmark_sets`` and its parameters in ``setting.benchmark``.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from ..envs.generator import stable_hash
from . import benchmark_sets
from .learners import ALGOS, RND_DEFAULTS

DEV_ALLOCATIONS = ("dev", "x_tune")
FAMILIES = ("recipequest", "techtree")
MODES = ("deterministic", "stochastic")
RECIPEQUEST_AGENTS = ("focused_sample", "maximal", "reference", "random")
TECHTREE_AGENTS = {"U": ("pooled", "isolated", "oracle", "reference", "random"),
                   "L": ("remember", "promote", "sham_promote", "adaptive", "none", "oracle_library",
                         "reference", "random")}

DEFAULTS: dict = {
    # tuning_from: the tuning config of a measurement config; tuning_map: tuning group -> learner id of
    # this config; tuning_record: {summary, sha256} written by ``apply-tuning`` (``tuning.py``); a
    # measurement config without a record that reproduces its learners refuses to run
    "run": {"name": "x-dev", "study": "X", "allocation": "dev", "ledger": "runs/X-ledger.json",
            "runs_dir": "runs", "requires_flag": False, "tuning_record": None, "checkpoints_from": None,
            "tuning_from": None, "tuning_map": None},
    # caps written into a NEW ledger file; an existing ledger keeps its own
    "ledger": {"allocations": {"dev": 300_000}, "adaptive_cap": 300_000, "reporting_cap": 300_000},
    # loader: "package.module:function" for an external benchmark (sources.py); max_variants: craft
    # slots per item type in the RecipeQuest layout
    "setting": {"name": "rq_f1", "family": "recipequest", "world": {}, "tasks": {}, "stream": {},
                "agent_study": None, "loader": None, "max_variants": 2, "benchmark": {}},
    "train": {"namespace": "x_rq_train_dev", "base_seed": 30_000, "seeds": [0], "budgets": [20_000],
              "n_envs": 8, "episodes_per_world": 16},
    "test": {"namespace": "x_rq_eval_dev", "base_seed": 10_000, "seeds": [0], "n_worlds": 2,
             "episodes_per_world": 16, "budgets": None},
    "validation": {"namespace": None, "base_seed": 10_000, "seeds": [], "n_worlds": 0,
                   "episodes_per_world": 16},
    "learners": [],
    "agents": [],
    "agent_params": {},
    "analysis": {"late_from": 8, "n_boot": 10_000, "boot_seed": 20261008, "select_metric": "cost_ratio_late",
                 "select_split": "validation", "tests": []},
}
LEARNER_DEFAULTS = {"id": None, "group": None, "algo": None, "params": {}, "trial": False, "rnd": None,
                    "step_penalty": 0.0, "modes": ["deterministic"]}
TEST_KEYS = {"name", "role", "a", "b", "metric", "expect", "budget", "split", "level"}
# per-world metrics (analysis.world_metrics); lower is better except for the success rates
METRICS = ("cost_ratio_late", "cost_ratio_early", "cost_ratio_all", "cost_ratio_late_minus_early",
           "steps_late", "steps_early", "steps_all", "success_late", "success_early", "success_all",
           "first_success_steps")
EXPECT = ("less", "greater", "not_less", "not_greater")
FREE = ("ledger.allocations", "setting.world", "setting.tasks", "setting.stream", "setting.benchmark",
        "agent_params")


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        where = f"{path}.{k}" if path else k
        if k not in base and path not in FREE:
            raise KeyError(f"unknown config key {where}")
        if isinstance(v, dict) and isinstance(base.get(k), dict) and path not in FREE:
            out[k] = _merge(base[k], v, where)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _learner(raw: dict) -> dict:
    extra = set(raw) - set(LEARNER_DEFAULTS)
    if extra:
        raise KeyError(f"unknown learner keys {sorted(extra)}")
    ln = {**copy.deepcopy(LEARNER_DEFAULTS), **copy.deepcopy(raw)}
    if not ln["id"] or ln["algo"] not in ALGOS:
        raise ValueError(f"a learner needs an id and an algorithm in {sorted(ALGOS)}: {raw}")
    ln["group"] = ln["group"] or ln["algo"]
    if ln["rnd"] is not None:
        bad = set(ln["rnd"]) - set(RND_DEFAULTS)
        if bad:
            raise KeyError(f"unknown rnd keys {sorted(bad)} in learner {ln['id']!r}")
    if not ln["modes"] or set(ln["modes"]) - set(MODES):
        raise ValueError(f"learner {ln['id']!r}: modes must be a nonempty subset of {MODES}")
    return ln


def namespaces(cfg: dict) -> list[str]:
    out = [cfg["train"]["namespace"], cfg["test"]["namespace"]]
    if cfg["validation"]["namespace"]:
        out.append(cfg["validation"]["namespace"])
    return out


def config_path(config: str | Path, ref: str | Path) -> Path:
    """A config path named inside a config (``checkpoints_from``, ``tuning_from``): a relative one is
    taken from the root of the tree holding ``config`` (``<root>/configs/baselines/x.yaml``), so a
    snapshot's configs refer to the snapshot's own files; else from the working directory."""
    ref = Path(ref)
    if not ref.is_absolute():
        parents = Path(config).resolve().parents
        if len(parents) > 2 and (parents[2] / ref).exists():
            return parents[2] / ref
    return ref


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    learners = raw.pop("learners", [])
    cfg = _merge(DEFAULTS, raw)
    cfg["learners"] = [_learner(x) for x in learners]
    if cfg["run"]["checkpoints_from"]:
        if learners or "train" in raw:
            raise ValueError("an evaluation-only config takes learners and train from checkpoints_from")
        cfg["_source_path"] = str(config_path(path, cfg["run"]["checkpoints_from"]))
        src = load_config(cfg["_source_path"])
        if src["run"]["checkpoints_from"]:
            raise ValueError("checkpoints_from must name a config that trains")
        if (src["setting"]["family"], src["setting"]["max_variants"]) != \
                (cfg["setting"]["family"], cfg["setting"]["max_variants"]):
            raise ValueError("checkpoints_from must have the same family and layout")
        cfg["learners"], cfg["train"] = copy.deepcopy(src["learners"]), copy.deepcopy(src["train"])
        cfg["_train_setting"], cfg["_source_name"] = src["setting"], src["run"]["name"]
        cfg["run"]["tuning_record"] = cfg["run"]["tuning_record"] or src["run"]["tuning_record"]
    if bool(cfg["run"]["tuning_from"]) != bool(cfg["run"]["tuning_map"]):
        raise ValueError("run.tuning_from and run.tuning_map go together")
    tmap = cfg["run"]["tuning_map"]
    if tmap and (len(set(tmap.values())) != len(tmap)
                 or set(tmap.values()) != {x["id"] for x in cfg["learners"]}):
        raise ValueError("run.tuning_map must map tuning groups one to one onto this config's learner ids")
    if cfg["test"]["budgets"] is not None and not set(cfg["test"]["budgets"]) <= set(cfg["train"]["budgets"]):
        raise ValueError("test.budgets must be training budgets")
    if cfg["setting"]["loader"] == benchmark_sets.LOADER:
        bench = benchmark_sets.params(cfg["setting"])
        if cfg["test"]["n_worlds"] != bench["n_worlds"] or cfg["test"]["seeds"] != [0]:
            raise ValueError("a benchmark test set plays its n_worlds worlds once: test.n_worlds must equal "
                             "setting.benchmark.n_worlds and test.seeds must be [0]")
    ids = [x["id"] for x in cfg["learners"]]
    if len(ids) != len(set(ids)):
        raise ValueError("learner ids must be unique")
    s = cfg["setting"]
    if s["family"] not in FAMILIES:
        raise ValueError(f"setting.family must be one of {FAMILIES}")
    if s["family"] == "recipequest":
        if set(cfg["agents"]) - set(RECIPEQUEST_AGENTS):
            raise ValueError(f"RecipeQuest agents are {RECIPEQUEST_AGENTS}")
    else:
        if s["agent_study"] not in TECHTREE_AGENTS:
            raise ValueError(f"a TechTree setting needs agent_study in {sorted(TECHTREE_AGENTS)}")
        if set(cfg["agents"]) - set(TECHTREE_AGENTS[s["agent_study"]]):
            raise ValueError(f"study {s['agent_study']} agents are {TECHTREE_AGENTS[s['agent_study']]}")
    budgets, n_envs = cfg["train"]["budgets"], cfg["train"]["n_envs"]
    if not budgets or any(b <= 0 or b % n_envs for b in budgets) or sorted(budgets) != budgets:
        raise ValueError("train.budgets must be increasing positive multiples of train.n_envs")
    if bool(cfg["validation"]["namespace"]) != bool(cfg["validation"]["seeds"]):
        raise ValueError("validation needs both a namespace and seeds, or neither")
    if len(set(namespaces(cfg))) != len(namespaces(cfg)):
        raise ValueError("train, test and validation namespaces must differ")
    for t in cfg["analysis"]["tests"]:
        extra = set(t) - TEST_KEYS
        if extra or not {"name", "a", "metric"} <= set(t):
            raise KeyError(f"analysis test needs name, a, metric and optionally b (got {sorted(t)})")
        if t.get("expect", "less") not in EXPECT:
            raise ValueError(f"unknown expectation in analysis test {t['name']!r}")
        if t["metric"] not in METRICS:
            raise ValueError(f"unknown metric {t['metric']!r}; known: {METRICS}")
    if cfg["analysis"]["select_metric"] not in METRICS:
        raise ValueError(f"unknown select_metric {cfg['analysis']['select_metric']!r}")
    roles = [t.get("role") for t in cfg["analysis"]["tests"]]
    if roles.count("primary") > 1:
        raise ValueError("at most one primary test per config")
    alloc = cfg["run"]["allocation"]
    for n in namespaces(cfg):
        if alloc in DEV_ALLOCATIONS and not n.endswith("_dev"):
            raise ValueError(f"allocation {alloc!r} may only use namespaces ending in '_dev' (got {n!r})")
        if not (alloc in DEV_ALLOCATIONS and n.endswith("_dev")) and not cfg["run"]["requires_flag"]:
            raise ValueError(f"namespace {n!r} with allocation {alloc!r} needs run.requires_flag: true")
    if alloc not in cfg["ledger"]["allocations"]:
        raise KeyError(f"allocation {alloc!r} is not declared in ledger.allocations")
    cfg["_source"] = str(path)
    return cfg


LATER_KEYS = ("benchmark", "budgets")  # optional keys added to the schema after the first runs


def _set(d: dict) -> dict:
    """A config block as hashed: optional keys added later are left out while unset, so they change no
    hash of an earlier run."""
    return {k: v for k, v in d.items() if k not in LATER_KEYS or v not in (None, {})}


def setting_hash(cfg: dict) -> str:
    """The setting a learner is trained on (that of ``checkpoints_from`` for an evaluation-only config)."""
    return stable_hash({"setting": _set(cfg.get("_train_setting", cfg["setting"])),
                        "train": {k: v for k, v in cfg["train"].items() if k != "seeds"}})


TRAINING_KEYS = ("algo", "params", "trial", "rnd", "step_penalty")


def train_hash(cfg: dict, learner: dict, seed: int) -> str:
    """What determines a training run: setting, training stream, the learner's training keys, seed."""
    return stable_hash({"study": "X", "setting": setting_hash(cfg), "seed": seed,
                        "learner": {k: learner[k] for k in TRAINING_KEYS}})


def eval_hash(cfg: dict, learner: dict, seed: int, budget: int, split: str, mode: str) -> str:
    """An evaluation run plays every seed of the split with one checkpoint and one action mode."""
    stream = {k: v for k, v in _set(cfg[split]).items() if k != "budgets"}
    return stable_hash({"train": train_hash(cfg, learner, seed), "budget": budget, "split": split,
                        "mode": mode, "stream": stream, "setting": _set(cfg["setting"])})


def agent_hash(cfg: dict, agent: str, test_seed: int, split: str = "test") -> str:
    """A repository agent's run plays one seed's stream of the split."""
    stream = {k: v for k, v in _set(cfg[split]).items() if k not in ("seeds", "budgets")}
    return stable_hash({"study": "X", "agent": agent, "seed": test_seed, "split": split, "stream": stream,
                        "setting": _set(cfg["setting"]), "params": cfg["agent_params"]})
