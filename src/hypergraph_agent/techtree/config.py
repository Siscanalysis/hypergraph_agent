"""Strict YAML configs for TechTree studies (``configs/unlock/``, ``configs/layers/``).

Files are merged onto ``DEFAULTS``; unknown keys raise. Variants override
``world`` and ``stream`` keys of the base config and may run a subset of its
``arms``, nothing else. Their schema
is this one, not the RecipeQuest schema of ``hypergraph_agent.config``.

Namespace rule: the ``dev`` allocation may only use namespaces ending in
``_dev``, and any other combination (a measurement namespace, or a ``_dev``
namespace charged to another allocation) needs ``run.requires_flag: true``; the
CLI then refuses to run without ``--allow-measurement``, so development can
never touch measurement worlds or allocations.

``analysis.verdict`` combines named tests into the study verdict: rule ``all``
(pass iff every ``require`` test passes) or rule ``precedence`` (``primary``,
``falsification``, ``conditions``, ``validity``; see ``analysis.verdict``).
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from ..envs.generator import stable_hash
from .generator import TechStreamConfig, TechWorldConfig

DEFAULTS: dict = {
    "run": {"name": "techtree", "study": "U", "allocation": "dev", "seeds": [0], "runs_dir": "runs",
            "ledger": "runs/U-ledger.json", "requires_flag": False},
    # caps written into a NEW ledger file; an existing ledger keeps its own
    "ledger": {"allocations": {"dev": 20_000}, "adaptive_cap": 20_000, "reporting_cap": 20_000},
    "world": {"n_primitives": 3, "n_slots": 0, "n_unlocks": 0, "unlock_level": 1, "n_levels": 3,
              "concepts_per_level": [3, 3, 3], "combo_length": 2, "length_rule": "linear",
              "reuse_depth": 0},
    "stream": {"namespace": "techtree_dev", "n_worlds": 2, "episodes_per_world": 12, "goal_levels": [3],
               "goal_rule": "uniform", "episode_budget": 100, "unlock_visibility": "announced",
               "signal": False, "reveal_levels": []},
    "variants": [],
    "arms": ["pooled"],
    # parameters of the study's agent module (free-form, validated by that module)
    "agent": {},
    "analysis": {"late_from": None, "n_boot": 10_000, "boot_seed": 20261008, "tests": [], "verdict": None},
}
STUDIES = ("U", "L", "U2", "P")
VERDICT_KEYS = {"all": {"require"}, "precedence": {"primary", "falsification", "conditions", "validity"}}
TEST_KEYS = {
    "paired": {"metric", "a", "b", "variant", "expect", "level"},
    "across": {"metric", "arm", "variant_a", "variant_b", "expect", "level"},
    "equivalence": {"metric", "a", "b", "variant", "margin_rel", "margin_abs", "level"},
    "ratio": {"metric", "a", "b", "variant", "band", "level"},
    "interaction": {"metric", "a", "b", "variant_hi", "variant_lo", "expect", "level"},
    "within_margin": {"metric", "a", "others", "variants", "margin_rel", "level"},
    "all_success": {"arm", "variants"},
    "at_least_reference": {"arm", "variants"},
}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        where = f"{path}.{k}" if path else k
        if k not in base and path not in ("ledger.allocations", "agent"):
            raise KeyError(f"unknown config key {where}")
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _merge(base[k], v, where)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    cfg = _merge(DEFAULTS, raw)
    if cfg["run"]["study"] not in STUDIES:
        raise ValueError(f"run.study must be one of {STUDIES}")
    names = set()
    for v in cfg["variants"]:
        extra = set(v) - {"name", "world", "stream", "arms"}
        if extra or "name" not in v:
            raise KeyError(f"variant needs a name and only world/stream/arms entries, got {sorted(v)}")
        if v["name"] in names:
            raise ValueError(f"duplicate variant {v['name']!r}")
        if not set(v.get("arms", [])) <= set(cfg["arms"]):
            raise ValueError(f"variant {v['name']!r} runs arms missing from the arms list")
        names.add(v["name"])
        _merge(DEFAULTS["world"], v.get("world", {}), "world")
        _merge(DEFAULTS["stream"], v.get("stream", {}), "stream")
    for t in cfg["analysis"]["tests"]:
        kind = t.get("type")
        if kind not in TEST_KEYS:
            raise KeyError(f"unknown analysis test type {kind!r}")
        extra = set(t) - TEST_KEYS[kind] - {"type", "name", "role"}
        if extra:
            raise KeyError(f"unknown keys {sorted(extra)} in analysis test {t.get('name')!r}")
        if t.get("expect", "less") not in ("less", "greater", "not_less", "not_greater"):
            raise ValueError(f"unknown expectation {t['expect']!r} in analysis test {t.get('name')!r}")
    tests = {t.get("name") for t in cfg["analysis"]["tests"]}
    verdict = cfg["analysis"]["verdict"]
    if verdict is not None:
        rule = verdict.get("rule")
        if rule not in VERDICT_KEYS or set(verdict) - VERDICT_KEYS[rule] - {"rule"}:
            raise KeyError(f"invalid analysis.verdict {verdict}")
        named = [n for k in VERDICT_KEYS[rule] for n in
                 (verdict.get(k) if isinstance(verdict.get(k), list) else [verdict.get(k)]) if n]
        missing = [n for n in named if n not in tests]
        if missing:
            raise KeyError(f"analysis.verdict names unknown tests {missing}")
    ns, alloc = cfg["stream"]["namespace"], cfg["run"]["allocation"]
    namespaces = {ns} | {v.get("stream", {}).get("namespace", ns) for v in cfg["variants"]}
    for n in namespaces:
        if alloc == "dev" and not n.endswith("_dev"):
            raise ValueError("the dev allocation may only use namespaces ending in '_dev'")
        if not (alloc == "dev" and n.endswith("_dev")) and not cfg["run"]["requires_flag"]:
            raise ValueError(f"namespace {n!r} with allocation {alloc!r} needs run.requires_flag: true")
    for variant in variants(cfg):
        stream_config(cfg, variant, 0)  # validates every variant
    cfg["_source"] = str(path)
    return cfg


def variants(cfg: dict) -> list[dict]:
    return cfg["variants"] or [{"name": "default"}]


def variant_arms(cfg: dict, variant: dict) -> list[str]:
    """The variant's arms, in the order of the config's arms list."""
    chosen = set(variant["arms"] if "arms" in variant else cfg["arms"])
    return [a for a in cfg["arms"] if a in chosen]


def merged(cfg: dict, variant: dict) -> tuple[dict, dict]:
    world = {**cfg["world"], **variant.get("world", {})}
    stream = {**cfg["stream"], **variant.get("stream", {})}
    return world, stream


def stream_config(cfg: dict, variant: dict, seed: int) -> tuple[TechStreamConfig, tuple[int, ...]]:
    w, s = merged(cfg, variant)
    world = TechWorldConfig(**{**w, "concepts_per_level": tuple(w["concepts_per_level"])})
    s = dict(s)
    reveal = tuple(s.pop("reveal_levels") or ())
    s["goal_levels"] = tuple(s["goal_levels"])
    return TechStreamConfig(seed=seed, world=world, **s), reveal


def run_hash(cfg: dict, variant: dict, arm: str, seed: int) -> str:
    """Hash of what determines a run's rows (study, name, world, stream, variant, arm, seed),
    independent of which arms or variants a command selected."""
    w, s = merged(cfg, variant)
    agent = {"agent": cfg["agent"]} if cfg.get("agent") else {}  # absent for studies U and L
    return stable_hash({"study": cfg["run"]["study"], "name": cfg["run"]["name"], "world": w, "stream": s,
                        "variant": variant["name"], "arm": arm, "seed": seed, **agent})
