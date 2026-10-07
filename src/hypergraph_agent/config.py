"""Experiment configuration: YAML files merged onto declared defaults.

Unknown keys are rejected so a typo cannot silently fall back to a default.
"""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from .envs.generator import TaskConfig, TaskStreamConfig, WorldConfig
from .training.ppo import PPOConfig

DEFAULTS: dict = {
    "run": {
        "name": "run", "phase": "smoke", "allocation": "smoke", "seeds": [0], "threads": 2,
        "wallclock_s": 1800, "runs_dir": "runs", "ledger": "runs/ledger.json",
        "requires_flag": None,
    },
    "budget": {"per_run_interactions": 2000, "batch_steps": 512, "reporting_eval_per_run": 600,
               "pretraining_interactions": 0,
               # caps written into a NEW ledger file (ignored when the ledger exists);
               # the default is the development session allowance
               "ledger_allocations": None, "ledger_adaptive_cap": None,
               "ledger_reporting_cap": None},
    "model": {"d": 64, "hidden": 64, "rounds": 3, "layers": 2, "heads": 4},
    "ppo": PPOConfig().to_dict(),
    "world": {
        "n_levels": 8, "items_per_level": 2, "pool_size": 6, "max_true": 3,
        "arity_weights": [0.2, 0.5, 0.3], "p_alternative": 0.25, "p_extra_item_input": 0.0,
    },
    "train": {
        "world_mode": "per_task", "n_worlds": 0, "shared_world_seed": 0, "goal_levels": None,
        "repeat_index": None,
        "tasks": {
            "profile": "known_structure", "depth_min": 1, "depth_max": 4,
            "n_distractor_rules": 1, "n_distractor_base": 1, "p_init_base": 0.0,
            "p_init_item": 0.0, "failure_prob": 0.0, "budget_factor": 1.0, "budget_slack": 2,
            "max_budget": 256,
        },
    },
    "eval": {
        "namespace": "val", "base_seed": 10_000, "n_tasks": 30, "track": "frozen",
        "greedy": False, "world_mode": None, "n_worlds": 0, "episodes_per_world": 1,
        "goal_levels": None, "tasks": {},
    },
    "topology": {"credible_mass": 0.9, "max_size": 3},
    "skills": {},
    "arms": [{"id": "gated", "encoder": "gated", "graph_mode": "known", "topology": "none",
              "use_posterior": True}],
}

ARM_DEFAULTS = {"encoder": "gated", "graph_mode": "known", "topology": "none",
                "use_posterior": True,
                # P3 only: library growth after the fork, dependency/contract revision
                "growth": False, "revision": False}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k not in base and path not in ("eval.tasks", "skills"):
            raise KeyError(f"unknown config key {path + '.' + k if path else k}")
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            out[k] = _merge(base[k], v, f"{path}.{k}" if path else k)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    arms = raw.pop("arms", None)
    cfg = _merge(DEFAULTS, raw)
    if arms is not None:
        cfg["arms"] = []
        for a in arms:
            extra = set(a) - set(ARM_DEFAULTS) - {"id"}
            if extra:
                raise KeyError(f"unknown arm keys {sorted(extra)}")
            cfg["arms"].append({**ARM_DEFAULTS, **a})
    cfg["_source"] = str(path)
    return cfg


def world_config(cfg: dict) -> WorldConfig:
    w = dict(cfg["world"])
    w["arity_weights"] = tuple(w["arity_weights"])
    return WorldConfig(**w)


def train_stream_config(cfg: dict, seed: int) -> TaskStreamConfig:
    t = cfg["train"]
    gl = tuple(t["goal_levels"]) if t["goal_levels"] else None
    return TaskStreamConfig("train", seed, t["world_mode"], t["n_worlds"], t["shared_world_seed"],
                            world_config(cfg), TaskConfig(**t["tasks"]), gl, t["repeat_index"])


def eval_stream_config(cfg: dict, namespace: str | None = None) -> TaskStreamConfig:
    e, t = cfg["eval"], cfg["train"]
    tasks = TaskConfig(**{**t["tasks"], **e["tasks"]})
    mode = e["world_mode"] or t["world_mode"]
    n_worlds = e["n_worlds"] or t["n_worlds"]
    gl = e["goal_levels"] if e["goal_levels"] is not None else t["goal_levels"]
    return TaskStreamConfig(namespace or e["namespace"], e["base_seed"], mode, n_worlds,
                            t["shared_world_seed"], world_config(cfg), tasks,
                            tuple(gl) if gl else None)


def ppo_config(cfg: dict) -> PPOConfig:
    return PPOConfig(**cfg["ppo"])
