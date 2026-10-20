"""Mechanical tuning amendment of study X: tuning stage -> measurement config.

    python -m hypergraph_agent.baselines apply-tuning --config configs/baselines/rq_f1_main.yaml

A measurement config names its tuning config (``run.tuning_from``) and maps each
tuning learner group onto one of its own learner ids (``run.tuning_map``).
``apply_tuning`` refuses unless every planned job of the tuning config has a
completed run and every candidate has full results in both action modes; it
then analyses the tuning config (``analysis.select_by_group``: lowest mean
selection metric, ties to the earlier candidate, then the deterministic mode),
freezes that analysis as ``artifacts/baselines/<tuning name>/tuning_summary.json``,
copies each group's chosen settings (``config.TRAINING_KEYS``) and mode into the
mapped learner of the measurement config and writes ``run.tuning_record``
(``summary``: that file, ``sha256``: its digest). It aborts, writing nothing,
when any group lacks a valid candidate. No human decision enters.

``verify_tuning`` runs before every measurement run: it checks the digest,
recomputes the choice from the frozen candidate scores and refuses when the
config's learners differ from it. An evaluation-only config is checked through
the config whose checkpoints it evaluates.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import yaml

from .config import MODES, TRAINING_KEYS, config_path, load_config


class TuningError(RuntimeError):
    pass


def sha256(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tuning_config(cfg: dict) -> dict:
    run = cfg["run"]
    if not run["tuning_from"] or not run["tuning_map"]:
        raise TuningError(f"{run['name']}: run.tuning_from and run.tuning_map are not declared")
    return load_config(config_path(cfg["_source"], run["tuning_from"]))


def recompute_choice(tuning: dict, tune_cfg: dict) -> dict:
    """group -> (candidate id, mode), recomputed from the candidate scores of a tuning analysis."""
    from .analysis import choose
    order = {ln["id"]: i for i, ln in enumerate(tune_cfg["learners"])}
    sign = -1.0 if tuning["metric"].startswith("success") else 1.0
    out = {}
    for group, g in tuning["groups"].items():
        scores = {tuple(k.rsplit("@", 1)): v for k, v in g["candidates"].items()}
        if not all(v is not None and np.isfinite(v) for v in scores.values()):
            raise TuningError(f"tuning group {group!r} has a candidate without a finite score")
        out[group] = choose(scores, order, sign)
    return out


def chosen_learners(cfg: dict, tune_cfg: dict, choice: dict) -> list[dict]:
    """The measurement learners with each mapped group's chosen settings and mode."""
    by_id = {ln["id"]: ln for ln in tune_cfg["learners"]}
    target = {lid: group for group, lid in cfg["run"]["tuning_map"].items()}
    out = []
    for ln in cfg["learners"]:
        group = target[ln["id"]]
        if group not in choice:
            raise TuningError(f"tuning group {group!r} (learner {ln['id']!r}) has no valid candidate")
        cand_id, mode = choice[group]
        cand = by_id[cand_id]
        if cand["group"] != group:
            raise TuningError(f"candidate {cand_id!r} is not in group {group!r}")
        new = copy.deepcopy(ln)
        new.update({k: copy.deepcopy(cand[k]) for k in TRAINING_KEYS})
        new["modes"] = [mode]
        out.append(new)
    return out


def _check_complete(tune_cfg: dict, cfg: dict) -> None:
    from .analysis import completeness
    comp = completeness(tune_cfg)
    if not comp["complete"]:
        raise TuningError(f"{tune_cfg['run']['name']}: {comp['missing']} of {comp['planned']} planned tuning "
                          f"jobs have no completed run")
    groups = {ln["group"] for ln in tune_cfg["learners"]}
    if set(cfg["run"]["tuning_map"]) != groups:
        raise TuningError(f"run.tuning_map keys {sorted(cfg['run']['tuning_map'])} are not the tuning groups "
                          f"{sorted(groups)}")
    for ln in tune_cfg["learners"]:
        if sorted(ln["modes"]) != sorted(MODES):
            raise TuningError(f"tuning candidate {ln['id']!r} is not evaluated in both modes")


def _learner_yaml(ln: dict) -> str:
    entry = {"id": ln["id"], "group": ln["group"], "algo": ln["algo"], "params": ln["params"]}
    if ln["trial"]:
        entry["trial"] = True
    if ln["rnd"] is not None:
        entry["rnd"] = ln["rnd"]
    if ln["step_penalty"]:
        entry["step_penalty"] = ln["step_penalty"]
    entry["modes"] = ln["modes"]
    text = yaml.safe_dump(entry, default_flow_style=True, sort_keys=False, width=10 ** 6).strip()
    pieces = re.split(r"(?<=,) ", text)  # a flow collection may break after any comma
    lines = ["  - " + pieces[0]]
    for piece in pieces[1:]:
        if len(lines[-1]) + 1 + len(piece) <= 106:
            lines[-1] += " " + piece
        else:
            lines.append("     " + piece)
    return "\n".join(lines)


def amend_text(text: str, learners: list[dict], summary: str, digest: str) -> str:
    """The measurement config's text with the learners block and ``tuning_record`` replaced."""
    lines = text.splitlines()
    start = next(i for i, x in enumerate(lines) if x.startswith("learners:"))
    end = next((i for i in range(start + 1, len(lines)) if re.match(r"[A-Za-z_]", lines[i])),
               len(lines))
    block = ["learners:  # written by apply-tuning from run.tuning_record", *map(_learner_yaml, learners)]
    out = "\n".join(lines[:start] + block + lines[end:]) + "\n"
    if len(re.findall(r"tuning_record: null", out)) != 1:
        raise TuningError("the config must contain exactly one 'tuning_record: null'")
    record = f"tuning_record: {{summary: {json.dumps(summary)},\n        sha256: {json.dumps(digest)}}}"
    return out.replace("tuning_record: null", record)


def verify_tuning(cfg: dict) -> dict:
    """Refuses (TuningError) unless the config's learners are what its tuning record selects."""
    if cfg["run"]["checkpoints_from"]:
        return verify_tuning(load_config(cfg["_source_path"]))
    rec = cfg["run"]["tuning_record"]
    if not isinstance(rec, dict) or not {"summary", "sha256"} <= set(rec):
        raise TuningError(f"{cfg['run']['name']}: run.tuning_record is unset; run apply-tuning first")
    path = Path(rec["summary"])
    if not path.exists() or sha256(path) != rec["sha256"]:
        raise TuningError(f"{cfg['run']['name']}: tuning summary {path} is missing or its sha256 differs")
    tune_cfg = tuning_config(cfg)
    tuning = json.loads(path.read_text())["tuning"]
    if tuning.get("incomplete"):
        raise TuningError(f"tuning record {path} is incomplete: {tuning['incomplete']}")
    choice = recompute_choice(tuning, tune_cfg)
    want = chosen_learners(cfg, tune_cfg, choice)
    keys = (*TRAINING_KEYS, "modes")
    for got, exp in zip(cfg["learners"], want):
        if {k: got[k] for k in keys} != {k: exp[k] for k in keys}:
            raise TuningError(f"{cfg['run']['name']}: learner {got['id']!r} differs from the tuning record")
    return {g: f"{c[0]}@{c[1]}" for g, c in choice.items()}


def apply_tuning(config: str | Path, artifacts: str | Path = "artifacts/baselines") -> dict:
    """Writes the tuning selection into the measurement config at ``config`` (see the module text)."""
    from .analysis import write_report
    cfg = load_config(config)
    if not cfg["run"]["requires_flag"] or cfg["run"]["checkpoints_from"]:
        raise TuningError("apply-tuning takes a measurement config that trains")
    if cfg["run"]["tuning_record"]:  # already applied: check it and stop
        return {"config": str(config), "status": "already applied", "choice": verify_tuning(cfg)}
    tune_cfg = tuning_config(cfg)
    _check_complete(tune_cfg, cfg)
    out = Path(artifacts) / tune_cfg["run"]["name"]
    summary = write_report(tune_cfg, out)
    tuning = summary["tuning"]
    lacking = {g: v for g, v in tuning.get("incomplete", {}).items()}
    missing = sorted(set(cfg["run"]["tuning_map"]) - set(tuning["groups"]))
    if lacking or missing:
        raise TuningError(f"no valid candidate for groups {missing}; incomplete {lacking}")
    choice = recompute_choice(tuning, tune_cfg)
    if any(choice[g] != (c["learner"], c["mode"]) for g, c in tuning["groups"].items()):
        raise TuningError("the recomputed choice differs from the analysis")
    learners = chosen_learners(cfg, tune_cfg, choice)
    frozen = out / "tuning_summary.json"
    frozen.write_text(json.dumps({k: v for k, v in summary.items() if k != "figure"}, indent=1, default=str))
    digest = sha256(frozen)
    path = Path(config)
    original = path.read_text()
    path.write_text(amend_text(original, learners, frozen.as_posix(), digest))
    try:
        verify_tuning(load_config(path))
    except Exception:
        path.write_text(original)
        raise
    return {"config": str(config), "status": "applied", "summary": frozen.as_posix(), "sha256": digest,
            "choice": {g: f"{c[0]}@{c[1]}" for g, c in choice.items()}}
