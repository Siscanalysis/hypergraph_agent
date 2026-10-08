import importlib
import pathlib
import pkgutil

import pytest

import hypergraph_agent
from hypergraph_agent.config import load_config
from hypergraph_agent.train import main as train_main
from hypergraph_agent.train import plan

ROOT = pathlib.Path(__file__).resolve().parents[1]
# TechTree configs (configs/unlock, configs/layers) have their own schema and tests
TECHTREE_DIRS = ("unlock", "layers")
CONFIGS = sorted(p for p in (ROOT / "configs").rglob("*.yaml")
                 if p.relative_to(ROOT / "configs").parts[0] not in TECHTREE_DIRS)


def test_importing_every_module_starts_nothing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for mod in pkgutil.walk_packages(hypergraph_agent.__path__, "hypergraph_agent."):
        importlib.import_module(mod.name)
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize("path", CONFIGS, ids=lambda p: p.name)
def test_every_config_loads_and_plans(path):
    cfg = load_config(path)
    out = plan(cfg)
    assert out["arms"] and out["interactions"]["per_run_adaptive_cap"] > 0
    assert "None" not in str(out["interactions"])


def test_full_study_requires_explicit_activation(capsys):
    study = ROOT / "configs" / "study.yaml"
    assert train_main(["--config", str(study), "--dry-run"]) == 0
    with pytest.raises(SystemExit):
        train_main(["--config", str(study)])


def test_unknown_config_keys_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("run: {name: x, phse: p1}\n")
    with pytest.raises(KeyError):
        load_config(bad)


def test_gitignore_keeps_runs_and_caches_out():
    text = (ROOT / ".gitignore").read_text()
    for pattern in ("runs/", ".venv/", "*.pt", "__pycache__/"):
        assert pattern in text
