import numpy as np
import pytest
import torch

from hypergraph_agent.agents.policy import ActorCritic
from hypergraph_agent.envs.recipequest import RecipeQuestEnv
from hypergraph_agent.skills.executor import Executor
from hypergraph_agent.training.budget import BudgetExhausted, BudgetMeter, SessionLedger
from hypergraph_agent.training.rollout import run_episode

from helpers import CHAIN_FACTS, ScriptedPolicy, chain_world, small_stream, task
from skill_fixtures import executor, ingot_key_library


def test_meter_caps_and_purposes():
    led = SessionLedger(None, allocations={"diagnostics": 10})
    m = BudgetMeter(led, "r", "diagnostics", run_cap=100)
    assert m.cap == 10  # the allocation is smaller than the run cap
    m.charge(4, "exploration")
    m.charge(3, "skill_practice")
    with pytest.raises(BudgetExhausted):
        m.charge(4, "candidate_validation")
    with pytest.raises(ValueError):
        m.charge(1, "free_lunch")
    m.flush()
    assert led.state["by_allocation"]["diagnostics"] == 7
    assert led.state["runs"]["r"]["by_purpose"] == {"exploration": 4, "skill_practice": 3}


def test_ledger_persists_and_separates_reporting_and_logical(tmp_path):
    path = tmp_path / "ledger.json"
    a = SessionLedger(path)
    a.register_run("r1", "p3", 50)
    m = BudgetMeter(a, "r1", "p3", 50)
    m.charge(20, "pretraining")
    m.flush()
    rep = BudgetMeter(a, "r1:eval", "p3", 5, kind="reporting")
    rep.charge(5, "reporting_eval")
    rep.flush()
    a.logical_charge("arm1", "r1", 20, "shared prefix")
    b = SessionLedger(path)  # a separate command sees the same allowance
    assert b.state["adaptive_total"] == 20 and b.state["reporting_total"] == 5
    assert b.state["logical_charges"][0]["n"] == 20
    assert b.remaining("p3") == 66_000 - 20
    with pytest.raises(ValueError):
        b.register_run("r1", "p3", 10)


def test_amendments_cannot_exceed_the_session_cap(tmp_path, capsys):
    from hypergraph_agent.ledger import main as ledger_main
    path = str(tmp_path / "l.json")
    ledger_main(["init", "--ledger", path, "--allocations", '{"p1": 150000}'])
    assert ledger_main(["amend", "--ledger", path, "--allocation", "diagnostics", "--add", "10000",
                        "--note", "x"]) == 0
    with pytest.raises(SystemExit):
        ledger_main(["amend", "--ledger", path, "--allocation", "diagnostics", "--add", "1", "--note", "y"])
    st = SessionLedger(path).state
    assert st["allocations"]["diagnostics"] == 10000 and st["amendments"][0]["note"] == "x"


def test_manifest_records_process_provenance(tmp_path):
    from hypergraph_agent.training.run import LOADED_CODE_SHA256, RunContext
    ctx = RunContext(tmp_path, "r", {"a": 1}, phase="p1", arm="x", seed=0)
    src = ctx.manifest["source"]
    assert src["loaded_code_sha256"] == LOADED_CODE_SHA256 and "commit_at_run_creation" in src


def test_two_handles_on_one_ledger_add_up(tmp_path):
    path = tmp_path / "ledger.json"
    a, b = SessionLedger(path), SessionLedger(path)
    a.record("x", "p1", 3, {"exploration": 3}, "adaptive")
    b.record("y", "p2", 4, {"exploration": 4}, "adaptive")
    assert SessionLedger(path).state["adaptive_total"] == 7


def test_budget_cut_drops_unexecuted_decision_and_bootstraps():
    t = small_stream().task(0)
    led = SessionLedger(None)
    m = BudgetMeter(led, "r", "diagnostics", run_cap=3)
    torch.manual_seed(0)
    pol = ActorCritic("incidence", d=16, hidden=16)
    ex = Executor(RecipeQuestEnv(), m, "exploration", torch.Generator().manual_seed(0))
    rec = run_episode(pol, ex, t, 0)
    assert rec.status == "budget_exhausted" and rec.n_primitive == 3 and m.used == 3
    assert len(rec.actions) == 3 and all(rec.valid) and not rec.terminal
    adv, tgt, mask = rec.advantages(1.0, 1.0)
    assert tgt[-1] == pytest.approx(rec.final_value)


def test_budget_cut_inside_an_option_excludes_the_partial_item():
    lib, _, _ = ingot_key_library()
    ex, m = executor(lib, cap=2)
    pol = ScriptedPolicy(["skill:achieve[ingot]"])
    rec = run_episode(pol, ex, task(chain_world(), "key", CHAIN_FACTS), 0)
    assert rec.status == "budget_exhausted" and m.used == 2
    assert rec.valid == [False] and rec.tau == [2]
    adv, tgt, mask = rec.advantages(1.0, 0.9)
    assert not mask.any()


def test_failed_and_invalid_calls_are_charged():
    lib, _, _ = ingot_key_library()
    ex, m = executor(lib)
    pol = ScriptedPolicy(["craft:r1_", "submit"])  # ineligible craft, premature submit
    t = task(chain_world(), "key", CHAIN_FACTS, budget=5)
    rec = run_episode(pol, ex, t, 0)
    assert m.used == 5 == rec.n_primitive and rec.n_noop == 5 and rec.status == "deadline"
