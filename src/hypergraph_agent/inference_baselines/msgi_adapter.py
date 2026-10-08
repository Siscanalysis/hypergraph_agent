"""MSGI's precondition inference (Sohn et al., ICLR 2020) through the authors' code.

The code is vendored unchanged in ``third_party/msgi`` (MIT; see its
``SOURCE.md``). MSGI infers each subtask's precondition from (completion
vector, eligibility) pairs with its own CART, decodes the tree into a sum of
products (OR of AND terms, literals positive or negated) and simplifies it.
Its full pipeline (``ILP.infer_graph``) needs the eligibility of every subtask
at every step, which MSGI's environment reveals and RecipeQuest does not:
here eligibility is observed only through an attempted craft (the previously
absent effect appeared or not). The adapter therefore runs the authors'
``cart_train``, ``decode_cart`` and ``simplify_Kmap`` per recipe on its
informative attempts, with the recipe's public pool and item inputs as the
features (MSGI's layer search would choose features among all subtasks), and
applies MSGI's two rules for subtasks that need no tree: eligible at every
observation means no precondition (MSGI's first layer); never eligible means
unknown (MSGI leaves such subtasks out of the inferred graph).

Scores: a pool candidate's score is the share of AND terms in which it is a
positive literal (1 or 0 with a single term). A recipe whose precondition is
unknown, or whose inference raised an error, has no model.

Public data only.
"""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

import torch

THIRD_PARTY = Path(__file__).resolve().parents[3] / "third_party" / "msgi"
_ILP = None
_NAMES = ("graph", "graph.ILP", "graph.graph_utils")


class Unavailable(RuntimeError):
    """The vendored MSGI code is not present (for example in a non-editable install)."""


def load_ilp():
    """An ``ILP`` instance of the vendored code, imported under its original
    package name ``graph`` without leaving that name in ``sys.modules``."""
    global _ILP
    if _ILP is None:
        gdir = THIRD_PARTY / "graph"
        if not (gdir / "ILP.py").exists():
            raise Unavailable(f"vendored MSGI code not found under {gdir}")
        saved = {k: sys.modules.pop(k) for k in _NAMES if k in sys.modules}
        pkg = types.ModuleType("graph")
        pkg.__path__ = [str(gdir)]
        sys.modules["graph"] = pkg
        try:
            mod = importlib.import_module("graph.ILP")
            utils = sys.modules["graph.graph_utils"]
        finally:
            for k in _NAMES:
                sys.modules.pop(k, None)
            sys.modules.update(saved)
        _ILP = mod.ILP(utils.dotdict(num_processes=1, act_dim=1, tr_epi=1, bonus_mode=0))
    return _ILP


@dataclass
class Precondition:
    status: str  # "tree", "always" (no precondition), "never" (unknown), "failed"
    terms: list = field(default_factory=list)  # AND terms: {feature: +1 | -1}
    error: str | None = None


def infer_precondition(X, y, features) -> Precondition:
    """MSGI's precondition of one subtask from boolean feature rows ``X`` and
    eligibility labels ``y`` (the authors' code; see the module docstring)."""
    if not len(y):
        return Precondition("never")
    yt = torch.tensor([bool(v) for v in y], dtype=torch.bool)
    if bool(yt.all()):
        return Precondition("always")
    if not bool(yt.any()):
        return Precondition("never")
    Xt = torch.tensor([[bool(v) for v in row] for row in X], dtype=torch.bool)
    ilp = load_ilp()
    try:
        mask = torch.ones(len(features), dtype=torch.int)
        root = ilp.cart_train(mask, Xt, yt)
        kmap = ilp.decode_cart(root, len(features))
        ilp.simplify_Kmap(kmap)  # as in ILP._infer_precondition: in place, return value unused
    except Exception as exc:  # the authors' code raises on degenerate trees
        return Precondition("failed", error=f"{type(exc).__name__}: {exc}")
    rows = []
    for row in kmap.tolist():  # duplicate AND terms merged, as in ILP._fillout_mat
        if row not in rows:
            rows.append(row)
    terms = [{f: int(v) for f, v in zip(features, row) if v != 0} for row in rows]
    return Precondition("tree", terms)


def msgi_scores(infos: dict, examples, sigs) -> tuple[dict, set, dict]:
    """Scores of every pool candidate of the recipes ``sigs`` from the
    informative attempts ``examples`` (``traces.CraftExample``). Returns
    (scores, recipes with a model, diagnostics)."""
    by_sig: dict = {}
    for e in examples:
        by_sig.setdefault(e.signature, []).append(e)
    scores, modelled = {}, set()
    diag = {"tree": 0, "always": 0, "never": 0, "failed": 0, "negated_literals": 0, "multi_term": 0,
            "errors": []}
    for sig in sigs:
        info = infos[sig]
        features = sorted(info.pool) + sorted(info.items)
        ex = by_sig.get(sig, [])
        pre = infer_precondition([[f in e.present for f in features] for e in ex],
                                 [e.success for e in ex], features)
        diag[pre.status] += 1
        if pre.error:
            diag["errors"].append(f"{sig}: {pre.error}")
        if pre.status in ("never", "failed"):
            continue
        modelled.add(sig)
        if pre.status == "always":
            scores.update({(sig, b): 0.0 for b in info.pool})
            continue
        diag["negated_literals"] += sum(v < 0 for t in pre.terms for v in t.values())
        diag["multi_term"] += len(pre.terms) > 1
        for b in info.pool:
            scores[(sig, b)] = sum(t.get(b, 0) > 0 for t in pre.terms) / len(pre.terms)
    return scores, modelled, diag
