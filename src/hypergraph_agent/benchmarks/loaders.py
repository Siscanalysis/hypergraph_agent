"""Loaders: each reads one benchmark's source files and returns ``RecipeGraph``s.

Every loader states, in the graph's ``notes``, what it changed or left out.
Sources and licenses are documented in ``benchmarks/<name>/SOURCE.md``.

* ``crafter``: ``data.yaml`` of Crafter 1.8.3 (collect, place, make).
* ``psketch_craft``: ``recipes.yaml`` of the Craft world of policy sketches,
  plus the two barrier rules of its ``worlds/craft.py``.
* ``msgi_mining``: the Mining subtask graph of MSGI (``mining_graph.json``).
* ``little_alchemy``: the Little Alchemy 1 and 2 combination tables of the
  Brändle et al. (2023) data archive (fetched, not committed).
* ``freeciv``: Freeciv technology trees (``classic`` and ``civ2civ3``
  rulesets) with the units and buildings each technology makes available
  (fetched, not committed).
"""

from __future__ import annotations

import json
import re
from functools import partial
from pathlib import Path

import yaml

from . import sources
from .graph import Recipe, RecipeGraph, Unlock, make_graph


def _path(benchmark: str, rel: str, path) -> Path:
    return Path(path) if path is not None else sources.require(benchmark, rel)


def _amounts(d: dict) -> tuple:
    return tuple(sorted((k, int(v)) for k, v in d.items()))


# ------------------------------------------------------------------ Crafter
CRAFTER_PLACED = {"stone": "placed_stone"}  # the placed block is not the inventory item
CRAFTER_UNMODELLED = ("eat_cow", "eat_plant", "defeat_zombie", "defeat_skeleton", "wake_up")


def load_crafter(path=None) -> RecipeGraph:
    """Elements: terrain materials (``mat:<name>``, base), inventory items,
    placed objects (``table``, ``furnace``, ``plant``, ``placed_stone``).
    Collecting needs the facing material and the listed tools; placing needs
    the used items and a target tile among ``where`` (one alternative recipe
    per tile material); making needs the used items and the nearby stations."""
    data = yaml.safe_load(_path("crafter", "data.yaml", path).read_text(encoding="utf-8"))
    recipes, base, kinds = [], set(), {}
    notes = ["quantities are kept as metadata only: facts are Boolean and nothing is consumed",
             "achievements " + ", ".join(CRAFTER_UNMODELLED) + " depend on creatures, plant growth or "
             "sleep, which data.yaml does not describe; they are not represented"]

    def mat(m):
        base.add(f"mat:{m}")
        kinds[f"mat:{m}"] = "resource"
        return f"mat:{m}"

    for m, info in data["collect"].items():
        for item in info["receive"]:
            recipes.append(Recipe(item, (mat(m),) + tuple(info.get("require", {})), f"collect:{m}",
                                  _amounts(info.get("require", {}))))
        if info.get("probability", 1) < 1:
            notes.append(f"collect:{m} succeeds with probability {info['probability']}; treated as certain")
    for name, info in data["place"].items():
        target = CRAFTER_PLACED.get(name, name)
        for w in info["where"]:
            recipes.append(Recipe(target, tuple(info["uses"]) + (mat(w),), f"place:{name}",
                                  _amounts(info["uses"])))
    for name, info in data["make"].items():
        recipes.append(Recipe(name, tuple(info["uses"]) + tuple(info["nearby"]), f"make:{name}",
                              _amounts(info["uses"])))
    for station in {s for info in data["make"].values() for s in info["nearby"]}:
        kinds[station] = "station"
    return make_graph("crafter", base, recipes, kinds=kinds, notes=notes, source="crafter")


def crafter_achievements(path=None) -> dict[str, str | None]:
    """Achievement -> the element whose first acquisition unlocks it (None
    for the achievements data.yaml does not describe)."""
    data = yaml.safe_load(_path("crafter", "data.yaml", path).read_text(encoding="utf-8"))
    out = {}
    for a in data["achievements"]:
        verb, _, obj = a.partition("_")
        if a in CRAFTER_UNMODELLED:
            out[a] = None
        elif verb == "place":
            out[a] = CRAFTER_PLACED.get(obj, obj)
        else:
            out[a] = obj
    return out


# ------------------------------------------------------------ policy sketches
def load_psketch_craft(path=None) -> RecipeGraph:
    """Primitives are gathered, workshops are facilities, a recipe needs its
    ingredients and its workshop (``_at``). ``gold`` sits behind water and
    needs a ``bridge``; ``gem`` sits behind stone and needs an ``axe``
    (``worlds/craft.py``); their locations are base facts ``src:gold`` and
    ``src:gem``."""
    data = yaml.safe_load(_path("psketch_craft", "recipes.yaml", path).read_text(encoding="utf-8"))
    workshops = [e for e in data["environment"] if e.startswith("workshop")]
    behind = {"gold": "bridge", "gem": "axe"}
    base = set(workshops) | {p for p in data["primitives"] if p not in behind} | {f"src:{p}" for p in behind}
    kinds = {w: "facility" for w in workshops}
    kinds.update({b: "resource" for b in base if b not in kinds})
    recipes = []
    for out, ins in data["recipes"].items():
        ing = {k: v for k, v in ins.items() if not k.startswith("_")}
        recipes.append(Recipe(out, tuple(ing) + (ins["_at"],), f"use:{ins['_at']}", _amounts(ing)))
    for item, tool in behind.items():
        recipes.append(Recipe(item, (tool, f"src:{item}"), f"get:{item}"))
    notes = ["quantities are kept as metadata only: facts are Boolean and nothing is consumed",
             "the gold and gem rules (bridge across water, axe through stone) come from worlds/craft.py",
             "in the source, using a workshop crafts every recipe of that workshop whose ingredients are "
             "held; here each recipe is a separate action"]
    return make_graph("psketch_craft", base, recipes, kinds=kinds, notes=notes, source="psketch_craft")


def psketch_goals(path=None) -> list[str]:
    """The ten tasks of the Craft world (``hints.yaml``) as goal elements."""
    hints = yaml.safe_load(_path("psketch_craft", "hints.yaml", path).read_text(encoding="utf-8"))
    return [re.match(r"\w+\[(\w+)\]", k).group(1) for k in hints]


# ---------------------------------------------------------------- MSGI Mining
def load_msgi_mining(path=None) -> RecipeGraph:
    """Subtasks are elements. A subtask without preconditions is a base fact.
    Every other subtask has one recipe per AND-node of its OR: the subtasks of
    that node plus the object the subtask is executed at (``obj:<name>``,
    implicit in the source: the agent must reach it), which is a base fact."""
    doc = json.loads(_path("msgi_mining", "mining_graph.json", path).read_text(encoding="utf-8"))
    sub = {s["id"]: s for s in doc["subtasks"]}
    base, kinds, recipes = set(), {}, []
    for rule in doc["rules"]:
        s = sub[rule["subtask"]]
        if not rule["or"]:
            base.add(s["name"])
            kinds[s["name"]] = "resource"
            continue
        obj = f"obj:{s['object']}"
        base.add(obj)
        kinds[obj] = "facility" if s["operation"] == "transform" else "resource"
        for node in rule["or"]:
            recipes.append(Recipe(s["name"], tuple(sub[j]["name"] for j in node) + (obj,),
                                  f"{s['operation']}:{s['object']}"))
    notes = ["subtask rewards and the per-task subtask subsets of the source are not part of the graph",
             "the object a subtask is executed at is added as a base requirement"]
    return make_graph("msgi_mining", base, recipes, kinds=kinds, notes=notes, source="msgi_mining")


# ------------------------------------------------------------- Little Alchemy
LA1_BASE = ("air", "earth", "fire", "water")


def load_little_alchemy(version: int = 2, path=None) -> RecipeGraph:
    """Two-element combinations. ``parents`` lists the pairs that create an
    element; a pair of one element with itself becomes a one-requirement recipe
    (amount 2 in metadata). Base: the ``prime`` elements (version 2) or air,
    earth, fire and water (version 1, whose file has no flags). Version 2's
    conditional elements become unlocks: ``progress`` (after N elements) or
    ``k_of_n`` (after ``min`` of the listed elements). Descriptions are not
    read."""
    if version not in (1, 2):
        raise ValueError("version must be 1 or 2")
    rel = f"data/alchemy{version}Gametree.json"
    data = json.loads(_path("little_alchemy", rel, path).read_text(encoding="utf-8"))
    names = {int(k): v["name"] for k, v in data.items()}
    if len(set(names.values())) != len(names):
        raise ValueError("element names are not unique")
    base = {v["name"] for v in data.values() if v.get("prime")} if version == 2 else set(LA1_BASE)
    recipes, unlocks, kinds = [], [], {b: "resource" for b in base}
    for k, v in data.items():
        for a, b in v["parents"]:
            na, nb = names[int(a)], names[int(b)]
            recipes.append(Recipe(v["name"], (na, nb), "combine", ((na, 2),) if na == nb else ()))
        cond = v.get("condition")
        if cond:
            kinds[v["name"]] = "unlockable"
            if cond["type"] == "progress":
                unlocks.append(Unlock(v["name"], "progress", (), int(cond["total"]), "progress"))
            elif cond["type"] == "elements":
                listed = tuple(sorted(names[int(e)] for e in cond["elements"]))
                unlocks.append(Unlock(v["name"], "k_of_n", listed, int(cond["min"]), "elements"))
            else:
                raise ValueError(f"unknown condition type {cond['type']!r}")
        if v.get("hidden"):
            kinds[v["name"]] = "secret"
    notes = ["element descriptions are game text and are not read",
             "a combination of an element with itself requires that element once (amount 2 kept as metadata)"]
    if version == 1:
        notes.append("version 1 has no base flags; base = air, earth, fire, water")
    return make_graph(f"little_alchemy_{version}", base, recipes, unlocks, kinds, notes,
                      source="little_alchemy", elements=names.values())


# ------------------------------------------------------------------- Freeciv
_SECTION = re.compile(r"^\[([a-z0-9_]+)\]\s*$", re.M)
_TOKEN = re.compile(r'"((?:[^"\\]|\\.)*)"|([A-Za-z_][A-Za-z0-9_]*)')


def _strip_comments(text: str) -> str:
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith((";", "#")))


def ruleset_sections(text: str) -> list[tuple[str, str]]:
    text = _strip_comments(text)
    marks = list(_SECTION.finditer(text))
    return [(m.group(1), text[m.end(): marks[i + 1].start() if i + 1 < len(marks) else len(text)])
            for i, m in enumerate(marks)]


def _field(body: str, key: str) -> str | None:
    m = re.search(rf'^{key}\s*=\s*(?:_\(\s*)?"((?:[^"\\]|\\.)*)"', body, re.M)
    if not m:
        return None
    value = m.group(1)
    return re.sub(r"^\?[^:]*:", "", value)  # drop translation qualifiers such as "?unit:"


def _reqs(body: str) -> list[dict]:
    """Rows of the ``reqs`` requirement vector as dicts keyed by its header."""
    m = re.search(r"^reqs\s*=\s*\{(.*?)\}", body, re.M | re.S)
    if not m:
        return []
    rows = [[a if a else b for a, b in _TOKEN.findall(line)] for line in m.group(1).splitlines()]
    rows = [r for r in rows if r]
    header, out = rows[0], []
    for r in rows[1:]:
        out.append(dict(zip(header, r)))
    return out


def load_freeciv(ruleset: str = "classic", path=None, unlockables: bool = True) -> RecipeGraph:
    """Technologies are elements; a technology with no prerequisite is a base
    fact; every other one has one recipe requiring ``req1``, ``req2`` and its
    ``root_req``. Technologies that require "Never" are left out. With
    ``unlockables``, every unit and building whose requirement vector names
    technologies is an element made available (unlock ``all``) when they are
    all known; its non-technology requirements are not represented."""
    root = Path(path) if path is not None else None

    def read(kind):
        rel = f"data/{ruleset}/{kind}.ruleset"
        p = root / f"{kind}.ruleset" if root is not None else sources.require("freeciv", rel)
        return p.read_text(encoding="utf-8")

    techs = {}
    for sec, body in ruleset_sections(read("techs")):
        if not sec.startswith("advance_"):
            continue
        name = _field(body, "name")
        key = _field(body, "rule_name") or name
        reqs = [_field(body, k) for k in ("req1", "req2", "root_req")]
        techs[key] = [r for r in reqs if r and r != "None"]
    notes, never = [], sorted(t for t, r in techs.items() if "Never" in r)
    for t in never:
        del techs[t]
    if never:
        notes.append(f"left out {len(never)} technolog(ies) that require 'Never': " + ", ".join(never))
    changed = True
    while changed:  # technologies that depend on a removed one are unreachable too
        changed = False
        for t in [t for t, r in techs.items() if any(x not in techs for x in r)]:
            del techs[t]
            notes.append(f"left out {t}: a prerequisite is unavailable")
            changed = True
    base = {t for t, r in techs.items() if not r}
    recipes = [Recipe(t, tuple(r), "research") for t, r in techs.items() if r]
    kinds = {b: "resource" for b in base}
    notes.append("root_req is added to the requirements of the technology that declares it")
    unlocks = []
    if unlockables:
        dropped_other = no_tech = negated = 0
        for kind in ("units", "buildings"):
            prefix = "unit" if kind == "units" else "building"
            for sec, body in ruleset_sections(read(kind)):
                if not sec.startswith(prefix + "_"):
                    continue
                name = _field(body, "name")
                rows = _reqs(body)
                tech, other = [], 0
                for row in rows:
                    if row.get("type") == "Tech":
                        if str(row.get("present", "TRUE")).upper() == "FALSE":
                            negated += 1
                        elif row.get("name") in techs:
                            tech.append(row["name"])
                    else:
                        other += 1
                dropped_other += other
                if not tech:
                    no_tech += 1
                    continue
                target = f"{prefix}:{name}"
                kinds[target] = "unlockable"
                unlocks.append(Unlock(target, "all", tuple(sorted(set(tech))), len(set(tech)), prefix))
        notes.append(f"{no_tech} units/buildings without a technology requirement are not represented; "
                     f"{dropped_other} non-technology requirement rows and {negated} negated technology "
                     f"rows are ignored")
    return make_graph(f"freeciv_{ruleset}", base, recipes, unlocks, kinds, notes, source="freeciv")


# ------------------------------------------------------------------ registry
GRAPHS = {
    "crafter": ("crafter", load_crafter),
    "psketch_craft": ("psketch_craft", load_psketch_craft),
    "msgi_mining": ("msgi_mining", load_msgi_mining),
    "little_alchemy_1": ("little_alchemy", partial(load_little_alchemy, 1)),
    "little_alchemy_2": ("little_alchemy", partial(load_little_alchemy, 2)),
    "freeciv_classic": ("freeciv", partial(load_freeciv, "classic")),
    "freeciv_civ2civ3": ("freeciv", partial(load_freeciv, "civ2civ3")),
}
BENCHMARKS = tuple(dict.fromkeys(b for b, _ in GRAPHS.values()))


def graphs_of(benchmark: str) -> list[str]:
    return [g for g, (b, _) in GRAPHS.items() if b == benchmark]


def load(graph: str) -> RecipeGraph:
    """Load a graph by name (see ``GRAPHS``); raises ``sources.DataMissing``
    when a fetch-only benchmark has not been downloaded."""
    return GRAPHS[graph][1]()


def available(benchmark: str) -> bool:
    """Whether every file the loaders read is present and verified (files with
    ``"role": "source"`` are needed only to regenerate committed files)."""
    files = [f for f in sources.manifest(benchmark)["files"]
             if not f.get("optional") and f.get("role", "data") == "data"]
    status = sources.verify(benchmark)
    return all(status[f["path"]] == "ok" for f in files)
