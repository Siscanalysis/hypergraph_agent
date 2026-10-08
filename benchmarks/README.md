# External benchmarks

Published recipe, crafting and technology graphs that test this repository's
questions, with their sources, licenses, loaders and statistics. The
questions:

- **Q1.** Hidden conjunctive preconditions with AND/OR structure: items made
  by recipes that need a hidden subset of candidate ingredients, with
  alternative recipes (RecipeQuest; studies R and M).
- **Q2.** Useful actions that are not listed, and achievements that unlock
  further ones (TechTree; study U).
- **Q3.** Discoveries reused as building blocks of later discoveries
  (compositional depth of a discovery graph; study L).

Every benchmark is read into one neutral format, a `RecipeGraph`
(`src/hypergraph_agent/benchmarks/graph.py`): base elements, derived elements,
recipes `effect <- AND(requirements)` (several recipes per effect are OR) and
optional unlocks (an element granted when all, or k of n, listed elements
are held, or after a number of discoveries). Converters build RecipeQuest
worlds and TechTree-like concept graphs from it. Licenses were verified on
2026-10-08 from each primary source (repository `LICENSE` file, data
archive page, GitHub license API); each selected folder's `SOURCE.md`
records what was seen.

## Layout and use

```
benchmarks/
  README.md                 this catalog
  generated_families.json   statistics of the repository's generators (F1, F2, TechTree)
  <name>/
    SOURCE.md               citation, URL, commit, license, files with SHA-256
    manifest.json           the same file list, read by fetch.py and the tests
    stats.json              graph statistics (generated)
    fetch.py                download and verify (fetch-only benchmarks)
    data/                   downloaded files, not tracked
```

```bash
python benchmarks/little_alchemy/fetch.py     # fetch-only data, verified by SHA-256
python benchmarks/freeciv/fetch.py
python benchmarks/msgi_mining/extract.py --check   # rebuild mining_graph.json from the source pickle
python -m hypergraph_agent.benchmarks.stats --write --families   # rewrite every stats.json
python -m hypergraph_agent.benchmarks.stats --export ../graphs/  # neutral JSON, outside the repository
python -m pytest -q tests/test_benchmarks.py
```

```python
from hypergraph_agent.benchmarks import loaders
from hypergraph_agent.benchmarks.to_recipequest import recipequest_world, sink_goals
from hypergraph_agent.benchmarks.to_techtree import techworld_from_graph, techtask

g = loaders.load("crafter")                     # RecipeGraph
cw = recipequest_world(g, sink_goals(g))        # one RecipeQuest world (16 items)
task = cw.task("diamond")                       # hidden prerequisites, as in study R
tc = techworld_from_graph(loaders.load("little_alchemy_2"), "linear", combo_length=1)
ttask = techtask(tc, tc.names[-1])              # a TechTree task on real parent structure
```

Graph names: `crafter`, `psketch_craft`, `msgi_mining`, `little_alchemy_1`,
`little_alchemy_2`, `freeciv_classic`, `freeciv_civ2civ3`. The loaders of
fetch-only benchmarks raise `DataMissing` with the fetch command when the
data is absent, and their tests are skipped with the same message.

## Catalog

Size counts are after loading (see each `SOURCE.md` for what the loader
drops). Effort is the work to integrate as a recipe graph.

| Candidate | Questions, and how | Source | License as verified (2026-10-08) | Decision | Size | Effort | Recommendation |
|---|---|---|---|---|---|---|---|
| Little Alchemy 2 (data of Brändle et al. 2023) | Q1: two-element AND recipes, 4.8 recipes per element (OR); Q2: 9 elements granted by progress or k-of-n unlocks; Q3: depth 15, 74% of derived elements reused, hubs used in up to 187 recipes | github.com/franziskabraendle/alchemy_empowerment and its Keeper data archive | Code MIT; archive states no license; the tables are content of the commercial game (Recloak) | fetch script only | 720 elements, 3,440 recipes, 371 KB | low (done) | **Selected.** The largest real combination graph; main Q3 source and an OR-heavy RecipeQuest source |
| Little Alchemy 1 (same archive) | Q1 (1.6 recipes per element), Q3 (depth 13) | same | same | fetch script only | 540 elements, 882 recipes, 109 KB | low (done) | **Selected** with version 2 (sparser OR) |
| Crafter | Q1: AND of materials, tools and stations, OR over placement tiles; Q2: 22 achievements forming a tech tree | github.com/danijar/crafter, PyPI `crafter==1.8.3` | MIT | data committed | 17 derived, 25 recipes, 2.6 KB | low (done) | **Selected.** The whole graph converts into one RecipeQuest world |
| Craftax | Q1, Q2 (Craftax-Classic reimplements Crafter; full Craftax adds dungeon floors, items and achievements) | github.com/MichaelTMatthews/Craftax | MIT | not selected (committable) | recipes live in JAX game code, no data table | high (transcription from code) | Use Crafter's table for the Classic tree; transcribe full Craftax only if a deeper Crafter-like tree is needed |
| MiniGrid / BabyAI unlock tasks | Q2 only weakly: a key opens a door that gives access to a box, chains of 2-3 steps | github.com/Farama-Foundation/Minigrid | Apache-2.0 (LICENSE file; the API reports NOASSERTION) | not selected | environment package | low to install, no recipe data | Too shallow for Q1-Q3; usable later as a navigation-heavy environment |
| DeepMind Alchemy, symbolic version | Q1 analogue: a hidden causal structure (potion effects, missing edges of a cube of stone states) resampled per episode, with a Bayes-optimal ideal observer | github.com/google-deepmind/dm_alchemy | Apache-2.0; repository archived | not selected | environment package | medium | Hidden transitions rather than AND recipes; a good external test of hidden-structure inference with an ideal-observer baseline, for a later study |
| TextWorld cooking games | Q1 weakly: a meal needs ingredients prepared in given ways | github.com/microsoft/TextWorld | MIT (LICENSE.txt) | not selected | game generator | medium | The recipe is written in the game's cookbook, so the preconditions are public |
| ScienceWorld | Q1 weakly (chemistry mixing, task dependencies) in a text interface | github.com/allenai/ScienceWorld | Apache-2.0 | not selected | environment whose simulator is written in Scala and runs on the JVM | high | A large text environment whose mixing tables are small; not a recipe-graph benchmark |
| WordCraft (Little Alchemy 2 in text) | Q1, Q3 | github.com/minqi/wordcraft | no license file (API: none) | not usable | small | n/a | Redistributes Little Alchemy 2 recipes without a license; the Little Alchemy fetch covers the same graph |
| MSGI Mining subtask graph (Sohn et al. 2020) | Q1 directly: hidden AND/OR subtask preconditions, inferred from interaction by MSGI's inductive logic programming; one OR | github.com/srsohn/msgi | MIT | data committed (JSON derived from the pickles) | 26 subtasks, 24 AND-nodes; task files 1.5-3.3 MB | low (done) | **Selected.** The closest published match to Q1, with published inference baselines |
| Craft world of policy sketches (Andreas et al. 2017) | Q1: ingredients plus a workshop; Q2: a bridge and an axe open the way to gold and gem | github.com/jacobandreas/psketch | Apache-2.0 | data committed | 11 derived, 590 bytes | low (done) | **Selected.** Small, maps directly onto RecipeQuest facilities, widely used |
| Freeciv technology trees | Q3: one or two prerequisites per technology (74 of 80 have two), depth 17, 85-90% reused; Q2: technologies unlock 114-119 units and buildings | github.com/freeciv/freeciv, tag `R3_2_6` | GPL-2.0-or-later | fetch script only | 87 technologies, about 200 elements per ruleset | low (done) | **Selected.** A real two-parent technology tree, the shape TechTree draws |
| Unciv technology tree | Q3, Q2 (lists of prerequisites, unlocked units and buildings) | github.com/yairm210/Unciv | MPL-2.0 | not selected (would be fetch-only) | JSON files | low | Redundant with Freeciv; add if a second tree with longer prerequisite lists is needed |
| Minetest Game crafting recipes | Q1: shaped recipes whose item groups act as OR | github.com/luanti-org/minetest_game (formerly minetest/minetest_game) | code LGPL-2.1, media CC BY-SA 3.0 (LICENSE.txt) | not selected (would be fetch-only) | recipes in Lua source | medium (Lua parsing) | A possible open crafting graph with real OR structure |
| Cataclysm: Dark Days Ahead recipes | Q1 strongly: a recipe is an AND over component groups, each group an OR list, plus tools and skills | github.com/CleverRaven/Cataclysm-DDA | CC BY-SA 3.0 (LICENSE.txt) | not selected (would be fetch-only, share-alike) | thousands of recipes in many JSON files of a 12 GB repository | high | The best large AND-of-OR source found; recommended as a later scale test |
| Minecraft item hierarchies (MineRL, MineDojo, minecraft-data) | Q2, Q3 (crafting chain to diamond tools) | github.com/minerllabs/minerl, github.com/MineDojo/MineDojo, github.com/PrismarineJS/minecraft-data | MineRL's LICENSE file is the CC BY-NC-SA 4.0 text; MineDojo code MIT; minecraft-data has no LICENSE file, its README states MIT; the recipes are content of a proprietary game | not usable | | | Crafter covers the same kind of chain under an open license |
| PDDL / IPC classical domains | Q1: STRIPS preconditions are conjunctions, and action-model learning recovers them from traces | github.com/AI-Planning/classical-domains, github.com/tomsilver/pddlgym | classical-domains: no license file; PDDLGym MIT, bundling domains of IPC origin whose licenses are not stated | not usable as committed data | | high (delete relaxation, grounding) | Revisit with a domain whose license is stated; macq (github.com/AI-Planning/macq, formerly QuMuLab/macq, MIT) generates traces and runs action-model learners |
| Combination lock, macro-operators | Q2 (constructions, docs/PRIOR_ART.md C10, C14) | | | not applicable | | | Constructions rather than data; covered by TechTree (study U) |

No game was scraped: every table above comes from a published repository or
data archive.

## Selected benchmarks

### Little Alchemy 1 and 2 (`little_alchemy/`, fetch only)

Two-element combination graphs from the data archive of Brändle et al.
(2023). Four base elements; every derived element is a pair of earlier
elements, with up to dozens of alternative pairs in version 2. Version 2's
conditional elements become unlocks (six after a number of discoveries, three
after k of a listed set). Deep (13 and 15 levels) and hub-dominated (one
element appears in 187 recipes). Used for Q3 statistics, for OR-heavy
RecipeQuest worlds (one per goal) and for TechTree worlds whose parents come
from the data. Limits: the elements' meanings matter to human players
(Brändle et al. show that empowerment, how many further elements an element
enables, guides them); agents here see only opaque types.

### Crafter (`crafter/`, committed, MIT)

The achievement tree of Crafter from `data.yaml`: collect (material plus
tools), place (items plus a target tile, one alternative per tile) and make
(items plus nearby stations). The whole graph fits one RecipeQuest world (16
items, 7 levels) after one promotion (`wood`). Navigation, survival, creatures
and quantities are not represented. `README.md` gives the package version for
running the environment itself.

### MSGI Mining (`msgi_mining/`, committed, MIT)

The subtask graph of the Mining domain (26 subtasks, AND-nodes with one OR:
`Light furnace` from firewood or coal), extracted from the MSGI repository's
pickles with a safe loader (`extract.py`). The object a subtask is executed
at becomes a base requirement, so every recipe has a hidden base part without
promotions. The whole graph has 23 derived subtasks, more than RecipeQuest's
16 item types, so worlds are built per goal. The train and evaluation task
files (subsets of subtasks with rewards) can be fetched and exported.

### Craft world of policy sketches (`psketch_craft/`, committed, Apache-2.0)

Eleven items: nine are made at three workshops from wood, grass and iron,
and gold and gem are collected once a bridge and an axe open the way.
Shallow (3 levels), no alternatives; maps directly onto RecipeQuest resources
and facilities. The ten benchmark tasks and their
sketches are in `hints.yaml`.

### Freeciv technology trees (`freeciv/`, fetch only, GPL-2.0-or-later)

The `classic` and `civ2civ3` rulesets of Freeciv 3.2.6: 87 technologies,
seven without prerequisites, the rest with one or two (plus a root
requirement where declared), 17 levels; technologies make 114 (classic) and
119 (civ2civ3) units and buildings available, represented as unlocks.

## Converters and their limits

**RecipeQuest** (`to_recipequest.py`). A world is the closure of one or more
goals. Base elements are base facts; derived elements are items. For each
recipe, requirements that are items become the public item inputs and
requirements that are base facts become the hidden true base subset, so what
is hidden is defined by the data. The pool is the true subset plus decoys
drawn from the world's other base facts (vocabulary base types without data
meaning only when the world has fewer base facts than the pool size). Changes
the conversion needs, all reported per world:

- RecipeQuest needs a nonempty hidden subset, so a recipe whose requirements
  are all items has its shallowest item requirement promoted to a base fact
  (gatherable in that world). In Little Alchemy and Freeciv most recipes
  combine only earlier discoveries (82-91%), so promotion keeps only 15-46% of
  a goal's original closure as items; the deeper part becomes base facts.
  Crafter needs one promotion, MSGI and the Craft world none.
- Recipes are dropped, each with its reason in `ConvertedWorld.dropped`, when
  they exceed the limit of two alternatives per item (the default; the
  shallowest are kept), when they would close a cycle, when they have more
  than three hidden (base) requirements and the item keeps another recipe
  (with none left the conversion fails), and when an alternative finds no
  public signature distinct from the item's other recipes: alternatives with
  the same item inputs need different pools, which a world with few base
  facts cannot always supply. Counted over the single-goal worlds of every
  goal, the last reason drops 4 alternatives in Crafter (the path-tile
  recipes of `table` and `furnace` in the `iron_pickaxe` and `iron_sword`
  worlds), 48 in Little Alchemy 1 and 138 in Little Alchemy 2.
- Unlocked elements are base facts (RecipeQuest has no counting unlocks), so
  they cannot be goals; quantities are ignored.
- The vocabulary has 16 item and 18 base types, so closures above that are
  refused (`ConversionError`): the whole of MSGI (23 items), Little Alchemy
  and Freeciv classic (17 items after promotion). `convertible_goals` lists the
  goals whose own closure converts: all of Crafter, the Craft world, MSGI,
  Little Alchemy 1 and both Freeciv rulesets, and 682 of the 715 derived
  elements of Little Alchemy 2. Of its 33 refusals, 8 are unlock-granted
  elements, which become base facts and so are not goals; 13 closures have
  more than 16 items and 12 more than 18 base facts.

Converted worlds are ordinary `World` objects: `make_task` builds tasks, the
reference solver solves every item of every tested world exactly, and
`public_view` hides the true subsets (tested with a canary pair).

**TechTree** (`to_techtree.py`). Always: a concept DAG (each derived element
with the requirements of its shallowest recipe as parents) and its reuse
statistics. When the data has the shape: a `TechWorld` whose concepts, parents
and levels come from the data, solvable by TechTree's reference and
environment. TechTree's rules are strict: a level-l concept needs two
distinct parents at levels l-1 and 1 (linear) or both at l-1 (doubling), and
its whole ancestry must obey them too. Only 3-9% of the derived elements of
Little Alchemy, Freeciv and MSGI fit (36% of the small Craft world, none of
Crafter); the best fit is Little Alchemy 1 under the linear rule with base
elements as level-1 concepts, 33 concepts in levels 2-6. The concatenation
order, hidden in TechTree, is fixed canonically, and alternatives are not
represented.

## How representative are the generators?

Statistics from `stats.json` and `generated_families.json` (F1 and F2: means
over 50 generated worlds and 200 tasks; TechTree: 50 worlds per
configuration).

**Graph structure.**

| Graph | Base | Derived | Recipes | AND arity mean / max | Recipes per derived (share with 2 or more) | Depth max (mean) | Recipes with only derived requirements | Derived reused | Uses of a derived element, mean / max | Shallowest derivation, mean size |
|---|---|---|---|---|---|---|---|---|---|---|
| F1 | 16.6 | 16 | 20.3 | 2.94 / 4.0 | 1.27 (0.27) | 8 (4.5) | 0 | 0.71 | 1.1 / 2.7 | 4.5 |
| F2 | 17.1 | 16 | 25.8 | 3.19 / 4.7 | 1.61 (0.61) | 8 (4.5) | 0 | 0.81 | 1.7 / 4.1 | 4.7 |
| Crafter | 10 | 17 | 25 | 2.20 / 5 | 1.47 (0.18) | 8 (4.1) | 0.24 | 0.59 | 2.1 / 10 | 4.4 |
| Craft (sketches) | 8 | 11 | 11 | 2.45 / 3 | 1.00 (0) | 3 (1.6) | 0 | 0.36 | 0.6 / 3 | 1.7 |
| MSGI Mining | 13 | 23 | 24 | 2.58 / 4 | 1.04 (0.04) | 8 (4.2) | 0 | 0.70 | 1.3 / 4 | 5.7 |
| Little Alchemy 1 | 4 | 536 | 882 | 1.96 / 2 | 1.65 (0.45) | 13 (7.2) | 0.88 | 0.52 | 3.0 / 81 | 13.4 |
| Little Alchemy 2 | 4 | 715 | 3,440 | 1.96 / 2 | 4.81 (0.92) | 15 (8.6) | 0.91 | 0.74 | 9.0 / 187 | 14.9 |
| Freeciv classic | 7 | 80 | 80 | 1.93 / 2 | 1.00 (0) | 17 (7.2) | 0.83 | 0.85 | 1.7 / 5 | 22.2 |
| Freeciv civ2civ3 | 7 | 80 | 80 | 1.93 / 2 | 1.00 (0) | 17 (7.4) | 0.83 | 0.90 | 1.7 / 4 | 25.1 |

**RecipeQuest tasks** (study R definitions: one task per converted world for
up to 40 sampled goals, hidden prerequisites, pools of 6).

| Source | Tasks | Optimal length | Hidden recipes | Alternatives needed | Entropy, bits | Headroom (brute force / optimum) | Budget | Promoted per world | Share of closure kept |
|---|---|---|---|---|---|---|---|---|---|
| F1 | 200 | 6.5 | 3.6 | 0.48 | 19.2 | 1.92 | 19.0 | 0 | 1 |
| F2 (pools of 8) | 200 | 11.0 | 9.3 | 2.99 | 60.8 | 1.88 | 27.1 | 0 | 1 |
| Crafter, per goal | 17 | 7.5 | 4.5 | 0.82 | 24.3 | 1.96 | 15.8 | 0.7 | 0.86 |
| Crafter, whole-graph world | 16 | 7.8 | 5.9 | 1.13 | 31.8 | 1.84 | 16.7 | 1 | 0.94 |
| Craft (sketches), per goal | 11 | 5.9 | 1.7 | 0 | 9.3 | 1.73 | 11.8 | 0 | 1 |
| MSGI Mining, per goal | 23 | 12.2 | 7.0 | 0.61 | 37.7 | 1.26 | 16.9 | 0 | 1 |
| Little Alchemy 1 | 40 | 7.5 | 8.6 | 2.43 | 46.2 | 1.76 | 18.8 | 6.1 | 0.24 |
| Little Alchemy 2 | 40 | 7.4 | 10.8 | 5.18 | 57.7 | 1.80 | 19.6 | 6.5 | 0.15 |
| Freeciv classic | 40 | 12.9 | 5.9 | 0 | 31.5 | 1.45 | 17.7 | 4.6 | 0.46 |
| Freeciv civ2civ3 | 40 | 10.8 | 4.6 | 0 | 24.8 | 1.51 | 15.8 | 4.2 | 0.41 |

**Concept DAGs (TechTree comparison).**

| Graph | Concepts | Levels | Parents per concept | Share of parents that are concepts | Reuse depth mean (max) | Reused as a parent | Children per concept mean (max) | Other parent of two-parent concepts: primitive / level 1 / level l-1 / between |
|---|---|---|---|---|---|---|---|---|
| TechTree L, d0 | 9 | 3 | 1.88 | 0.71 | 0 (0) | 0.58 | 1.33 (4.1) | 0 / 1 / 0 / 0 (linear rule) |
| TechTree L, d2 | 9 | 3 | 1.87 | 0.71 | 1.0 (2) | 0.58 | 1.33 (4.1) | 0 / 1 / 0 / 0 |
| Crafter | 17 | 8 | 2.29 | 0.72 | 3.1 (7) | 0.59 | 1.65 (7) | 0.80 / 0.20 / 0 / 0 |
| MSGI Mining | 23 | 8 | 2.61 | 0.50 | 3.2 (7) | 0.65 | 1.30 (4) | 1 / 0 / 0 / 0 |
| Little Alchemy 1 | 536 | 13 | 1.95 | 0.91 | 6.2 (12) | 0.42 | 1.77 (58) | 0.16 / 0.10 / 0.08 / 0.66 |
| Little Alchemy 2 | 715 | 15 | 1.92 | 0.90 | 7.4 (14) | 0.41 | 1.72 (64) | 0.19 / 0.06 / 0.07 / 0.69 |
| Freeciv classic | 80 | 17 | 1.93 | 0.89 | 6.2 (16) | 0.85 | 1.71 (5) | 0.07 / 0.15 / 0.14 / 0.63 |
| Freeciv civ2civ3 | 80 | 17 | 1.93 | 0.89 | 6.4 (16) | 0.90 | 1.71 (4) | 0.07 / 0.14 / 0.21 / 0.58 |

In a TechTree world reuse depth counts only compositional concepts, so d0
worlds have none although their parents gate them; in a recipe graph every
derived element is built from its parents. The "between" share is the
fraction whose other parent sits strictly between level 1 and level l-1.

What this says about the generators (structure only, no performance claim):

1. **Depth and reuse.** Real discovery graphs are much deeper than the
   generated families and reuse discoveries far more: mean reuse depth
   6.2-7.4 in Little Alchemy and Freeciv against at most 1 (maximum 2) in
   study L's worlds, and 3 in Crafter and MSGI. Study L's depth range (0-2)
   sits at the shallow end of what real graphs show.
2. **Where the hidden part lives.** Every generated recipe needs 1-3 hidden
   base facts. In Little Alchemy and Freeciv 82-91% of recipes combine only
   earlier discoveries, so the "hidden subset of a pool of base facts" form
   fits them only after promoting discoveries to base facts. Crafter and MSGI
   (once the execution object is a base fact) have the generators' form.
3. **OR structure.** The share of derived elements with alternative recipes
   orders the sources as Crafter 0.18, F1 0.27, Little Alchemy 1 0.45, F2
   0.61 and Little Alchemy 2 0.92 (4.8 recipes per element); MSGI (0.04) and
   Freeciv (0) have almost none.
   Converted Little Alchemy 2 tasks need 5.2 alternative recipes, more than
   F2 (3.0), the family in which the sampling walker did not break even.
4. **Arity and hubs.** Real recipes are smaller (about 2 requirements) than
   generated ones (about 3, one item plus 1-3 base facts), and real graphs
   have hubs (an element used in 81 or 187 recipes) that generators never
   produce (at most 6, typically 3-4: the largest hub is 4 in F1 and 6 in
   F2, and each world's largest hub averages 2.7 in F1 and 4.1 in F2).
5. **Headroom.** The brute-force headroom that gave the walkers room in study
   R (1.9 in F1 and F2) is similar in Crafter and Little Alchemy (1.8-2.0) but
   lower in Freeciv (1.5) and MSGI (1.3): there, sampling hypotheses can save
   at most a third (Freeciv) or a fifth (MSGI) of the brute-force steps.
6. **TechTree's level rules.** Mixed-level parent pairs ("between", 58-69%)
   dominate Little Alchemy and Freeciv, and a primitive second parent
   dominates Crafter and MSGI, while TechTree's linear rule always pairs the
   deeper parent with a level-1 concept. Only a few percent of real elements
   obey the rules with their whole ancestry.

## Recommendations

**Running the repository's agents on these benchmarks.**

- RecipeQuest walkers (study R arms `focused_sample`, `maximal`,
  `random_omit`): a converted world is an ordinary `World`, but the walker
  runner draws worlds from `TaskStream` and `WorldConfig`. The smallest
  addition is a stream that serves fixed converted worlds and draws each
  episode's goal among the world's items, played for 16 episodes per world as
  in study R. Good first sets: the Crafter whole-graph world (16 items, 7
  levels) and the Craft world (11 items) for recurrence within one world; MSGI
  per-goal worlds for deep chains with little headroom; Little Alchemy 2
  per-goal worlds as an OR-heavy family (the statistics predict F2-like
  behaviour, which the study can pre-register); Freeciv per-goal worlds for
  long chains without OR. Use several conversion seeds per goal (the seed
  changes pools and type assignment, not the hidden structure).
- TechTree arms (studies U and L): `techworld_from_graph` gives worlds with
  real parent structure (Little Alchemy 1, linear rule, base elements as
  level-1 concepts: 33 derived concepts over levels 2-6; `max_per_level`
  caps the width). The study runner builds worlds from configs, so it needs a
  world source that accepts these. Promote versus remember (study L) can then
  be measured on reuse depths up to 5 instead of 2.
- Report the conversion record with any result: promoted elements, dropped
  recipes and kept share are part of what was tested.

**Comparing with open-source baselines.**

- MSGI Mining: MSGI (github.com/srsohn/msgi, MIT) infers the same AND/OR
  graph from interaction with inductive logic programming and executes it
  with GRProp; its train/eval task files are fetched by `fetch.py
  --optional`. A fair comparison runs both on the same task files with the
  same episode budgets and scores (a) precision and recall of the inferred
  preconditions, which MSGI reports, against the walkers' posterior, and (b)
  steps or reward per episode.
- Action-model learning: macq (github.com/AI-Planning/macq, MIT) learns STRIPS
  preconditions from traces. Traces from converted worlds (as PDDL, the
  delete-free encoding is direct) give an established Q1 baseline for
  precondition recovery from the same evidence.
- Little Alchemy: the empowerment models of Brändle et al. (code MIT) and the
  participant data they share are a baseline for which combination to try
  next, a Q2/Q3 question the walkers do not address yet.
- Craft world: the modular sketch policies of Andreas et al. receive the
  sketches in `hints.yaml`; they are the reference for the skill library of
  phase P3 under the same supervision.
- Crafter: published scores (DreamerV2, PPO, Rainbow, random, human; in the
  Crafter repository's `scores/`) measure achievement success in the pixel
  environment over 1M steps and are not comparable with the converted
  symbolic world; a comparison needs an agent acting in `crafter==1.8.3` or
  a symbolic Crafter.

## Not tracked

Downloaded files go to `benchmarks/<name>/data/`, which the rule
`benchmarks/*/data/` in the repository's `.gitignore` keeps out of Git (a
test checks the rule). Nothing in this folder needs a
package beyond the repository's dependencies; `extract.py` uses torch and
numpy.
