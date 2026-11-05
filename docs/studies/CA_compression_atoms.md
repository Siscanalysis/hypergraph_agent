# Study CA: atoms from compressing the agent's own press sequences

Status: protocol frozen on 2026-11-05 by the commit that freezes this file, before any measurement
run, after an independent review (ready to freeze after its required changes S1 to S5; its decisions
are recorded under Implementation); the measurement runs from a clean worktree of that commit. This
is experiment RB-B of docs/studies/RB_rule_builder_design.md (roadmap item 6i, idea I3), a follow-up
of study L. It adds an opt-in generator option, `latent`, whose default leaves every existing world,
task, reference plan, run hash and agent decision unchanged (Implementation). Development used only
the allocation `dev` of `runs/CA-ledger.json` and the namespace `ca_dev`; its outcomes were seen
before the predictions below were written and are reported under Implementation as calibration, not
results. The measurement namespace (`techtree_ca`) has never been used.

## Question and claim

Question: in study L a discovery is an exact event (a firing reveals a concept's sequence), and
`promote` makes every discovered concept an atom of new hypotheses, which cut late-half cost by about
two thirds where discoveries are built from earlier ones (reuse depth 2). Can atoms proposed by
compressing the agent's own press sequences, successful and failed, replace those exact discovery
events? And does a rule learner's cost follow the description length of the hidden composites at
its action layer (primitives; primitives and discovered atoms; atoms and compressed fragments)?

CA keeps L's engine, memory, evidence and replay planner and changes only where the atoms come from.
`promote_stitch` and `promote_bpe` use the fragments that a compressor (Stitch; byte-pair encoding)
finds in the agent's own press sequences, recompressed between episodes. The `latent` option builds
worlds whose higher concepts share hidden fragments that are never concepts: no discovery event can
make one an atom, but they recur in what the agent presses.

Claim under test (one primary): on latent worlds at reuse depth 2, `promote_stitch` takes fewer
actions per episode than `promote` in the late half of each world's stream (episodes 6-11).
Falsification: equivalence within 10% of `promote`'s mean. Conditions for a pass: the compressed
atoms help beyond plain remembering (C1), and on study L's worlds at depth 2, where every atom a
search needs is a discovered concept, compression costs no more than exact discovery within 10% (H1,
the harm check). A benefit on latent worlds is expected by construction (the design note, Section 5):
`promote`'s candidates never contain a latent composite. CA1's comparator is `promote`'s two-atom
rule, not exact discovery in general: in development the exact learner with a richer hypothesis order
(`enum_atoms`) matched the oracle library at latent depth 2 (11.9 against 11.9 late actions). The
informative parts are the harm check, the comparison of the two compressors, the controls
(`remember`, `sham_promote`) and the layer factor: the same enumerating rule learner over primitives
(`remember`, identical to it by construction), over primitives and discovered atoms (`enum_atoms`),
and with compressed fragments added (`enum_frag`), next to every world's description length of its
composites at each layer.

The study measures hand-written elimination agents in a toy world. It says nothing about learning
agents, and nothing about compression beyond the two compressors and the declared settings.

## Relation to earlier work

- Study L (docs/studies/L_layered_discovery.md) supplies the layers, the arms `remember`, `promote`
  and `sham_promote`, the reuse depth, the endpoint and the precedence verdict; study U the pooled
  explorer and the environment's validity; study P (docs/studies/P_imperfect_skills.md) asked
  whether promotion still pays when promoted atoms are executed by imperfect learned skills. CA
  keeps exact execution and changes the source of the atoms. Study HP2
  (docs/studies/HP2_schema_priors.md) added an opt-in generator option to RecipeQuest the same way:
  off by default and shown inert by frozen-behaviour digests.
- docs/PRIOR_ART.md, Section 8.4, positions promotion as library learning (K6, DreamCoder; K8,
  options learned by compression), chunking (K9) and macro-operators (C14), with the utility problem
  (K11) as its known cost; K12 (substructure discovery by minimum description length) builds a
  hierarchical description of a graph by replacing recurring substructures. The layer factor's
  description length, the fewest tokens of an alphabet that spell a sequence, takes from E11
  (Rissanen, 1978; Section 12) only the idea of the shortest description: the token count charges
  nothing for the alphabet or the model, unlike E11's two-part description length. CA replaces exact
  discoveries by a compression of the agent's own experience and measures the utility cost directly
  (the harm check and `sham_promote`).
- docs/PRIOR_ART.md, Section 14, registers the closest work for I3 (LC1 to LC6). Stitch (LC1, Bowers
  et al., POPL 2023) finds by top-down search the abstractions that most compress a corpus of
  programs and is the tool used here (stitch-core 0.1.29). LC2 (Yunis et al., NeurIPS 2024, "Subwords
  as Skills: Tokenization for Sparse-Reward Reinforcement Learning") applies byte-pair encoding to
  the discretized actions of demonstrations and uses the resulting skills as the action space of
  online reinforcement learning; PRISE (LC3, Zheng et al., ICML 2024) applies it to quantized action
  codes of offline data and uses the skill tokens in behaviour cloning, without reward, which is not
  reinforcement learning. Sequitur (LC4, Nevill-Manning and Witten, JAIR 7, 1997) infers from a
  sequence a hierarchical structure of rules that generate only the input (not strictly a grammar,
  as its authors note). Byte-pair encoding itself is LC5 (Gage, 1994, as a data-compression
  algorithm; its end page and DOI are unconfirmed) and LC6 (Sennrich, Haddow and Birch, ACL 2016, as
  subword segmentation), whose form CA follows: the merges are the vocabulary and no pair crosses
  the boundary of a sequence. LC2 and LC3 compress demonstrations or offline data gathered apart from
  the learner; CA compresses the agent's own successful and failed press sequences within one world,
  between its episodes, uses the fragments as atoms of a search rather than as actions, and compares
  them with exact discovery events in one engine.
- LC7 (Section 14) is the closest precedent: Lange and Faisal (CCN 2019; arXiv 1907.12477) treat an
  agent's own on-policy rollouts as sentences, infer action grammars from them (from successful
  episodes in the online version) and flatten the rules into macro-actions for a value-learning
  agent. CA, too, compresses the agent's own experience; it uses the result as atoms of new
  hypotheses and measures it against exact discovery events.

In the parts read for Section 14, docs/PRIOR_ART.md records no controlled comparison, in one engine,
of atoms compressed from an agent's own experience with atoms taken from exact discovery events, with
a harm check where the discoveries suffice and a random-atom control; that is what this study adds.
It claims no new method.

## Protocol (frozen)

### Latent worlds (the generator option)

`world.latent` in a TechTree config sets `TechWorldConfig.latent`; every config that does not give
the key builds exactly the worlds it built before and never reaches the latent code (Implementation).
The composites of a world are its concepts of levels 2 and 3, compositions and flat concepts alike.
With `latent: 2`:

- Fragments: two distinct base sequences of length 2 that are not level-1 combos, so no concept's
  sequence equals one and no firing can reveal one as a concept.
- Compositions: a compositional concept keeps the sequence of one parent (at level 3 its level-2
  parent, the reused discovery; at level 2, where both parents are at level 1, one drawn uniformly)
  and a fragment takes the place of the other parent's sequence, in the hidden concatenation order.
  Gating is unchanged: both parents must be held.
- Sharing: fragments, kept positions and the assignment of fragments to concepts are drawn from their
  own stream for every concept of levels 2 and 3 whatever the reuse depth, every fragment for at
  least two of them, and redrawn until every level's latent compositions are distinct.
- Flat concepts of a latent world also avoid the concatenation of a concept and a fragment (study
  L's rule (ii), extended), so only compositions are spelled by two such pieces.
- Consequences at depth 2: a level-2 concept is a combo and a fragment, a level-3 concept its level-2
  parent's sequence and a fragment; no composite is the concatenation of two concepts, so `promote`'s
  candidates never contain one. Over primitives and lower concepts a level-3 composite takes 3 tokens
  (2 in study L's worlds at depth 2), and 2 once the fragments are tokens. At depth 1 only level 2 is
  built from fragments, and level 3 is flat.
- The public view is unchanged: agents cannot tell a latent world from another (tested).
- The option needs the linear length rule, no unlocks, reuse depth 1 or more, and at most half as
  many fragments as concepts of levels 2 and above (validated).

### Settings

| Setting | Variant | Namespace | Seeds x worlds | Episodes | Late half | Role |
|---|---|---|---|---|---|---|
| L layers, depth 0 | `d0` | `techtree_ca` | 0-7 x 4 (32) | 12 | 6-11 | no structure to compress |
| L layers, depth 1 | `d1` | `techtree_ca` | 0-7 x 4 (32) | 12 | 6-11 | secondary |
| L layers, depth 2 | `d2` | `techtree_ca` | 0-7 x 4 (32) | 12 | 6-11 | harm check |
| latent, depth 1 | `latent_d1` | `techtree_ca` | 0-7 x 4 (32) | 12 | 6-11 | secondary |
| latent, depth 2 | `latent_d2` | `techtree_ca` | 0-7 x 4 (32) | 12 | 6-11 | primary |

All five settings are variants of `configs/compression/ca_main.yaml`. Common to all: study L's
layers (3 base primitives; three levels of three concepts with public lengths 2, 4 and 6; a level-l
concept has hidden parents at levels l-1 and 1 and fires only while both are held); the goal a
level-3 concept, uniform; a budget of 100 actions per episode; 12 consecutive episodes per world;
nothing supplied (study L's phase L2). Reuse depth d: concepts of levels 2..d+1 are compositions,
the others flat.

One namespace for all five settings. Worlds depend on (namespace, seed, world index) and the world
parameters, and the latent draw neither consumes the other streams nor rejects an attempt that the
plain world accepts unless no latent draw is valid, so the settings share world keys, parents,
orders and level-1 combos world by world (coupled worlds, as across study L's depths; 28 or more of
30 tested seeds are coupled).

### Arms

| Arm | Kind | Atoms of new hypotheses, search order |
|---|---|---|
| `promote` | adaptive | Study L's: the known concepts (pairs, then primitive windows) |
| `promote_stitch` | adaptive | The fragments Stitch finds in the agent's corpus, same rule |
| `promote_bpe` | adaptive | The fragments byte-pair encoding finds in it, same rule |
| `sham_promote` | adaptive | Study L's control: random base sequences of known lengths |
| `remember` | adaptive | Study L's: none, primitive windows only (the layer-0 learner) |
| `enum_atoms` | adaptive | Layer 1: windows by description length over primitives and concepts |
| `enum_frag` | adaptive | Layer 2: the same with Stitch's fragments added |
| `random` | reporting | Uniform primitives, submits when the goal is held |
| `oracle_library` | reporting, privileged | Every concept supplied (study L's anchor); reports lengths |
| `reference` | reporting, privileged | Replays the evaluator's shortest plan |

`promote`'s rule, shared by the two compression arms: a search at level l first tests the
concatenations of ordered pairs of distinct atoms whose lengths add up to the level's length, then
primitive windows. `sham_promote` has as many random base sequences as `promote` has known concepts,
each of a known concept's length.

Every setting runs `remember`, `promote`, `promote_stitch`, `promote_bpe`, `enum_atoms`,
`enum_frag`, `oracle_library` and `reference`; `random` runs at depth 2 and latent depth 2,
`sham_promote` at latent depth 2. Study U's `pooled` is the same agent as `remember` in every world,
and the enumerator at layer 0 (`enum_prim`) acts as `remember` too; the factory builds both and their
identical rows are tested, so neither is run.

All adaptive arms share study L's engine (`techtree/explorers.py`): sure gating, exact firings,
cross-episode memory per world, the replay planner and the pooled walk. They differ only in the
hypothesis sets offered to a new search. Common random numbers: every arm of a seed starts from the
same agent seed and per-world symbol preference.

### Corpus and compressors (definitions)

Corpus (agent parameter `corpus: tests`). Every compression arm keeps, per world, the press
sequences of its own that its evidence counts as tests:

- successful press sequences: the sequence that fired, one per firing the agent observes
  (discoveries, replays and incidental firings alike: the trace suffix of the fired concept's
  public length);
- failed press sequences: every window that the engine's evidence newly marks as tested without
  firing during the episode (its eliminations, at every level), and every replay that did not fire
  its target (the replayed sequence, once per new failure clause).

Sequences of fewer than two presses are dropped. The compressors see the sequences only, neither
which concept fired nor whether a sequence succeeded. The corpus persists across the episodes of a
world and is dropped with it, as the engine's memory is. The arms record their presses through a
wrapper of their own public environment handle (`compression.PressLog`) and read the engine's
evidence through its public attributes.

The corpus `tests` was chosen after the outcomes of the corpus `firings` were seen (Implementation,
development record). `firings` holds almost no failed press sequences (failed replays are rare), so
it does not implement "successful and failed". The independent review's in-memory check found that
the most literal reading, each episode's whole press sequence (`episodes`), gave gains similar to
those of `tests` on latent worlds and more harm on study L's worlds, so `tests` does not favour a
pass.

Compression between episodes. At the start of every episode whose world corpus has grown since the
last compression, the arm compresses the whole corpus and replaces its library of fragments; the
library is fixed during an episode (a sleep phase, as in wake-sleep library learning). Compression
therefore needs a fragment to recur, and it is one episode late.

- Stitch (stitch-core 0.1.29): each sequence written as a chain, `(a1 (a0 (a2 e)))` for the
  presses 1, 0, 2; abstractions of arity at most 1 with no metavariable or body in function
  position (`--no-curried-metavars`, `--no-curried-bodies`), so an abstraction is a contiguous
  fragment followed by a hole (the rest of a sequence) or by the end marker, possibly through earlier
  abstractions, which are inlined; Stitch's default single-task pruning (one task per corpus
  sequence and `allow_single_task` off, so an abstraction must occur in at least two corpus
  sequences, identical sequences counting separately) and default costs; at most 50 abstractions.
- Byte-pair encoding (`compression.bpe_fragments`, 20 lines): at most 50 merges of the most
  frequent adjacent token pair, counted over the corpus, ties broken by the smaller pair, stopping
  when no pair occurs twice; every merged token is a fragment.
- Fragments of length 2 to 6 (the longest public length) are kept, each once, in the order found.
- Both caps bind with this corpus and are declared vocabulary sizes, not stopping rules: Stitch
  returned 50 abstractions in 40 of the 44 compressions of each development run the review replayed,
  and byte-pair encoding made 50 merges in 319 of the 480 development episodes. After one-press and
  repeated fragments are dropped, the effective vocabularies were about 35 fragments for Stitch (at
  most 40) and about 48 for byte-pair encoding. The rows record each compression's effort
  (`ca_compress_effort`, the abstractions returned or the merges made, and `ca_compress_cap`), and
  `row_summaries.libraries` reports the share of episodes whose library came from a compression at
  the cap (descriptive).

The alternative corpora `firings` (successful press sequences and failed replays only) and
`episodes` (each episode's whole press sequence, split at `wait` and `submit`) are implemented as
agent parameters and reported under Implementation; they are not measured.

### Layer factor and description lengths

The description length of a sequence over an alphabet is the fewest tokens whose concatenation
spells it; single primitives are always tokens. For every composite c of a world, from the links the
runner supplies to `oracle_library` (`compression.LayerOracle`):

- layer 0: the length of c's sequence (primitives);
- layer 1: over the primitives and the sequences of the concepts of lower levels (the atoms exact
  discovery supplies once the lower levels are known);
- layer 2: the same with the fragments that Stitch, with the measurement's settings, finds in the
  world's own concept sequences added, except a fragment equal to c's sequence: Stitch's greedy
  result on the world's own sequences, each seen once;
- a fourth value, the parent residues: the same as layer 1 with the residues added. A residue is the
  part of a concept's sequence that one parent's sequence leaves at the other parent's place, when it
  is no concept's sequence (from the sequences and parents in the links). At latent depth 2 the
  residues are exactly the hidden fragments (tested); elsewhere a flat concept that starts or ends
  with a parent's sequence by chance adds a residue.

The enumerating learner at layer L orders, before every search, the open windows of the level by
their description length over its own alphabet (layer 0: primitives; layer 1: and the known
concepts; layer 2: and its library's Stitch fragments) and tests the windows of the smallest value by
the engine's pooled walk.

`row_summaries.layers` of the analysis reports per setting the mean description lengths of the
worlds' composites by level and layer (the four values above); per layer learner and world the
description length of the world's level-3 composites at the learner's layer next to its actions to
the first success and its late and overall actions per episode; and Spearman rank correlations of
that description length with each cost over every world and setting, per learner and pooled.
`row_summaries.libraries` reports, per setting and compression arm, the library at the end of each
world: its size and the shares of its fragments that are one of the world's concept sequences, that
are a contiguous part of at least two of them without being one, and the rest; and the cap share
above. Every description length is descriptive: no test reads it.

### Endpoints

Per world (the unit), per arm: `steps_late`, the mean actions per episode over episodes 6-11
(failures count their whole budget); `steps_all`; `first_success_steps` (actions in the world up to
and including its first successful episode, censored at the world's total); success rates and cost
ratios early and late; discoveries; the row summaries above; interactions from the ledger.

### Primary contrast, falsification and verdict

CA1 (primary), latent depth 2, 32 worlds: `promote_stitch` - `promote` in `steps_late`, paired by
world; holds when the upper end of the 95% world-cluster bootstrap interval is below 0. F1
(falsification by equivalence): the 90% interval of the same difference lies within plus or minus
10% of `promote`'s mean at latent depth 2.

Conditions for reading CA1 as a replacement of discovery events:

- C1, latent depth 2: `promote_stitch` - `remember` has an upper end below 0. Without it CA1 could
  hold with no value from compression: in development `promote` (36.6 late actions) was dearer than
  `remember` (33.8) at latent depth 2, its two-atom candidates being futile there.
- H1 (the harm check), depth 2: the upper end of the 95% interval of `promote_stitch` - `promote` is
  at most 10% of `promote`'s mean (non-inferiority where every needed atom is a discovered concept).

Verdict (`techtree.analysis.verdict`, rule `precedence`), applied in this order:

0. If CA1, F1 or a validity test has no verdict because a run it reads is incomplete, or the runs of
   the measurement come from more than one code identity (checked by hand before the analysis,
   Commands), CA has no verdict; if CA1 holds without F1 and C1 or H1 has no verdict, CA has no
   verdict.
1. A failed validity test fails CA ("fail (validity)").
2. CA1 holds and F1 does not: "pass" if C1 and H1 hold, otherwise "confounded", reported with the
   failed condition: "confounded (H1)" reads that compression helps where discoveries miss structure
   and costs where they do not; "confounded (C1)" that the gain is only `promote`'s futile
   candidates.
3. CA1 and F1 both hold: "negligible".
4. F1 holds and CA1 does not: "falsified" (compressed atoms do no better than exact discoveries on
   latent worlds).
5. Neither holds: "inconclusive".

Power notes. The development per-world SD of the CA1 difference was 24.0; with 32 worlds, F1 (plus
or minus 3.66 in development) can hold only if that SD is below about 12.6 (1.645 SD / sqrt(32)
within the margin at a true difference of 0), so "falsified" and "negligible" are practically
unreachable and a null result reads "inconclusive". H1's development SD was 9.0 against a margin of
about 0.97; H1 holds at zero true harm only if the SD is below about 2.8 (1.96 SD / sqrt(32) within
the margin), so a pass needs `promote_stitch` to act like `promote` in nearly every world at depth 2.

### Validity tests

V1: the reference solves every task; V2: the oracle library solves every task; V3: the oracle library
never uses fewer actions than the reference (every setting). A failed validity test fails CA.

### Secondary tests (each with its own criterion, no multiplicity correction)

"Upper end" and "lower end" are those of the 95% interval of the paired difference; the harm margins
are 10% of the named arm's mean.

| Test | Setting | Contrast (paired by world) | Metric | Holds when |
|---|---|---|---|---|
| S1 | latent d2 | `promote_bpe` - `promote` | late actions | upper end below 0 |
| S2 | latent d2 | `promote_stitch` - `promote_bpe` | late actions | the interval excludes 0 (two-sided) |
| S3 (harm) | d2 | `promote_bpe` - `promote` | late actions | upper end at most 10% of `promote`'s mean |
| S4 | latent d2 | `promote_stitch` - `sham_promote` | late actions | upper end below 0 |
| S5 | d0 | `promote_stitch` - `promote` (no benefit) | late actions | upper end not below 0 |
| S6 | latent d1 | `promote_stitch` - `promote` | late actions | upper end below 0 |
| S7 | latent d2, d2 | interaction of `promote_stitch` - `promote` | late actions | upper end below 0 |
| S8 | latent d2 | `promote_stitch` - `promote` | actions, whole stream | upper end below 0 |
| LF1 | d2 | `enum_atoms` - `remember` | late actions | upper end below 0 |
| LF2 | latent d2 | `enum_atoms` - `remember` | late actions | upper end below 0 |
| LF3 | latent d2 | `enum_frag` - `enum_atoms` | late actions | upper end below 0 |
| LF4 | d2 | `enum_frag` - `enum_atoms` | late actions | upper end at most 10% of `enum_atoms`'s mean |
| LF5 | d0 | `enum_atoms` - `remember` (no benefit) | late actions | upper end not below 0 |

S2 is two-sided: the config holds it as two tests on the same 95% interval, S2a (upper end below 0:
Stitch cheaper) and S2b (lower end above 0: byte-pair encoding cheaper), and S2 holds when either
does, reported with its direction. The two compressors' effective vocabularies differ (about 35 and
about 48 fragments; Corpus and compressors), which colours S2. Power of the margin tests: S3's
development SD was 10.6 and LF4's 2.2, against margins of about 0.97 and 0.99, which hold at zero true
difference only if the SD is below about 2.8 and 2.9; S3 can hold only if byte-pair encoding acts like
`promote` in nearly every world, while LF4 is within reach.

Thirteen tests with a criterion: if none of their hypotheses were true and the tests were
independent, the chance that at least one holds anyway would be about 1 - 0.95^13, or 0.49; the
verdict rests on the primary and its conditions alone. Descriptive, read by no test: every arm's
table; the layer and library summaries, including the Spearman correlations and the cap shares; the
cost and discovery curves (the TechTree figures); `random` (the null at depth 2 and latent depth 2).

### Predictions (written after the development checks)

Written on 2026-11-05, after the two development passes of 2026-10-29 (eight worlds per setting,
outcomes seen; Implementation). Eight worlds give wide intervals, so the ranges below are wide.

- Verdict: "confounded (H1)" (CA1 holds, F1 does not, C1 holds, H1 fails). CA1 estimate between -6
  and -20 actions (development -14.4 [-29.7, 1.0]; six of seven untied worlds lower); C1 between -4
  and -15 (development -11.5); H1 estimate between +3 and +10 against a margin of about 1 (development
  +8.0 [2.4, 14.1]). Reason: with the failed tests in its corpus, Stitch's library holds about 35
  fragments, about two thirds of them neither concepts nor parts shared by two concepts; on latent
  worlds the shared fragments it does find put the composites among its two-atom candidates, which
  `promote`'s never contain; on study L's worlds the same spurious fragments come first and delay
  the searches (first success at depth 2 after 228 actions against 28 for `promote` in
  development). The next likeliest outcome is "inconclusive" (CA1 narrowly missing, as in
  development).
- S1 holds, estimate between -5 and -18 (development -11.6); S2 does not hold, estimate between -8
  and +8 (development -2.7 [-14.9, 9.4]); S3 does not hold, estimate between +2 and +9 against about
  1 (development +5.0); S4 holds, between -5 and -20 (development -13.5); S5 holds (no benefit at
  depth 0; estimate between 0 and +15, development +9.5); S6 does not hold, between -5 and +15
  (development +7.8); S7 holds, between -10 and -30 (development -22.4); S8 holds narrowly or not,
  between -3 and -12 (development -9.1 [-22.0, 2.4]).
- LF1 and LF2 hold: `enum_atoms` stays near the oracle library at depth 2 and latent depth 2 (late
  actions about 10 and 12; development -18.7 and -21.9 against `remember`): over primitives and
  discovered atoms a latent composite takes only one token more than study L's composites. LF3 does
  not hold, `enum_frag` dearer by +5 to +20 (development +15.2, all eight worlds): its corpus fills
  the cheap tiers with spurious fragments ahead of the 3-token words that hold the latent
  composites. LF4 does not hold, narrowly (estimate between 0 and +3 against about 1; development
  +0.8 [0, 2.4]). LF5 holds (development +7.3).
- Layer factor, descriptive: the layer-1 description length ranks `enum_atoms`'s costs over worlds
  and settings closely (Spearman about 0.9; development 0.90 for actions to the first success),
  mostly through the contrast between depth 2 and the flat settings; the layer-2 length (Stitch's
  greedy result on the world's own sequences) ranks `enum_frag`'s costs less well (0.6 to 0.75;
  development 0.72), because the learner's own library is not that result. The parent-residue length
  of a level-3 composite is 2 at latent depth 2 in every world.
- V1, V2 and V3 hold.

### Statistics

The unit is a world; worlds are independent across seeds and coupled across settings. Contrasts
are paired by world (the interaction S7 across settings, since the worlds are coupled); 10,000
percentile bootstrap resamples of worlds with seed 20261008, 95% intervals (90% for F1); an exact
two-sided sign test over worlds is reported with each interval (`techtree.analysis`). Spearman
correlations are descriptive.

### Budget and runtime

A run's cap is the sum of its 48 episode budgets (4,800), so no run is cut short. From the dry runs:

| Config | Runs | Adaptive cap total | Reporting cap total | Interactions |
|---|---|---|---|---|
| `ca_main.yaml` (measurement) | 344 | 1,190,400 | 460,800 | expected about 615,000 and 115,000 |
| `ca_dev.yaml` (first development pass) | 86 | 297,600 | 115,200 | used 147,366 and 28,755 |
| `ca_dev_tests.yaml` (second development pass) | 30 | 144,000 | 0 | used 75,661 and 0 |

The expected use scales the development use of the measurement's design (eight worlds per setting)
to 32 worlds. Ledger `runs/CA-ledger.json`, declared identically in the three configs (tested) and
created by the first development run: allocations `dev` 300,000 and `ca_main` 1,190,400 adaptive
(the measurement's adaptive cap total), adaptive cap 1,490,400 (their sum), reporting cap 700,000.
After development it holds 223,027 adaptive (`dev`) and 28,755 reporting interactions; 76,973 `dev`
interactions remain. The expected use leaves about 575,000 adaptive interactions of `ca_main` for
deterministic reruns.

Runtime: one process on one CPU core (Stitch runs single-threaded), with a peak working set of about
240 MB in the timing check of Implementation. From the development rates, Stitch with the
measurement's corpus took 13.6 to 51.8 s per run of four worlds (`promote_stitch` 13.6 to 38.7 s,
`enum_frag` 1.6 to 51.8 s), byte-pair encoding 1.6 to 4.3 s, and every other arm at most about 2 s;
the second development pass (30 runs) took 9 min 51 s and the first (86 runs) 3 min 15 s. The
measurement's 344 runs (80 of them with Stitch on this corpus) should take about 45 minutes, up to
about 75 under load. It runs at night.

### Stopping and rerun rules

Every run plays its whole stream; no cap can bind. The measurement config runs from one snapshot, as
one process: never two processes at once, because duplicate complete runs make the analysis refuse
(`techtree.analysis.select_runs`). The measurement command carries `--resume`, which does nothing on
a first run and, after an interruption, repeats only missing or incomplete runs; it is refused when a
run to repeat was made by other code, and the analysis uses a rerun only when it has the same run
hash and the same code identity. Its output is appended to a log file, so a resume keeps the
first run's log. Before the analysis, the code
identity (loaded package code and commit) of every run's manifest is checked by hand to be one
(Commands); with more than one, CA has no verdict. The analysis runs once, by hand, after every run
has been checked complete. No interim analysis; no change of arms, parameters, settings, endpoints or
criteria after any measurement row exists, other than dated amendments. A fault that needs a code fix
means a new freeze and a fresh measurement from the new snapshot under a dated amendment, with a new
run name (the run hash does not include the code, so the old name would collide with the old runs)
and a logged ledger amendment first if the `ca_main` allocation cannot cover it. A stale
`runs/CA-ledger.lock` older than a minute may be removed (a live holder keeps it for milliseconds).

### Commands (from a clean snapshot, after review and freeze)

From the repository root, which holds `runs/` (and the ledger) and `artifacts/`; `S` is a clean
worktree of the freeze commit. The preflight checks stitch-core's version before the dry run and
before any resumed run, and the results' write-up records the version:

```bash
git worktree add ../snapshots/CA-<hash> <hash>
S=../snapshots/CA-<hash>
.venv/Scripts/python.exe -c "import importlib.metadata as m; assert m.version('stitch-core')=='0.1.29'"
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.techtree \
    --config $S/configs/compression/ca_main.yaml --dry-run
# at night, one process only (about 45 minutes); after an interruption: the preflight, then the same
PYTHONPATH=$S/src PYTHONUNBUFFERED=1 .venv/Scripts/python.exe -m hypergraph_agent.techtree \
    --config $S/configs/compression/ca_main.yaml --allow-measurement --resume >> runs/ca-main.log 2>&1
# one code identity across the runs
PYTHONPATH=$S/src .venv/Scripts/python.exe -c "from pathlib import Path; \
from hypergraph_agent.techtree.analysis import collect_runs; runs = collect_runs(Path('runs'), 'ca-main'); \
codes = {tuple(r['code']) for r in runs}; print(len(runs), 'runs,', sum(r['complete'] for r in runs), \
'complete; code identities:', codes); assert len(codes) == 1"
# once, after every run has completed
PYTHONPATH=$S/src .venv/Scripts/python.exe -m hypergraph_agent.techtree.analysis \
    --config $S/configs/compression/ca_main.yaml --runs runs --out artifacts/techtree/ca-main
```

The analysis writes `artifacts/techtree/ca-main/analysis.json` (with `row_summaries.layers` and
`row_summaries.libraries`) and the two figures, and prints every test and the verdict.

### Limits stated in advance

- Gains by construction: the latent option is built for this question, so a benefit of compression
  on latent worlds at depth 2 is expected. CA1's comparator is `promote`'s two-atom rule: the latent
  composites are one token longer than study L's over primitives and discovered concepts, and the
  richer exact learner `enum_atoms` already matched the oracle library at latent depth 2 in
  development. The harm check, the controls and the layer factor carry the information.
- Power: F1 and H1 need per-world SDs far below the development values (Primary, power notes), so
  "falsified" and "negligible" are practically unreachable and H1 passes only if `promote_stitch`
  acts like `promote` in nearly every world at depth 2; the margin test S3 is limited the same way.
- Every test reads late actions (S8 the whole stream), so the early cost of compression at depth 2 is
  judged by no test: in development `promote_stitch` first succeeded after 228 actions against 28 for
  `promote`.
- `promote` uses whole discovered concepts as atoms. An exact learner that subtracted known parents
  from discovered sequences would read the latent fragments off the discovery events directly; it is
  not implemented (the oracle's parent residues do this, descriptively).
- The compressors see the content of the discovery events (the sequences that fired) next to the
  failed tests; what they replace is the rule that every discovery is an atom, not the information
  that the events carry.
- In the corpus, consecutive tested windows overlap, so the agent's own walk passes Stitch's
  single-task pruning (its parts recur across sequences); successes recur once per firing while
  failures enter once per world, so frequency carries the outcome.
- Two compressors with one declared setting each and one corpus definition are measured, and the
  corpus was chosen after the first development pass. The corpus decides much: without failed tests
  Stitch's library was the concepts themselves (no gain, no harm), and a Stitch criterion requiring
  reuse across distinct sequences (identical sequences counting once), or other budgets, could rank
  the arms differently.
- Both caps bind with this corpus; they are declared vocabulary sizes, and the effective vocabularies
  differ between the compressors (about 35 and about 48 fragments).
- The failed tests grow with search effort, so compression costs most where the search is longest
  (depth 0 and latent depth 1), and Stitch's runtime grows with the corpus.
- The layer-2 description length is Stitch's greedy result on the world's own sequences; it can miss
  a fragment that is not shared by two concepts (latent depth 1) and is not the learner's own library.
  The parent residues are exact at latent depth 2 only. Both are descriptive.
- The results are tied to stitch-core 0.1.29 (checked before every run of the measurement).
- One code identity across all runs is checked by hand before the analysis (Stopping and rerun rules).
- The development calibration of the measurement's design combines the second pass's compression
  arms with the first pass's anchors, which ignore the corpus (tested); the measurement runs every
  arm in one config.
- Only hand-written elimination agents are tested; nothing is claimed about learning agents.

## Implementation

- `src/hypergraph_agent/techtree/generator.py` (the `latent` option only): `TechWorldConfig.latent`
  and its validation, the hidden fields `TechWorld.fragments` and `TechWorld.fragment_of` (empty by
  default), `LatentDraw`, `_latent`, study L's rule (ii) extended in `_flat` (an extra argument whose
  default leaves the rule as it was), and the fragments in `debug_view` of latent worlds. With
  `latent` 0, the default, no code path that generated a world before is changed.
- `src/hypergraph_agent/techtree/config.py`: study `CA` in `STUDIES`, and `OPTIONAL_WORLD_KEYS`, which
  lets a config give `world.latent` without adding it to `DEFAULTS`; the key is absent from a loaded
  config unless the config gives it, so every earlier config loads to the same dictionary and keeps
  its run hashes (tested). This goes beyond registering the study, as HP2's optional keys did in the
  walker's `config.py`, because otherwise the option could not be reached from TechTree configs; the
  review approved it.
- `src/hypergraph_agent/techtree/study.py`: study `CA` in `STUDY_AGENTS` (`compression`,
  `make_compression`).
- `src/hypergraph_agent/techtree/compression.py`: the compressors (`stitch_library` and
  `bpe_library`, which also report their effort, `stitch_fragments`, `bpe_fragments`, the chain
  encoding and parser), description lengths (`description_lengths`, `min_tokens`,
  `parent_residues`), the arms (`PressLog`, `Recorder`, `CompressedPromote`, `Enumerator`,
  `LayerOracle`), the factory `make_compression` (which also builds study L's arms and study U's
  `pooled` unchanged), the agent parameters (`corpus`, `stitch_iterations`, `bpe_merges`),
  `CompressionError` and the row summaries `layer_summary` and `library_summary` (`ROW_SUMMARIES`).
  stitch_core is imported where it is used, so the package imports without it (as study X's learner
  libraries); a missing package stops a Stitch arm with a message naming the version. A panic of
  Stitch's backend becomes a `CompressionError`: the arm counts it in its rows
  (`ca_compress_errors`) and uses an empty library for that episode, and the oracle counts it too and
  reports layer 2 equal to layer 1 for that world. None occurred in development; a nonzero count in
  the measurement is reported with the results and the affected runs are listed, and no test reads it.
- The module is under the source check of the TechTree agent modules (no reflective builtins,
  dunder, frame or traceback attributes, no private attributes of other objects); its import
  allowlist is the agent modules' plus `layered` and `stitch_core`. The addition of `stitch_core` is
  the one extension (approved by the review): the compressor under test is an external library that
  receives only lists of the agent's own presses written as text and returns text; it holds no
  reference to an environment, a task or a world, and the module still imports nothing of the
  evaluator side (`generator`, `study`, `analysis`, `config`). A test also checks that no attribute
  named like a hidden world field is read from anything but `self`.
- Configs (`configs/compression/`, own schema, listed in `tests/test_hygiene.py`): `ca_main.yaml`
  (measurement), `ca_dev.yaml` (first development pass, corpus `firings`, every arm),
  `ca_dev_tests.yaml` (second development pass, corpus `tests`, the three arms whose actions depend on
  the corpus).
- `tests/test_compression.py`: the frozen-behaviour digests below; the latent worlds (fragments hidden
  and shared, compositions, coupling with plain worlds and across depths, the extended flat rule,
  validation, an exact reference against an independent full-state search, a public view that never
  carries the option); the optional config key and the registration; the chain parser, Stitch on a
  planted fragment, byte-pair encoding, the compressors' effort, description lengths, the parent
  residues and Spearman; the factory and parameters; `pooled` and layer 0 identical to `remember`;
  sound evidence of every compression arm; the three corpora; the library, its effort fields and its
  macro candidates; the enumerator's tiers; the oracle's description lengths and its identity with
  study L's oracle; arms without compression ignoring the corpus; the source check and the optional
  import; the configs (one ledger declaration, caps, flags, fresh namespaces, the development passes
  mirroring the measurement); the measurement flag; an end-to-end run with the analysis and its row
  summaries; the row summaries on hand-made rows; and the hygiene of the study's files (line length,
  dashes, local paths). stitch-core is optional for the package, and the tests that need it skip
  without it (19 of 57).

Frozen behaviour (default off). The tests embed digests computed with the committed code before the
option existed (the TechTree code of 7ffa394, unchanged up to ec5b442): every one of the 12 TechTree
configs of studies U, L, U2, P and PG loads to the same dictionary; their worlds, tasks with reference
lengths and plans, and run hashes are generated identically, as are study X's four TechTree test
streams and the four TechTree benchmark families; and every arm of every variant of those 12 configs
(239 agent cases, one world of four episodes each, played as the runner plays them) produces identical
rows. On 2026-10-29 the committed analyses of studies L (phases L1 and L2) and U regenerated byte for
byte with the working tree from their measurement runs (`analysis.json` and both figures each), and
the 25,056 measured rows of L1, L2 and U matched the regenerated tasks (task key, world key, goal level
and reference length).

Development commands (from the repository root; analyses into a scratch folder outside the
repository):

```bash
python -m hypergraph_agent.techtree --config configs/compression/ca_dev.yaml
python -m hypergraph_agent.techtree --config configs/compression/ca_dev_tests.yaml
python -m hypergraph_agent.techtree.analysis --config configs/compression/ca_dev.yaml \
    --runs runs --out ../scratch/ca-dev
```

The calibration of the measurement's design analyses the runs of `ca_dev_tests.yaml` for the three
compression arms together with the runs of `ca_dev.yaml` for every other arm, with the tests of
`ca_dev.yaml` (`techtree.analysis.analyse` on the two sets of collected runs).

### Development record (calibration, not results)

Prototypes (2026-10-29, in memory on the scratch namespace `proto_ca_dev`, not charged; their
interactions were not counted): with intermediate code, the compressors read either each episode's
whole press sequence or one press sequence per chunk of the engine. Whole episodes gave libraries
made mostly of the exploration walk's frequent pairs and triples (three primitives make every short
sequence frequent), and the arms acted like `remember`; chunks never held a composite as a unit (a
composite whose parents were just typed fires at once, so its replay chunk is empty). The definition
was then fixed at the level of tests: what fired, and what the evidence counts as tested without
firing.

First development pass (2026-10-29, 10:35 to 10:38 local time): `ca_dev.yaml`, corpus `firings`,
seeds 0-1, four worlds per seed and setting, every arm: 86 runs, all completed, 147,366 adaptive and
28,755 reporting interactions, 3 min 15 s in all. Stitch's library was almost only the concepts' own
sequences (at the end of a world 8 to 9 fragments, a share of 0.93 to 1.00 concept sequences), so
`promote_stitch` acted like `promote` (late actions at depth 2: 9.7 and 9.7; at latent depth 2: 37.4
against 36.6), while byte-pair encoding kept shared parts (shares 0.09 to 0.23) and gained at latent
depth 2 (24.1).

In-memory checks (2026-10-29, final code, the `ca_dev` worlds with agent seeds other than the
runner's): with the corpus `tests`, Stitch at latent depth 2 and depth 2 took 26.4 and 20.7 late
actions and byte-pair encoding 33.9 and 28.7; with `episodes`, 26.5 and 32.4, and 29.4 and 29.0. A
Stitch criterion requiring reuse across distinct sequences (one task per distinct sequence, so
identical sequences count once; namespace `proto_ca_dev`, 12 worlds) gave 38.4 at latent depth 2
and 28.2 at depth 2.

Second development pass (2026-10-29, 10:54 to 11:04 local time): `ca_dev_tests.yaml`, corpus `tests`,
the three compression arms on the worlds, seeds and agent seeds of the first pass: 30 runs, all
completed, 75,661 adaptive interactions, 9 min 51 s in all.

Late actions per episode of the measurement's design (second-pass compression arms, first-pass
anchors), eight worlds per setting:

| Arm | d0 | d1 | d2 | latent_d1 | latent_d2 |
|---|---|---|---|---|---|
| `remember` | 45.2 | 45.8 | 28.6 | 43.4 | 33.8 |
| `promote` | 38.2 | 43.8 | 9.7 | 44.2 | 36.6 |
| `promote_stitch` | 47.7 | 41.9 | 17.7 | 51.9 | 22.2 |
| `promote_bpe` | 60.1 | 35.3 | 14.7 | 45.1 | 25.0 |
| `sham_promote` | | | | | 35.8 |
| `enum_atoms` | 52.6 | 38.3 | 9.9 | 47.7 | 11.9 |
| `enum_frag` | 51.2 | 39.1 | 10.7 | 48.4 | 27.1 |
| `random` | | | 96.5 | | 92.8 |
| `oracle_library` | 13.4 | 11.5 | 9.1 | 13.7 | 11.9 |
| `reference` | 12.6 | 10.8 | 7.0 | 12.0 | 8.6 |

Tests on these eight worlds per setting (intervals over eight worlds are wide): CA1 -14.4 [-29.7,
1.0], six of seven untied worlds lower (per-world SD 24.0); F1 not inside (90% [-27.6, -1.3] against
plus or minus 3.66); C1 -11.5 [-23.1, -1.5]; H1 +8.0 [2.4, 14.1] against a margin of 0.97 (SD 9.0);
the precedence rule would read "inconclusive" here, CA1 missing narrowly. S1 -11.6 [-23.5, -1.4]; S2
-2.7 [-14.9, 9.4] (neither side holds); S3 +5.0 [0.5, 12.7] against 0.97 (SD 10.6); S4 -13.5 [-28.0,
-0.2]; S5 +9.5 [-0.04, 21.1]; S6 +7.8 [-7.8, 24.0]; S7 -22.4 [-38.1, -8.9]; S8 -9.1 [-22.0, 2.4]; LF1
-18.7 [-29.5, -7.5]; LF2 -21.9 [-33.3, -10.8]; LF3 +15.2 [7.6, 23.6], higher in all eight worlds; LF4
+0.8 [0, 2.4] against 0.99 (SD 2.2); LF5 +7.3 [-6.0, 21.6]; V1, V2 and V3 held (no violation in 480
episodes each).

Libraries at the end of a world, corpus `tests`: Stitch 33 to 36 fragments on average (at most 40),
with shares 0.18 to 0.25 concept sequences, 0.13 to 0.19 shared parts and 0.62 to 0.65 neither;
byte-pair encoding 46 to 50 (50 merges in 319 of 480 episodes), shares 0.12 to 0.18, 0.10 to 0.14 and
0.69 to 0.76; `enum_frag`'s Stitch library 16 to 35. No compression error occurred.

Description lengths of the worlds' level-3 composites (layers 0, 1 and 2, means over eight worlds):
depth 0: 6, 4.71, 3.92; depth 1: 6, 4.71, 4.17; depth 2: 6, 2.00, 2.00; latent depth 1: 6, 4.79,
4.08; latent depth 2: 6, 3.00, 2.38. Spearman correlations over 40 worlds per learner, against actions
to the first success: `enum_atoms` 0.90, `enum_frag` 0.72, the three learners pooled 0.40 (`remember`
has a constant length). The parent residues were added after the development runs (decision 8), so
the development rows do not carry them.

Uncharged interactions, all in memory: 157,170 counted (the in-memory checks of 2026-10-29,
100,912 and 14,136; the replay, timing and memory checks of 2026-11-05, 42,122) and the prototypes,
uncounted; the independent review's replays on the `ca_dev` worlds added 34,739 (8,785 with Stitch,
25,954 with byte-pair encoding).

Decisions that came from development:

1. The corpus is fixed at the level of tests (prototypes). The first pass's corpus `firings` held
   almost no failed press sequences (failed replays are rare) and made Stitch's library the concepts
   themselves; because the question names successful and failed press sequences, the measurement uses
   the corpus `tests`, with every failed test. `firings` and `episodes` stay implemented as agent
   parameters.
2. Stitch keeps its default single-task pruning (identical sequences count separately); the
   criterion of reuse across distinct sequences (identical sequences count once), tried in memory,
   gained nothing at latent depth 2 and cost more at depth 2.
3. The caps of 50 abstractions and 50 merges stay as declared vocabulary sizes. Tuning them on eight
   worlds would fit noise.
4. `latent: 2` (two fragments for the six composites of levels 2 and 3, every fragment shared): with
   one fragment the latent draw often needed a further attempt, which breaks the coupling with the
   plain worlds (508 of 800 draws coupled, against 798 of 800 with two).
5. The second pass ran only the arms whose actions depend on the corpus; the other arms ignore it
   (tested), so the first pass's runs serve as their calibration (a full rerun of every arm would
   exceed the 76,973 `dev` interactions left), and the measurement runs every arm in one config.
6. S2's direction was set after both passes; the review made it two-sided.
7. Made after the development runs, affecting no development row (checked on 2026-11-05, below):
   stitch_core is imported where it is used, a missing package names the version, backend panics
   become `CompressionError` (counted, an empty library), and the oracle counts such errors too.
8. Made after the review, affecting no development row: the rows record each compression's effort
   and cap, the library summary reports the share of episodes at the cap, and the oracle reports the
   parent residues as a fourth description length.

Checks of 2026-11-05 before the review (HEAD ec5b442 with this working tree, Python 3.12): 16 stored
development runs (the three compression arms at depth 2 and latent depth 2 of the second pass, and
every arm at latent depth 2 of the first, seed 0) replayed in memory with the current code gave
identical rows in every stored field except the per-episode timing; a timing and memory check played
two Stitch runs of four worlds at depth 0 (38.5 and 59.8 s, peak working set about 240 MB). Both
development analyses (the first pass's, with its figures, and the calibration of the measurement's
design) regenerated byte for byte. The frozen-behaviour digests were recomputed with the committed
code of ec5b442 (extracted into a scratch folder) and with the working tree, and matched.

Decisions of the independent review (2026-11-05), made before the freeze:

9. Accepted as declared: the corpus `tests` (with the disclosure above), H1 and C1 as conditions, the
   latent design (with CA1's comparator stated), the borrowed anchors of the second pass, `pooled`
   and `enum_prim` not run, the layer-2 description length as a descriptive measure, the change to
   `config.py` and `stitch_core` in the import allowlist (with the version preflight).
10. Both caps bind (Stitch returned 50 abstractions in 40 of 44 compressions per replayed run); they
    are declared vocabulary sizes, and the rows now record each compression's effort.
11. Power notes for F1, H1, S3 and LF4; S2 two-sided; the verdict reported with its failed condition.
12. Run rules: `--resume` in the measurement command, a log file, one process, one code identity
    checked by hand before the analysis, a new run name for a fresh measurement after a code fix,
    and the stitch-core preflight.
13. The parent residues added as a second descriptive description length (decision 8); "best case"
    wording replaced by "Stitch's greedy result on the world's own sequences".
14. Wording: censoring at the world's total, composites including flat concepts, `pooled` equal to
    `remember` in every world, the handling of compression errors, the runtime up to about 75 minutes
    under load; the limits extended as above.

Checks after the review's changes (2026-11-05): `tests/test_compression.py` 57 passed; the dry runs of
the three CA configs gave the figures above, and neither `ca_main.yaml`'s runs, caps and run hashes
nor the ledger declaration changed (its analysis tests split S2 into S2a and S2b, and comments
changed).

## Amendments

None.

## Results

None yet: no measurement run exists.

## Verdict and limits

None yet.
