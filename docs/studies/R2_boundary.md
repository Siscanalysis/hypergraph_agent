# Study R2: where the amortization result stops replicating

Status: protocol frozen on 2026-10-20 by the commit that freezes this file,
before any measurement run; measurement runs execute that commit from a clean
checkout. Sections 1-3 were first drafted before any
R2 run, development or measurement, from generator statistics (the
measurement namespaces and the scratch namespace `r2_design_dev`, no
interaction) and study R's recorded runs. An independent review followed the
development runs of Section 4 (one world per family), so the review changes
listed in Section 4 were made with those runs seen: the family DH and its
prediction, the renaming of X to I and its role as positive control, the
pre-registered readings of D with DH, of F2-A, of the decomposition and of the
slack outcomes, and the stricter bounds check. The expected column of the
budget table also postdates them. No other prediction, reading or criterion
was changed after the development runs. Development used only the `dev`
allocation of `runs/R2-ledger.json` and the namespace `r2_dev`. Sections 5 and
6 will report the measurement runs, which may start only after the protocol is
frozen by a commit and the ledger amendment under Budget is recorded.

## 1. Question and claim

Study R found that sampling among the hypotheses consistent with the evidence
(`focused_sample`) makes episodes 9-16 of a world about 18% cheaper than the
brute-force plan (`maximal`) in the Stage D task family F1, and that the result
does not hold in a deeper family F2: there the sampler never broke even within
16 episodes, its whole late excess came from failed episodes (budget exhausted
after contradicted guesses) and failure grew with goal depth. F2 differs from
F1 in five ways at once: goals at depth 3-5 instead of 1-3, an alternative
recipe for 60% of the items instead of 25%, pools of 8 candidates instead of 6,
a second item input for a quarter of the recipes above level 2, and (relative
to the longer plans) a tight budget of two steps above the public brute-force
bound. Study R2 asks which of these properties breaks the result.

The claims under test are narrow and about this generator only: whether one of
the F2 properties, added alone to F1, removes or reverses the late advantage;
whether removing it alone from F2 restores it; whether a depth effect remains
when the brute-force headroom of F1 is restored; and whether a looser budget
in F2 (and in the primary family) turns the failures into a late advantage.

## 2. Relation to earlier work

Study R (docs/studies/R_replication.md) is the direct predecessor; its
protocol, analysis code and run-selection rules are reused unchanged (Section
4), and its walker code (`agents/walker.py`) is frozen and not modified here.
R's verdict lists as a limit that "the boundary between them (depth,
alternatives, pool size or budget) is not located" and that budgets were fixed
at two steps above the brute-force bound; R2 addresses exactly these two
limits. R2 runs no failure noise (R covered it) and no random omission control
(R showed that using the evidence beats omitting candidates at the same rate).

docs/PRIOR_ART.md, Section 8.1, frames the axes R2 separates: under goal-only
observation one episode is one group test on the union of the omissions along
the chain (N6), so a failed goal carries at most one bit about all the recipes
of the chain jointly and blame ambiguity grows with chain depth; alternative
recipes play the role of the disjunctive model in N7 and N8
(identifiability); pool size sets the hypothesis entropy and, separately, the
brute-force headroom. Section 8.1 also notes that a second family is a
replication only if it varies these axes; R varied them together, R2 varies
them one at a time. Worlds remain the units (N13, R16).

## 3. Protocol (frozen)

### Families

Every family uses the study R conventions: hidden prerequisites
(`unknown_prerequisites`), intermediate items unobserved
(`observe_items: goal_only`), no distractor base facts, failure probability 0,
worlds of 8 levels x 2 items with requirements of 1-3 candidates, budget =
public brute-force bound + slack (slack 2 unless stated). Each family plays 10
worlds x 16 episodes per seed, seeds 0-4, and seed s plays evaluation base
seed 10000 + 1000 s (`eval_world_offset_by_seed`), so a family has 50
independent worlds.

| Family | Generator (difference in bold) | Goal depth | Slack | Namespace | Design role |
|---|---|---|---|---|---|
| D | F1 | **3-5** | 2 | `r2_d` | **primary**; one factor from F1 |
| DH | F1, **pools of 8** | **3-5** | 2 | `r2_dh` | D at F1's brute-force headroom |
| A | F1, **alternative recipe 0.6** | 1-3 | 2 | `r2_a` | one factor from F1 |
| P | F1, **pools of 8** | 1-3 | 2 | `r2_p` | one factor from F1 |
| I | F1, **second item input 0.25** | 1-3 | 2 | `r2_i` | one factor from F1; F1-side positive control |
| F2-D | F2 | **1-3** | 2 | `r2_f2_d` | one factor removed from F2 |
| F2-A | F2, **alternative recipe 0.25** | 3-5 | 2 | `r2_f2_a` | one factor removed from F2; near-matched deep check |
| F2-P | F2, **pools of 6** | 3-5 | 2 | `r2_f2_p` | one factor removed from F2 |
| F2 | F2 | 3-5 | 2 | `r2_f2` | replica of R's F2 on fresh worlds (F2-side control); base of the F2 slack levels |
| F2S6, F2S12 | F2 | 3-5 | **6, 12** | `r2_f2` | budget; same worlds and tasks as F2 |
| DS6, DS12 | F1 | 3-5 | **6, 12** | `r2_d` | budget; same worlds and tasks as D |

F1 is the Stage D generator (alternative recipe with probability 0.25, pools of
6, no second item input); F2 is study R's F2 (alternative recipe 0.6, pools of
8, second item input 0.25). The namespaces have never been used (checked
against every config and recorded run); development used `r2_dev` only.

Design. The initial design was one factor at a time from F1 (D, A, P and the
second-item-input family, then called X) and two looser budgets in F2,
optionally also in D; the D slack levels are included. Changes:

- Leave-one-out families from F2 (F2-D, F2-A, F2-P) were added before any R2
  run. Adding a factor to F1 asks whether it is sufficient to break the
  result; removing it from F2 asks whether it is necessary. One-at-a-time
  additions to a shallow base can miss a factor that acts only together with
  depth (the alternative recipes, for example, add 2.9 hidden recipes per task
  at depth 3-5, F2 against F2-A, but 1.0 at depth 1-3, A against F1). F2-X was
  not added: removing the second item input changes F2 by about 0.2 hidden
  recipes per task and about 1.3 bits of entropy (statistics on the scratch
  namespace `r2_design_dev`, generator only), so F2-X would be another F2
  replica.
- The second-item-input family is kept as specified, but at goal depth 1-3
  only level-3 recipes can take a second item input, so it is structurally
  close to F1 (table below). It tests the second input only where it is rare
  and cannot speak to its role at depth 3-5. At review it was renamed I (the
  letter X names another study) and pre-registered as the F1-side positive
  control at R2's code (Endpoints).
- DH was added at review. The predicted readings of the other families split
  almost exactly by brute-force headroom, and D has about 12% less headroom
  than F1 (pooled 1.61 against 1.83), so a reversal in D could come from depth
  or from the lower headroom deep tasks bring in this generator. DH is D with
  pools of 8, which restores F1's headroom at depth 3-5 (pooled 1.80; 1.81 on
  the reviewer's scratch namespace). It also enlarges the hypothesis class as
  P does at depth 1-3; P shows the effect of that change alone.

The slack levels of a family share its namespace, worlds and tasks; only the
budget differs (by exactly the slack difference, tested). Pairing across slack
levels is by world and task, not by random draws: the walkers start from the
same seed (same arm ids and variant name), but their draws desynchronize at
the first episode that reaches the smaller budget, and from then on the two
walks also differ in their evidence. The slack families run `focused_sample`
only. At failure probability 0 `maximal` and the privileged reference never
reach the budget (W2, with a margin of at least 3 steps) and no decision of
theirs reads it, so their episodes at slack 6 or 12 are identical to those at
slack 2 (tested on development streams). The analysis uses the base family's
bounds for a slack family only after checking, seed by seed, that the recorded
generator settings agree apart from the slack, and, episode by episode against
the borrowed runs themselves, that the task, goal depth and optimal length
agree and that the recorded budgets differ by exactly the slack difference,
with no episode missing. A refused borrowing leaves the slack family
`incomplete`.

### Family statistics (generator, reference solver and brute-force plan only)

Over the 800 measurement tasks of each family; no environment interaction
(`python -m hypergraph_agent.analyze_boundary families`). The F1 and F2 columns
marked (R) are study R's measurement tasks, recomputed with the same code
(they agree with R's table; R printed F1's mean optimal length 6.65 as 6.7).
"Crafts in the optimal plan" is the number of recipes the optimal plan uses;
"headroom" is the brute-force plan (what `maximal` executes at failure
probability 0, checked against all 1,600 recorded `maximal` episodes of R's F1
and F2: identical lengths) over the optimum.

| | F1 (R) | D | DH | A | P | I | F2-D | F2-A | F2-P | F2 | F2 (R) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| optimal length, mean (range) | 6.6 (3-12) | 11.5 (7-18) | 11.6 (7-16) | 6.5 (3-11) | 6.7 (3-13) | 7.0 (3-14) | 6.2 (3-12) | 11.9 (7-23) | 11.0 (6-18) | 10.9 (6-17) | 11.0 (6-17) |
| hidden recipes per task, mean (range) | 3.7 (2-9) | 6.4 (4-14) | 6.4 (4-13) | 4.7 (2-11) | 3.7 (2-10) | 3.8 (2-9) | 5.1 (2-11) | 7.1 (4-14) | 10.0 (4-18) | 10.0 (4-19) | 10.1 (4-18) |
| alternative recipes per task | 0.57 | 1.03 | 0.89 | 1.36 | 0.57 | 0.55 | 1.62 | 1.34 | 3.41 | 3.53 | 3.45 |
| hypothesis entropy per task, bits | 19.7 | 34.5 | 41.7 | 25.4 | 24.4 | 20.2 | 33.1 | 46.4 | 53.4 | 65.1 | 65.6 |
| crafts in the optimal plan | 1.98 | 3.97 | 4.02 | 2.00 | 2.02 | 2.08 | 2.00 | 4.21 | 4.09 | 4.08 | 4.06 |
| distinct recipes per world over 16 episodes | 8.9 | 12.9 | 13.0 | 10.9 | 9.0 | 8.6 | 11.3 | 13.5 | 17.3 | 17.5 | 17.5 |
| recurrence, episodes 2-16 (9-16) | 0.91 (0.98) | 0.93 (0.98) | 0.93 (0.98) | 0.93 (0.98) | 0.92 (0.98) | 0.92 (0.98) | 0.93 (0.98) | 0.94 (0.98) | 0.96 (0.99) | 0.95 (0.99) | 0.95 (0.99) |
| headroom, per-task mean (pooled) | 1.91 (1.83) | 1.64 (1.61) | 1.84 (1.80) | 1.93 (1.85) | 2.31 (2.20) | 1.90 (1.82) | 2.43 (2.30) | 1.83 (1.77) | 1.68 (1.64) | 1.90 (1.86) | 1.88 (1.84) |
| budget, mean (range) | 19.4 (12-27) | 24.6 (19-31) | 25.8 (21-31) | 20.7 (13-27) | 21.6 (15-27) | 19.6 (13-27) | 22.6 (15-27) | 26.3 (22-31) | 26.9 (20-31) | 27.2 (22-31) | 27.4 (22-31) |
| budget / optimum, pooled | 2.92 | 2.13 | 2.23 | 3.20 | 3.24 | 2.81 | 3.62 | 2.21 | 2.44 | 2.50 | 2.50 |

The slack levels have the structure of their base family; only the budget
changes:

| | D | DS6 | DS12 | F2 | F2S6 | F2S12 |
|---|---|---|---|---|---|---|
| budget, mean (range) | 24.6 (19-31) | 28.6 (23-35) | 34.6 (29-41) | 27.2 (22-31) | 31.2 (26-35) | 37.2 (32-41) |
| budget / optimum, pooled | 2.13 | 2.48 | 3.00 | 2.50 | 2.86 | 3.41 |
| smallest budget minus brute-force length | 3 | 7 | 13 | 3 | 7 | 13 |

In every family every reference is exact, the brute-force enumeration is not
truncated, no budget reaches the cap of 256 and the brute-force plan fits
every budget, so `maximal` cannot fail (DECISIONS W2).

What the table shows, before any run: depth is the only single factor that
changes the length of the plans (crafts 3.97 against 1.98; F2: 4.08) and it
nearly doubles the entropy per task; it also lowers the brute-force headroom
(pooled 1.61 against 1.83), so even a perfectly informed sampler has less to
gain in D, and D has the tightest budget relative to the optimum (2.13). Pools
of 8 raise the entropy per recipe and the headroom (P: 2.20); alternative
recipes raise the number of distinct recipes per world; the second item input
changes almost nothing at depth 1-3 (I). DH has D's plan length (crafts 4.02)
at about F1's headroom (pooled 1.80), with a larger hypothesis class (41.7 bits
against D's 34.5). DH and F2-A differ only in the second item input, so F2-A
(pooled headroom 1.77) is a second deep family near F1's headroom, on other
worlds but not an independent test of the generator.

### Arms

| Arm | Kind | Families |
|---|---|---|
| `focused_sample` | adaptive | all; the study R walker unchanged (focused walk to a node consistent with every observation, then 50 Metropolis moves among such nodes) |
| `maximal` | reporting | the nine slack-2 families; the slack levels use their base family's runs |
| `reference` | reporting, privileged | as `maximal` |

### Endpoints

As in study R. For each world and arm, the late-half cost ratio: primitive
steps over optimal steps, summed over episodes 9-16 of the world; the early
half is episodes 1-8. A failed episode counts with every step it used, its
whole budget. The contrast is the paired per-world difference `focused_sample`
minus `maximal`.

**Reading of a family.** The mean paired late difference over its 50 worlds
with a 95% world-cluster bootstrap interval, read three ways: `advantage` if
the upper end is below 0 (study R's pass criterion), `reversed` if the lower
end is above 0 (R's F2 was +0.202 [0.120, 0.280]), `inconclusive` otherwise.
A family with an incomplete run, fewer than 50 paired worlds or no usable
bounds (a refused borrowing) is `incomplete`.

**Primary**: family D, the reading of its late difference. D is the primary
because (i) depth is the only single factor that changes plan length, the
quantity the failure mechanism R describes depends on (wrong guesses on long
chains exhausting the budget); (ii) in R's recorded runs the late advantage
shrinks with goal depth already inside F1 and late failure grows with depth in
F2 (predictions below); (iii) it is a one-factor change from the family where
the result holds, unlike the slack contrasts, whose cost endpoint mixes two
opposite effects (below). Its one-factor reading is about depth together with
the lower headroom that depth brings in this generator; DH and the
decomposition separate the two (below).

**Secondaries**, each with the same reading and reported separately: DH
(depth at F1's headroom), A, P, I (sufficiency of one factor), F2-D, F2-A,
F2-P (necessity of one factor), F2 (a replication of R's F2), F2S6, F2S12,
DS6, DS12 (looser budgets).

**Depth against headroom (pre-registered).** D and DH are read together:

| D | DH | Reading |
|---|---|---|
| `reversed` | `reversed` | depth matters beyond headroom |
| `reversed` | `advantage` | headroom, not depth: D's reversal is explained by the lower headroom deep tasks bring in this generator |
| `reversed` | `inconclusive` | headroom accounts for at least part of D's reversal; that depth matters beyond headroom is not shown |
| not `reversed` | any | no reversal to attribute; DH is reported as a secondary |

F2-A is the near-matched deep check: it is read the same way as DH. If DH and
F2-A agree, the reading rests on two deep families near F1's headroom (on
different worlds; they differ only in the second item input). If they
disagree, the depth-against-headroom reading is reported as unresolved.

**Attribution of a reversal (pre-registered).** The pooled late excess of
`focused_sample` over `maximal` is split into the episodes the sampler failed
and those it solved (the parts add up to the excess). A positive excess is
attributed to the failed episodes when they carry at least half of it
(failed part at least the solved part), else to the solved episodes. For D:
attributed to failed episodes, the reversal follows the failure mechanism
(wrong guesses on long chains exhausting the budget); attributed to solved
episodes, the sampler is not cheaper than the brute-force plan even when it
succeeds, which is what lower headroom alone predicts. The same attribution is
reported for every reversed family.

**Controls (pre-registered).** I is the F1-side positive control at R2's code:
it is structurally near F1, where study R found an advantage, so it is
expected to read `advantage`; if it does not, D's reading and every other
one-factor-from-F1 reading carry that caveat. F2 is the F2-side control: if
it is not `reversed`, every comparison with F2 carries that caveat.

**Slack contrasts (secondary, pre-registered).** For F2S6 and F2S12 against
F2, and DS6 and DS12 against D: per world, the late difference in the looser
family minus that in its base (the same worlds and tasks), with the same
bootstrap; read as `lower` (interval below 0: the looser budget lowers the
sampler's late excess), `higher` or `inconclusive`. Reported with each: the
change of the early difference and the change of the sampler's late success,
each with its interval; "late success higher" means that the interval of the
change lies above 0. The cost endpoint mixes two effects of a looser budget:
an episode with the same history can only get longer (the walker never reads
the budget, so up to the old deadline it does exactly the same thing, and a
failure now counts the larger budget), while a longer episode leaves more
evidence for later ones.

**Reported with every family (pre-registered, no verdict)**: the
difference-in-differences DiD = (late difference) - (early difference) with
its interval, as R's amortization check; the paired late success difference
with its interval; the decomposition of the pooled late excess into failed
and solved episodes of the sampler, its attribution and both arms' ratios on
the solved ones; late failure by goal depth; the number of failed late
episodes that ended at the deadline after at least one contradicted guess.

**Descriptive**: break-even episode (as in R: the first episode from which the
pooled per-episode ratio of `focused_sample` stays at or below that of
`maximal`; "never" if none); early, late and overall pooled ratios; success;
per-seed late ratios; cost ratio by episode index; omitted fraction of
candidates; exact sign tests on the per-world late differences.

### Predictions

Written before any R2 run, except the DH row, which was written after the
development runs of the other families and before DH's own (Section 4).
Sources: study R's results and, where marked (*), descriptive numbers computed
for this protocol from R's recorded runs at failure probability 0 before any
R2 run and not reported in R. In R's F1 the pooled late ratios of
`focused_sample` against `maximal` were 1.39 against 2.01 at goal depth 1,
1.45 against 1.86 at depth 2 and 1.65 against 1.74 at depth 3 (*), so the
advantage shrinks with depth both because the sampler gets costlier and
because the brute-force headroom shrinks. Early failure of the sampler was 0,
0.08 and 0.38 at depths 1, 2 and 3 (*): in F1 shallow tasks are nearly
single-recipe group tests. In R's F2, which has no shallow tasks, late failure
was 0.18, 0.42 and 0.59 at depths 3, 4 and 5 (*; computed as 27 of 146, 52 of
125 and 76 of 129 late episodes; R's text gives 0.19 for depth 3), against
0.02 at depth 3 in F1 (*). In the same F2 runs the sampler's pooled late ratio
was about flat across depths (2.04, 2.08, 2.04) while `maximal`'s fell (2.01,
1.83, 1.70) (*): within F2 the reversal grows with depth mainly through the
headroom. F2 also has larger pools and more alternatives, so these
comparisons alone do not separate depth from the rest, which is what D, DH and
F2-D do.

| Family | Predicted reading | Further predictions |
|---|---|---|
| D (primary) | `reversed`, through depth together with the lower headroom it brings here | late difference between +0.05 and +0.25; break-even never; the excess attributed to failed episodes; late failure growing with depth |
| DH | `inconclusive` | `maximal` about 0.2 dearer than in D (headroom 1.80 against 1.61) while the sampler pays for 7 more bits per task, so D's predicted +0.05 to +0.25 shifts to about -0.15 to +0.05; written after the development runs (see the note below) |
| A | `advantage` | point estimate above R's F1 estimate (-0.357): a descriptive comparison of point estimates, no test |
| P | `advantage` | the larger headroom (2.20 pooled) keeps the advantage despite 1.2 more bits per recipe |
| I | `advantage` | close to R's F1 (structurally near F1); the positive control |
| F2-D | `advantage` | shallow tasks restore near-single-recipe tests even with alternatives, pools of 8 and second inputs |
| F2-A | `reversed` | depth 3-5 is kept (headroom 1.77) |
| F2-P | `reversed` | depth 3-5 is kept (headroom 1.64) |
| F2 | `reversed` | replicates R's F2 (+0.202 [0.120, 0.280]) |
| F2S6, F2S12 | not `advantage` | late success of the sampler higher than in F2 |
| DS6, DS12 | not `advantage` | late success of the sampler higher than in D |
| slack contrasts | not `lower` | early difference higher (longer early episodes); late success higher |

Note on DH. Its prediction differs from F2-A's although the two families
differ only in the second item input. F2-A's was written before any run; DH's
after the development runs, in which F2-A's single world favoured the sampler
(late 1.49 against 1.93) and D's did not (1.50 against 1.38). The shift
computed in the table uses only numbers available before those runs; both
predictions are kept as written.

Taken together the predictions say: among the four structural factors, goal
depth, with the lower or higher headroom it brings in this generator, is
sufficient and necessary for the loss of the advantage; at F1's headroom (DH)
the loss is predicted to shrink to an inconclusive reading, so these
predictions do not claim that depth matters beyond headroom. The tight budget
is predicted not to be what breaks it: extra budget should raise late success
but lengthen failing and recovering episodes about as much as the extra
evidence shortens later ones. The predictions are not criteria; every family
is read and reported whatever the predictions say.

### What each outcome would mean

| Outcome | Interpretation |
|---|---|
| D `reversed` | depth, with the lower headroom it brings in this generator, is sufficient to reverse the result; DH and the attribution say whether depth matters beyond headroom |
| D `inconclusive` | depth (with its lower headroom) removes a demonstrable advantage; a reversal is not shown |
| D `advantage` | depth alone is not sufficient; F2-A and F2-P show whether depth with larger pools (F2-A) or with more alternatives (F2-P) is |
| D and DH | the depth-against-headroom table above |
| I not `advantage` | the positive control fails at R2's code; D's reading and every one-factor-from-F1 reading carry that caveat |
| F2-D `advantage` | depth is necessary in F2: F2 with shallow goals keeps the advantage |
| F2-D not `advantage` | the other F2 properties together remove a demonstrable advantage even with shallow goals (reverse it if `reversed`) |
| A, P or I not `advantage` | that factor alone removes a demonstrable advantage at depth 1-3 (reverses it if `reversed`) |
| F2-A or F2-P `advantage` | removing alternatives (F2-A) or the larger pools (F2-P) restores the advantage: that factor is necessary in F2 |
| F2S12 or F2S6 `advantage` | R's F2 failure arises from an interaction of the family's structure with the tight budget rule: with more room the advantage returns |
| a slack contrast `lower` without `advantage` | the budget contributes but is not sufficient |
| a slack contrast not `lower`, late success higher | the budget causes the failures, but the extra steps cost about what the extra evidence saves; the reversal is not an artifact of the budget rule at these slack levels |
| a slack contrast not `lower`, late success not higher | the failures are not a consequence of the budget at these slack levels |
| F2 not `reversed` | R's F2 result does not replicate on fresh worlds; every F2-based comparison is then read with that caveat |

### Statistics

The unit is a world; seeds change only the walker's random stream and which
worlds it plays. Intervals are percentile bootstraps over worlds (10,000
resamples with replacement, seed 20261008, the same resamples for every
statistic of a contrast), each world's value held fixed (study R's
`cluster_bootstrap_mean`). There is one primary; the twelve secondary families
and four slack contrasts carry their own readings with no multiplicity
correction: each reading has two 2.5% tails, so about 0.8 of the sixteen
secondary readings would be `advantage`, `reversed`, `lower` or `higher` by
chance if no effect existed. A family or contrast needs 50 paired worlds and
complete runs. Analysis:
`python -m hypergraph_agent.analyze_boundary endpoints --runs runs --prefix r2-
--out artifacts/boundary`.

### Budget

Ledger `runs/R2-ledger.json`, declared identically in every config of
`configs/boundary/`: allocations `dev` 6,000 and `r2` 278,000 (adaptive cap
284,000), reporting cap 352,000. The ledger file was created by the first
development run with the earlier declaration (`dev` 6,000, `r2` 257,000,
adaptive cap 263,000, reporting cap 310,000), before DH was added; the new
declaration is entered in the ledger file by a logged amendment before any
measurement run (as for study M), with development usage unchanged.

A run's cap is the largest sum of task budgets among the five seed streams of
its family, so no cap can cut a run short (no episode exceeds its budget). The
`r2` allocation covers the sum of the stream budgets of all adaptive runs and
the reporting cap that of all reporting runs plus the development reporting
runs, so the ledger cannot cut a run short either. Worst case = sum of stream
budgets (dry runs, generator only). The expected adaptive use scales the
interactions per task of the development run of the same family (one world,
so a rough figure) to 800 tasks; the reporting use is exact, because at
failure probability 0 `maximal` executes the brute-force plan and the
reference the optimal one in every episode, so it is the sum of those lengths
over the measurement tasks (generator only). The expected column was filled
in after the development runs. A rerun (Stopping rules) is charged again; the
gap between expected and worst-case use, about 60,000 adaptive
and 155,900 reporting interactions, covers reruns.

| Item | Runs | Cap per run | Worst case | Expected |
|---|---|---|---|---|
| D `focused_sample` | 5 | 3,988 | 19,676 | 17,450 |
| DH `focused_sample` | 5 | 4,172 | 20,652 | 16,850 |
| DS6 `focused_sample` | 5 | 4,628 | 22,876 | 20,100 |
| DS12 `focused_sample` | 5 | 5,588 | 27,676 | 20,900 |
| A `focused_sample` | 5 | 3,400 | 16,576 | 11,850 |
| P `focused_sample` | 5 | 3,495 | 17,299 | 9,900 |
| I `focused_sample` | 5 | 3,288 | 15,694 | 9,400 |
| F2-D `focused_sample` | 5 | 3,717 | 18,073 | 12,600 |
| F2-A `focused_sample` | 5 | 4,256 | 21,037 | 16,850 |
| F2-P `focused_sample` | 5 | 4,356 | 21,509 | 16,700 |
| F2 `focused_sample` | 5 | 4,448 | 21,792 | 21,000 (study R's F2 used 19,240) |
| F2S6 `focused_sample` | 5 | 5,088 | 24,992 | 20,300 |
| F2S12 `focused_sample` | 5 | 6,048 | 29,792 | 23,750 |
| **Adaptive, allocation `r2`** | 65 | | 277,644 (allocated 278,000) | 217,650 |
| `maximal` and `reference`: D, DH, A, P, I | 50 | as the family | 179,794 | 24,089 + 25,923 + 14,771 + 17,085 + 15,726 |
| `maximal` and `reference`: F2, F2-D, F2-A, F2-P | 40 | as the family | 164,822 | 24,969 + 16,469 + 26,333 + 23,307 |
| **Reporting** | 90 | | 344,616 (348,044 of the 352,000 cap left after development) | 188,672 |
| Development, allocation `dev` | 31 | | 5,520 adaptive, 6,920 reporting | 4,353 adaptive and 3,956 reporting used |

### Stopping rules

As in study R. Every run plays its whole stream; the caps cannot bind. Exactly
one kind of rerun is allowed: a run that did not complete (an error or an
interruption) may be rerun with the same commit, config hash, seed, arm and
variant; runs are deterministic, so the rerun replays the same stream; it is
used, both runs are listed and both are charged to the ledger. A completed run
is never replaced, and any other repeat is ignored and listed. A family or
contrast that still needs an incomplete run, or has fewer than 50 paired
worlds, is reported as incomplete. Results are analyzed only after every run
of a family (and of its base, for a slack family) has finished; no arm,
parameter, family, reading or prediction changes after the freeze.
Measurement runs execute the freeze commit from a clean checkout (DECISIONS
S1).

### Whatever the primary's reading

The estimate, its interval, the per-world differences, the decomposition with
its attribution and the break-even episode of D are reported first, then DH
and the depth-against-headroom reading, then every other secondary. If D is
not `reversed` the conclusion says what was shown (an `inconclusive` D does
not show that depth alone is harmless, an `advantage` shows that it is not
sufficient) and points to the families that locate the boundary in that case
(F2-A, F2-P). If I is not `advantage`, D's reading is reported with that
caveat. Study R's statements are qualified only by what R2 shows.

### Not included

Failure noise and the random omission control (both covered by study R);
exact posterior sampling and an adaptive group-testing control; F2-X (see
Families); a full factorial of the four structural factors (with R's F1, the
one-at-a-time, leave-one-out and DH families and F2 cover 10 of its 16
cells); a looser budget only in the early episodes, which would isolate the
evidence effect of a longer episode from its direct cost (the generator
applies one slack to a whole stream); a family that restores headroom without
enlarging the hypothesis class (the generator ties headroom to pool size);
shared candidate pools; other priors over the hypothesis class.

## 4. Implementation

- `configs/boundary/`: one measurement config per family (`d.yaml`,
  `dh.yaml`, `d_s6.yaml`, `d_s12.yaml`, `a.yaml`, `p.yaml`, `i.yaml`,
  `f2.yaml`, `f2_s6.yaml`, `f2_s12.yaml`, `f2_minus_d.yaml`,
  `f2_minus_a.yaml`, `f2_minus_p.yaml`; run names `r2-<family>`) and a
  development copy of each (`dev_*.yaml`: run names `r2dev-<family>`,
  namespace `r2_dev`, allocation `dev`, seed 0, one world x 16 episodes).
  Runs: `python -m hypergraph_agent.walk --config configs/boundary/d.yaml` (one
  process per seed with `--seeds`).
- `evaluation/boundary.py`: the family statistics and the W2 check (generator,
  reference solver and the brute-force plan, with a check that its derivation
  enumeration is not truncated), and the endpoints. Run selection, unit
  tables, the paired contrast, the difference-in-differences, the bootstrap
  and the break-even episode are imported from `evaluation/replication.py`
  unchanged; added are the three-way reading, the failed and solved
  decomposition with its attribution, failure by depth, the
  depth-against-headroom reading, the control caveats, the use of a base
  family's bounds by its slack levels (`borrow_check`: recorded generator
  settings per seed; task, goal depth, optimal length and budget difference
  per episode, against the borrowed runs themselves) and the slack contrasts
  (the same check between the two samplers). Two configs of one family are
  refused. On study R's recorded runs it reproduces R's F1 and F2 late
  differences, intervals, worlds lower, break-even episodes and F2's
  decomposition (+0.215 = +0.238 from 155 failed of 400 episodes - 0.023 from
  solved ones).
- `analyze_boundary.py`: `families` (the tables above, no interaction) and
  `endpoints` (`boundary_summary.json`, `.md` and a cost-by-episode figure).
- `tests/test_boundary.py`: configs (one ledger declaration, fresh namespaces,
  caps equal to the largest stream budget sum, one changed factor per family,
  slack levels pairing with their base by world and task), the brute-force
  length against `maximal`'s executed episodes, the slack invariance of the
  bounds, the statistics against study R's F1 and F2 columns, the truncation
  check, and the analysis on invented run records (readings, attribution,
  controls, depth against headroom, refused borrowings).

Changes from the independent review (made after the development runs below):
the family DH with its configs, prediction and pre-registered reading with D;
F2-A as near-matched deep check; the attribution rule for a reversal; the
renaming of X to I (configs `i.yaml`, `dev_i.yaml`, namespace `r2_i`) and its
role as positive control; the slack outcome rows; pairing across slack levels
stated as by world and task; the stricter bounds check and the `incomplete`
reading of a refused borrowing; the truncation check; refusal of duplicate
family labels; per-seed late ratios and omitted fraction in the report; the
new ledger declaration.

Equivalence and development checks:

- The brute-force length computed from the generator equals the executed
  length of every one of the 1,600 `maximal` episodes recorded in study R's F1
  and F2 at failure probability 0; the family statistics reproduce R's family
  table; the endpoint code reproduces R's F1 and F2 cells (above).
- Slack invariance of the bounds (tests, in memory, on development streams of
  D and F2): `maximal` and the reference play identical episodes at slack 2 and
  12, apart from the recorded budget.
- Development runs (`configs/boundary/dev_*.yaml`; allocation `dev`, namespace
  `r2_dev`, seed 0, one world x 16 episodes per family): every run completed
  its 16 tasks without reaching its cap. The I runs were made before the
  renaming and are recorded as `r2dev-x-*`; I's development config differs
  from them only in its run name, so they were not repeated. The 31
  development runs used 4,353 adaptive interactions (of the 6,000 `dev`
  allocation) and 3,956 reporting interactions, 8,309 in all; 28 of them
  (4,016 and 3,446) ran before the review, the three DH runs after it. The
  analysis of these runs paired every slack family with its base family's
  bounds on the same 16 tasks (no mismatch). DH's development world coincides
  with F2-A's on every recipe its tasks use (no second item input was drawn at
  levels 1-5; the worlds differ only at levels 6-8), so the two development
  runs are identical episode by episode and DH's adds no observation of its
  own.
- Runtime per task of `focused_sample` (development, one core, failure
  probability 0): D 0.19 s, DH 0.32, DS6 0.33, DS12 0.40, A 0.30, P 0.09, I
  0.09, F2-D 0.08, F2-A 0.19, F2-P 0.14, F2 0.46 (study R's F2: 0.44), F2S6
  0.37, F2S12 0.57; `maximal` and the reference 0.012 s or less. The whole
  measurement is therefore about 50 minutes on one core (800 tasks per
  family), or about 10-12 minutes with one process per seed (`--seeds`).
- Development observations on one world per family, not evidence (pooled late
  ratios of `focused_sample` and `maximal`; success of `focused_sample` over
  all and late episodes): D 1.50 and 1.38 (0.50, 0.88); DH 1.49 and 1.93
  (0.62, 0.88; identical to F2-A, above); DS6 1.73 (0.56, 0.75); DS12 1.79
  (0.94, 0.88); A 1.59 and 1.81 (0.94, 1.00); P 1.56 and 2.58 (0.94, 1.00); I
  1.34 and 1.79 (0.94, 1.00); F2-D 1.72 and 2.15 (0.75, 0.88); F2-A 1.49 and
  1.93 (0.62, 0.88); F2-P 1.65 and 1.64 (0.81, 0.88); F2 2.32 and 1.84 (0.19,
  0.38); F2S6 1.99 (0.81, 0.88); F2S12 2.31 (0.69, 0.88). The slack levels use
  the base family's `maximal`. The predictions written before these runs were
  not changed after them.

## 5. Results

All 155 measurement runs (65 `focused_sample`, 90 `maximal` and `reference`)
executed the freeze commit 1816ad3 from a clean checkout and completed; none
was rerun. Because family DH was added, every cap was raised by a logged
amendment before the first run (r2 278,000; adaptive 284,000; reporting
352,000). The runs used 205,106 adaptive and 188,672 reporting interactions. The
frozen analysis wrote `artifacts/boundary/` (summary table, JSON, figure); the
borrowed bounds of the four slack families matched their base runs in all 800
shared episodes each.

Late half (episodes 9-16), `focused_sample` minus `maximal`, paired per world,
50 worlds per family:

| Family | Changed from | Late ratios | Difference [95% CI] | Reading |
|---|---|---|---|---|
| D (primary) | F1 + depth 3-5 | 1.65 / 1.62 | +0.029 [-0.044, 0.098] | inconclusive |
| DH | D + pools of 8 | 1.81 / 1.82 | -0.005 [-0.100, 0.085] | inconclusive |
| A | F1 + alternatives 0.6 | 1.62 / 1.89 | -0.271 [-0.379, -0.162] | advantage |
| P | F1 + pools of 8 | 1.60 / 2.24 | -0.639 [-0.746, -0.535] | advantage |
| I (positive control) | F1 + second item input | 1.42 / 1.84 | -0.417 [-0.497, -0.341] | advantage |
| F2-D | F2 without depth | 1.84 / 2.34 | -0.496 [-0.611, -0.377] | advantage |
| F2-A | F2 without alternatives | 1.88 / 1.80 | +0.074 [-0.019, 0.163] | inconclusive |
| F2-P | F2 without pools of 8 | 1.92 / 1.65 | +0.263 [0.175, 0.354] | reversed |
| F2 (replica) | slack 2 | 2.09 / 1.87 | +0.215 [0.109, 0.319] | reversed |
| F2S6, F2S12 | F2, slack 6 and 12 | 2.10, 2.09 / 1.87 | +0.226, +0.216 | reversed |
| DS6, DS12 | D, slack 6 and 12 | 1.61, 1.58 / 1.62 | -0.008, -0.033 | inconclusive |

- Primary: D is inconclusive: depth (with its lower headroom) removes a
  demonstrable advantage (study R's F1: -0.357); a reversal is not shown, and
  under study R's criterion D does not pass. Break-even episode 15; late excess
  0.038 (failed episodes 0.080, solved -0.042). The depth-against-headroom
  reading is "D not reversed: no reversal to attribute"; DH (also
  inconclusive) is reported as a secondary.
- The positive control I keeps the advantage, so F1's result holds at R2's
  code. Alternatives alone (A) and pools of 8 alone (P) keep it too.
- Necessity: removing depth from F2 (F2-D) restores a clear advantage;
  removing the alternatives (F2-A) removes the demonstrable reversal without
  restoring the advantage (inconclusive); removing the pools of 8 (F2-P) leaves
  it reversed. Depth is necessary for the reversal; necessity of the
  alternatives is not shown.
- Budget: no change of the late difference is shown with looser budgets (F2S12 vs F2
  +0.001 [-0.095, 0.099]; DS12 vs D -0.062 [-0.133, 0.009]) although they raise
  late success (F2: +0.345 [0.278, 0.417] at slack 12). At slack 12 F2's late
  excess comes from solved episodes (2.01 against 1.88): the budget causes
  the failures, but the extra steps cost about what the evidence saves, so
  the reversal is not an artifact of the budget rule.
- In every reversed family except F2S12 (where it comes from solved episodes),
  the late excess comes from failed episodes, every one of them ending at the
  deadline after a contradicted guess.

## 6. Verdict and limits

The amortization advantage of sampling among consistent hypotheses is lost
with depth: goals three to five levels deep remove it (inconclusive rather
than reversed), also with F1's headroom restored (DH), and depth combined with
the alternatives and second item inputs of F2 reverses it. Alternatives, larger pools and a second item input do not
remove it on their own, and a looser budget does not restore it, since it
turns failed episodes into long successful ones at about the same cost.


Limits known in advance: one generator; five structural factors but
not all their combinations; I tests the second item input only where it is
rare; DH restores the brute-force headroom only together with a larger
hypothesis class, and DH and F2-A differ only in a factor that is
structurally minor, so they are not independent tests of the generator; slack
levels change both the length of failing episodes and the evidence they
leave; failure probability 0 only; worlds are simulated and the prior is
uniform over the hypothesis class.
