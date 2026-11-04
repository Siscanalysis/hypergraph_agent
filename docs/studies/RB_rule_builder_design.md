# Rule builder: design of a research path (draft, not a protocol)

Status: design, 2026-10-29; RB-A, RB-B and RB-C (both stages) chosen and in preparation as studies RL,
CA, HP1 and HP2. Nothing here is frozen and nothing has been run. Each
experiment below becomes its own study (protocol, ledger, independent review, freeze, clean-checkout
measurement, independent results check) only once it is chosen. Tools were installed and timed on
this machine (Windows, Python 3.12) on small synthetic tasks shaped like these worlds; the timings
are indications, not a benchmark.

## 1. The ideas, separated

The proposal combines six ideas. Each is stated as a claim an experiment can falsify, next to what
the repository already covers, so that no experiment repeats a finished study.

| Id | Idea | Testable claim | Already covered here |
|---|---|---|---|
| I1 | Rules as formulas or code-like programs over actions, including concatenations of actions | A rule language that matches a world's regularities finds them with fewer interactions than agents without it | TechTree composites are contiguous concatenations found by pooled elimination (study U); study L's `promote` searches concatenations of discovered concepts |
| I2 | Learning rules by symbolic regression and a classifier, or by a direct classical method | At these rule sizes, exact search for the smallest consistent rule recovers the rules, also under noise, faster and more reliably than stochastic search | Study X2 (results pending): the posterior against MSGI's CART, macq's Observer and ARMS (MAX-SAT) and SAM |
| I3 | The alphabet depends on the layer of actions | A rule learner's cost follows the rule's description length at its layer; atoms found by compressing the agent's own traces can replace exact discovery events | Study L: exact promotion beats remembering at reuse depth 2 (-21.0 [-27.5, -14.7] actions per episode in phase L1) |
| I4 | Hybrid: the graphs supply the objects, rules are learned over them and fed back as priors | A prior learned over graph objects (requirement sizes, item classes, recipe families) speeds up the exact posterior where the worlds share that structure and costs little where they do not | Nothing: the posterior uses a uniform prior over its class |
| I5 | A separate rule module in parallel with the graph module | Arbitration between modules pays only when the rule module's language covers worlds the posterior's class misses | Nothing |
| I6 | Rules passed between agents | Rules carry most of a teacher's value at a small fraction of the message size | Nothing |

Two measured facts shape the design:

- Symbolic regression is the wrong tool here. The rules are Boolean (which items a recipe requires)
  or sequential (which presses fire a composite). In the timing runs, Boolean symbolic regression
  (gplearn) took 13-20 s per fit and missed one disjunct in each of three disjunctive tasks, and
  PySR needs a Julia runtime. Exact search for the smallest disjunctive rule consistent with
  successes and failures (answer-set programming with clingo, or MaxSAT with python-sat's RC2;
  both are already in this repository's environment) recovered the true rule in all four tasks in
  0.01-1 s, also with 10% of successes recorded as failures, using soft constraints. Symbolic
  regression would fit only numeric laws (counts, costs), which these worlds do not have.
- For sequences, the typed hypothesis class does the work. Enumerating the 1,092 contiguous words
  of length at most 6 found a planted 4-press composite uniquely in 2 ms, while a passive automaton
  learner (AALpy's RPNI) reached 0.77 accuracy with 29 states for a 5-state target and a
  temporal-logic learner (Scarlet) found nothing in 120 s. Compression found planted fragments
  exactly: Stitch in 0.14-0.18 s, byte-pair encoding in 0.03 s (with spurious extra tokens).

A further fact for I4: the generator draws requirement sizes 1, 2 and 3 with weights 0.2, 0.5 and
0.3, while the posterior's uniform prior over its 41 hypotheses implies 0.15, 0.37 and 0.49. A
prior learned from the agent's own solved recipes is the simplest rule over graph objects.

## 2. Experiments, in the recommended order

### RB-A. Smallest consistent rules against the posterior (I2; X2 follow-up; offline)

- Question: does exact search for the smallest consistent rule match the exact posterior, and does
  it win where the posterior's class is misspecified or the evidence noisy?
- Logs: X2's measurement logs (read only), plus new logs, collected once from dev-style
  namespaces, from three variants: action failures with probability 0.1 (slips); requirements of 4
  or 5 items (outside the posterior's class of at most 3); alternative recipes whose attempts are
  logged without saying which recipe applied (only if the environment can log that).
- Arms: the posterior (reference); smallest DNF consistent with all evidence (clingo; RC2 as a
  cross-check); the same with soft constraints for slips; RIPPER (wittgenstein); CART; FIGS
  (imodels). Boolean symbolic regression (gplearn) is an exploratory arm only.
- Metrics: as X2 (AUROC of requirement ranking where the learner gives scores, F1 of the inferred
  set, open-loop acting cost), plus runtime.
- Expected: equal on X2's in-class logs (both search the same class exactly); the DNF learner ahead
  out of class; the posterior ahead under slips unless the DNF learner has soft constraints.
- Cost: hours of daytime-light offline compute; new log collection of minutes.

### RB-B. Atoms from compression against exact promotion (I3; L follow-up; TechTree)

- Question: can atoms proposed by compressing the agent's own traces replace the exact discovery
  events that study L's `promote` uses, and does a rule learner's cost follow the rule's
  description length at its layer?
- Arms: `promote` (exact discoveries as atoms), `promote_stitch` and `promote_bpe` (atoms are
  fragments that Stitch or byte-pair encoding finds in the agent's successful and failed press
  sequences), `sham` (study L's control: atoms that are not concepts), `remember`, `pooled`,
  `random`, `reference`.
- Settings: study L's layers at reuse depths 0, 1 and 2, and a new opt-in generator option
  `latent` (higher concepts share sub-sequences that are never concepts themselves, so no discovery
  event reveals them; default off, so frozen studies stay byte-identical).
- Primary: `promote_stitch` - `promote`, late actions per episode, `latent` worlds at reuse depth 2.
  Harm check: the same contrast on study L's worlds (compression must not be worse than exact
  discovery where discoveries exist).
- Layer factor: the same enumerating rule learner over primitives, over primitives plus discovered
  atoms, and over atoms plus compressed fragments; report the description length of each world's
  composites at each layer next to the learner's actions to success.
- Cost: TechTree runs take seconds to minutes; daytime-light.

### RB-C. Learned priors over graph objects (I4; the hybrid; RecipeQuest)

- Stage 1, offline and without any environment change: the exact posterior with a requirement-size
  prior learned by empirical Bayes from the recipes the agent has already pinned down (within a
  world, then across the worlds of a stream), against the uniform prior and against the generator's
  true weights (the bound). On M1 and X2 logs: AUROC and F1 at each episode count.
- Stage 2, online, with a new opt-in generator option `schema`: base facts belong to K classes and
  recipes to F families; a hidden schema gives each family the classes its requirement draws from.
  Version 1 has public class and family labels; version 2 hides the classes, which the agent infers
  from the facts that co-occur in its consistent requirement sets (objects inferred by the graph).
  Lifted rules ("a family-t recipe requires a fact of class c") are learned by exact enumeration of
  schemas (a small space; clingo for rules with class variables) and enter the sampler as a prior,
  averaged against a no-schema model so a wrong schema loses weight.
- Arms: `posterior` (today's `focused_sample`), `hybrid` (learned prior), `rules_only` (acts on the
  rules' most probable requirement without the posterior's consistency check), `oracle_schema`
  (true schema as prior; the bound), `maximal`, `reference`.
- Primary: `hybrid` - `posterior`, late cost ratio, schema worlds. Falsification: equivalence within
  10%. Harm check: the same contrast on no-schema worlds within a margin.
- Cost: stage 1 offline, minutes; stage 2 runs of the size of studies R and M (about an hour, at
  night).

### RB-D. Rules between agents (I6; after RB-C stage 2)

- A teacher plays N worlds that share a schema; a student plays fresh worlds and receives, at equal
  message size where possible: nothing; the teacher's rules; its raw logs; its learned prior
  parameters; the true schema. A shift condition gives the student worlds where the rules are wrong.
- Metric: the student's late cost against message size in bits. Claim: rules recover most of the
  value of the logs at a small fraction of the size and, checked by the posterior, do no harm under
  a shift.

### RB-E. Parallel rule module with arbitration (I5; only as a misspecification study)

- A rule module built from the same evidence and the same hypothesis class as the posterior is
  dominated by it, so this tests arbitration only where the rule module's language is wider:
  streams mixing in-class worlds with worlds whose requirements exceed the posterior's class (from
  RB-A).
- Architectures: propose-and-verify (the rule module proposes, the posterior checks consistency),
  a bandit per world over the two modules, and arbitration by relative uncertainty.
- Primary: regret against the better module per world type.

### Later

- I1 in full needs conditional composites (for example, a sequence that fires only while a unit is
  held), a new TechTree family; enumeration of typed conditional words against a temporal-logic
  learner (scarlet-ltl) and an automaton learner (AALpy). For contiguous composites, plain
  enumeration already wins.
- Code-like exploration policies (program search over a small language of exploration programs)
  and numeric rules for symbolic regression stay ideas: the generalized-planning tools found run on
  Linux only, and language-model program search needs an external API.

## 3. Tools (installed and timed on this machine)

| Tool | Licence | Status on Windows, Python 3.12 | Use here |
|---|---|---|---|
| clingo 5.8.2 | MIT | wheel, already in the repository's environment | smallest consistent DNF; lifted rules with class variables (RB-A, RB-C) |
| python-sat 1.9.dev15 (RC2) | MIT | wheel, already in the environment | MaxSAT cross-check of the DNF learner (RB-A) |
| scikit-learn 1.9.1 (CART) | BSD-3 | wheel | classifier baseline (RB-A) |
| wittgenstein 0.3.5 (RIPPER) | MIT | pure Python | rule-list baseline (RB-A) |
| imodels 3.0.4 (FIGS) | MIT | pure wheel; no CORELS class in this version | rule-based baseline (RB-A) |
| stitch-core 0.1.29 | MIT | wheel | compression-based atoms (RB-B) |
| byte-pair encoding | none | 20 lines in the repository | compression-based atoms (RB-B) |
| AALpy 2.0.0 | MIT | pure wheel | automaton baseline for sequence rules (later) |
| scarlet-ltl 0.0.4 | MIT (ltlf2dfa: LGPL-3.0) | pure wheel | temporal-logic baseline (later) |
| gplearn 0.4.3 | BSD-3 | pure wheel | exploratory symbolic regression arm (RB-A) |
| PySR 2.7.2 | Apache-2.0 | needs Julia (downloaded at first import) | not planned |

Not usable here without Linux or WSL: Popper (needs SWI-Prolog), DreamCoder, babble, D2L and
dlplan, flexfringe (MSYS2 build), the differentiable ILP systems (Python 2.7 era), pyoperon (no
Windows build). LILO and WorldCoder need a language-model API.

## 4. Closest prior work (records verified; see docs/PRIOR_ART.md once a study is frozen)

- I2: Yu, Ignatiev, Stuckey and Le Bodic, optimal decision sets and lists with SAT (JAIR 72, 2021);
  Angelino et al., CORELS (JMLR 18, 2018); Wang et al., Bayesian rule sets (JMLR 18, 2017); Cohen,
  RIPPER (ICML 1995).
- I3: Bowers et al., Stitch (POPL 2023); Yunis et al., subwords as skills (NeurIPS 2024); Zheng et
  al., PRISE (ICML 2024); Nevill-Manning and Witten, Sequitur (JAIR 7, 1997).
- I4: Kemp, Goodman and Tenenbaum, learning to learn causal models (Cognitive Science 34, 2010);
  Kemp, Perfors and Tenenbaum, overhypotheses (Developmental Science 10, 2007); the infinite
  relational model (AAAI 2006); Sohn et al., MTSGI (UAI 2022), the closest RL work; Diuk, Cohen and
  Littman, object-oriented MDPs (ICML 2008).
- I5: Daw, Niv and Dayan (Nature Neuroscience 8, 2005); Lee, Shimojo and O'Doherty (Neuron 81,
  2014); mixtures of experts (Neural Computation 3, 1991); Neural Production Systems (NeurIPS 2021).
- I6: Maclin and Shavlik, advice-taking learners (MLJ 22, 1996); Taylor and Stone, cross-domain
  transfer (ICML 2007); Torrey et al. (ECML 2005); da Silva et al., agents teaching agents (JAAMAS
  34, 2020).
- I1: Segovia-Aguas, Jimenez and Jonsson, programs for generalized planning (AIJ 272, 2019); LEAPS
  (NeurIPS 2021); learned reward machines (NeurIPS 2019); Neider and Gavran (FMCAD 2018) and
  Scarlet (TACAS 2022).

## 5. Risks

- Gains by construction (RB-B on `latent` worlds, RB-C on schema worlds): the informative parts
  are the harm checks, the oracle bounds and RB-C's version 2, where the classes come from the
  graph.
- RB-A must not reopen X2: it reads X2's logs and adds learners in a separate, frozen protocol;
  X2's verdict stands as published.
- Long runs follow the user's rule: only at night unless the user permits otherwise.
