# Follow-up studies

Four separate questions, each with its own protocol, ledger, configs and
verdict. A study's protocol section is committed before any of its
measurement runs; development runs use a separate namespace and a `dev`
allocation, and are reported as such.

| Study | Question | Environment | Ledger | Configs |
|---|---|---|---|---|
| [R](R_replication.md) | Does the Stage D amortization result (sampling among consistent hypotheses beats the brute-force plan once evidence accumulates) replicate on more worlds and seeds, a second task family and random action failures? | RecipeQuest, goal-only items | `runs/R-ledger.json` | `configs/replication/` |
| [M](M_markov_ranking.md) | Does a pairwise Markov-graph representation with proximity ranking (personalized PageRank) match the hypergraph representation, and which of the components (representation, ranking, sampling) contribute? | RecipeQuest, goal-only items | `runs/M-ledger.json` | `configs/markov/` |
| [U](U_composites_unlocks.md) | When useful actions are not listed (secret sequences of primitive actions) and achievements unlock further actions, what does discovery cost, and does the environment behave as a valid test of it (pooled versus one-at-a-time testing, crossed with composite length, a progress signal and unlock visibility)? | TechTree | `runs/U-ledger.json` | `configs/unlock/` |
| [L](L_layered_discovery.md) | Does using discoveries as atoms of new hypotheses (a node of a higher layer) help beyond remembering them as literal links, and only when later discoveries reuse earlier ones (reuse depth)? | TechTree | `runs/L-ledger.json` | `configs/layers/` |
| [R2](R2_boundary.md) | Which property of the deeper family breaks study R's result: depth, alternatives, pool size, a second item input or the budget? | RecipeQuest, goal-only items | `runs/R2-ledger.json` | `configs/boundary/` |
| [U2](U2_techtree_baselines.md) | How does the pooled explorer compare with generic exploration baselines (iterated width, flat macro expansion, an archive)? | TechTree | `runs/U2-ledger.json` | `configs/unlock2/` |
| [P](P_imperfect_skills.md) | Does promotion still pay when discoveries are executed by imperfect learned skills, and does an admission gate help? | TechTree | `runs/P-ledger.json` | `configs/skills/` |
| [X](X_open_source_baselines.md) | How do open-source reinforcement-learning learners compare with the repository's agents?  | RecipeQuest, TechTree, benchmarks | `runs/X-ledger.json` | `configs/baselines/` |

Each study file has the same sections: question and claim, relation to earlier
work, protocol (frozen), implementation, results, verdict and limits. Studies
U and L share an environment but answer different questions: U is about the
cost of finding unlisted composites and the validity of the test, L about how
a discovery should be represented once found.
