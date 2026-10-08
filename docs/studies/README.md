# Follow-up studies

Four separate questions, each with its own protocol, ledger, configs and
verdict. A study's protocol section is committed before any of its
measurement runs; development runs use a separate namespace and a `dev`
allocation, and are reported as such.

| Study | Question | Environment | Ledger | Configs |
|---|---|---|---|---|
| [R](R_replication.md) | Does the Stage D amortization result (sampling among consistent hypotheses beats the brute-force plan once evidence accumulates) replicate on more worlds and seeds, a second task family and random action failures? | RecipeQuest, goal-only items | `runs/R-ledger.json` | `configs/replication/` |
| [M](M_markov_ranking.md) | Does a pairwise Markov-graph representation with proximity ranking (personalized PageRank) match the hypergraph representation, and which of the components (representation, ranking, sampling) contribute? | RecipeQuest, goal-only items | `runs/M-ledger.json` | `configs/markov/` |
| [U](U_composites_unlocks.md) | Can agents find actions that are not listed (composites of primitive actions) and use what they unlock? | TechTree | `runs/U-ledger.json` | `configs/unlock/` |
| [L](L_layered_discovery.md) | Does promoting a discovered link to a node of a higher layer help, beyond remembering the link, and only when discoveries are compositional? | TechTree | `runs/L-ledger.json` | `configs/layers/` |

Each study file has the same sections: question and claim, relation to earlier
work, protocol (frozen), implementation, results, verdict and limits. Studies
U and L share an environment but answer different questions: U asks whether
unlisted composites can be found at all, L asks how a discovery should be
represented once found.
